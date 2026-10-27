"""CPU检查：LoRA注入、冻结范围、梯度及小模型保存重载；不进行优化器更新。"""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_manual_aug import train_relation_lora as cfg
from experiments.deberta_manual_aug.row_priority_gradients import separate_gradients
from experiments.deberta_manual_aug.relation_lora_model import (
    add_lora, build_model, parameter_report, save_trainable_weights, load_trainable_weights,
    load_saved_model,
)


class ToyTokenizer:
    cls_token_id = 1
    sep_token_id = 2


def check_freezing(model):
    report = pd.DataFrame(parameter_report(model))
    assert not report.loc[report.group.eq('original_encoder'), 'trainable'].any()
    assert report.loc[report.group.isin(['lora', 'task_heads']), 'trainable'].all()
    modules = [name for name, module in model.deberta_model.named_modules() if hasattr(module, 'lora_A')]
    assert len(modules) == 2 * model.deberta_model.config.num_hidden_layers
    assert all(name.endswith(('.query_proj', '.value_proj')) for name in modules)
    return report


def check_gradients(model, batch):
    versions = [parameter._version for parameter in model.parameters()]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    logits = model(batch['input_ids'], batch['attention_mask'], None)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    row_loss, pair_loss = cfg.compute_losses(logits, batch, row_criterion, pair_criterion)
    row, pair = separate_gradients(row_loss, pair_loss, [p for _, p in named], cfg.PAIR_WEIGHT)
    assert any('lora_B.' in name and gradient is not None and gradient.abs().sum() > 0
               for (name, _), gradient in zip(named, row))
    assert any(name.startswith('output.') and gradient is not None and gradient.abs().sum() > 0
               for (name, _), gradient in zip(named, row))
    assert all(g is None or torch.isfinite(g).all() for g in row + pair)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert versions == [parameter._version for parameter in model.parameters()]


def check_toy():
    torch.manual_seed(42)
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
        share_att_key=True, max_relative_positions=16, position_buckets=8,
        position_biased_input=False,
    )
    backbone = AutoModel.from_config(config)
    clean_backbone = copy.deepcopy(backbone.state_dict())
    model = DebertaDiCoModel(backbone, ToyTokenizer(), feature_size=0, use_lexical_features=False).eval()
    ids = torch.tensor([
        [1, 5, 6, 2, 7, 2, 0, 0], [1, 7, 2, 5, 6, 2, 0, 0],
        [1, 8, 9, 2, 10, 2, 0, 0], [1, 10, 2, 8, 9, 2, 0, 0],
        [1, 11, 2, 12, 2, 0, 0, 0], [1, 13, 2, 14, 2, 0, 0, 0],
    ])
    mask = ids.ne(0).long()
    batch = {'input_ids': ids, 'attention_mask': mask,
             'labels': torch.tensor([0, 1, 2, 2, 3, 3]),
             'pair_indices': torch.tensor([[0, 1], [2, 3]]), 'pair_targets': torch.tensor([0, 2])}
    settings = dict(cfg.lora_settings(), r=4, lora_alpha=8)
    with torch.no_grad():
        before = model(ids, mask, None)
    rng = torch.get_rng_state().clone()
    add_lora(model, settings)
    assert torch.equal(rng, torch.get_rng_state())
    model.eval()
    with torch.no_grad():
        after = model(ids, mask, None)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    check_freezing(model)
    check_gradients(model, batch)

    # 人为填入非零小数，只为检验保存内容，不是训练或优化器更新。
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B.' in name or not name.startswith('deberta_model.'):
                parameter.add_(torch.randn_like(parameter) * 0.001)
        expected = model(ids, mask, None)
    other_backbone = AutoModel.from_config(copy.deepcopy(config))
    other_backbone.load_state_dict(clean_backbone, strict=True)
    other = DebertaDiCoModel(other_backbone, ToyTokenizer(), feature_size=0, use_lexical_features=False)
    add_lora(other, settings)
    with tempfile.TemporaryDirectory(prefix='dico_lora_check_', dir=ROOT / 'artifacts') as temporary:
        assert Path(temporary).resolve().is_relative_to((ROOT / 'artifacts').resolve())
        path = Path(temporary) / 'best_model.pth'
        save_trainable_weights(model, path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert any('lora_B.' in name for name in state)
        assert all(name in state for name, _ in model.named_parameters() if not name.startswith('deberta_model.'))
        assert not any('base_layer.weight' in name for name in state)
        load_trainable_weights(other, path)
    other.eval()
    with torch.no_grad():
        restored = other(ids, mask, None)
    torch.testing.assert_close(expected, restored, rtol=0, atol=0)
    print('小模型检查通过：初始输出不变、RNG不变、冻结/梯度正确、非零LoRA和全部任务头精确重载。')


def check_real():
    cfg.set_seed(cfg.SEED)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    model = build_model(cfg.MODEL_NAME, tokenizer, False, cfg.lora_settings()).eval()
    train, _, _ = cfg.read_train()
    dataset = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    group = next(dataset[index] for index in range(len(dataset)) if dataset.groups[index][1] >= 0)
    batch = collate_groups([group])
    with torch.no_grad():
        before = model(batch['input_ids'], batch['attention_mask'], None)
    rng = torch.get_rng_state().clone()
    add_lora(model, cfg.lora_settings())
    assert torch.equal(rng, torch.get_rng_state())
    model.eval()
    with torch.no_grad():
        after = model(batch['input_ids'], batch['attention_mask'], None)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    report = check_freezing(model)
    check_gradients(model, batch)
    # 检查实际评估入口使用的“原NLI缓存+配置+小权重+tokenizer”完整加载链。
    with tempfile.TemporaryDirectory(prefix='dico_lora_check_', dir=ROOT / 'artifacts') as temporary:
        model_dir = Path(temporary)
        assert model_dir.resolve().is_relative_to((ROOT / 'artifacts').resolve())
        tokenizer.save_pretrained(model_dir / 'tokenizer')
        saved_config = {'model_name': cfg.MODEL_NAME, 'use_lora': True,
                        'lora': cfg.lora_settings(), 'base_revision': model.deberta_model.config._commit_hash}
        (model_dir / 'run_config.json').write_text(json.dumps(saved_config), encoding='utf-8')
        save_trainable_weights(model, model_dir / 'best_model.pth')
        restored, restored_tokenizer, _ = load_saved_model(model_dir, torch.device('cpu'))
        assert restored_tokenizer.cls_token_id == tokenizer.cls_token_id
        with torch.no_grad():
            restored_logits = restored(batch['input_ids'], batch['attention_mask'], None)
        torch.testing.assert_close(after, restored_logits, rtol=0, atol=0)
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())
    print('真实NLI DeBERTa CPU检查通过：初始输出一致、两项损失反向正常、完整保存/加载一致、未运行优化器。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='额外读取本地NLI DeBERTa，在CPU检查前向与梯度')
    args = parser.parse_args()
    check_toy()
    if args.real:
        check_real()
