"""CPU检查架构、Row CE梯度和保存重载；不进行任何优化器参数更新。"""
import argparse
import copy
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from project_paths import get_model_cache_dir
from experiments.deberta_manual_aug import train_relation_row as cfg
from experiments.deberta_manual_aug.relation_lora_model import (
    add_lora, save_trainable_weights, load_trainable_weights,
)
from experiments.deberta_manual_aug.relation_row_model import DebertaRowModel, build_model, load_saved_model


class ToyTokenizer:
    cls_token_id = 1
    sep_token_id = 2


class GradientRecorder(torch.optim.SGD):
    def step(self, closure=None):
        # 只记录梯度，不调用真正的参数更新。
        self.recorded = [None if p.grad is None else p.grad.detach().clone()
                         for group in self.param_groups for p in group['params']]


def toy_model(head_type):
    torch.manual_seed(cfg.SEED)
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
        share_att_key=True, max_relative_positions=16, position_buckets=8,
        position_biased_input=False,
    )
    model = DebertaRowModel(AutoModel.from_config(config), ToyTokenizer(), head_type)
    return add_lora(model, dict(cfg.lora_settings(), r=4, lora_alpha=8))


def check_toy(folder):
    cls = toy_model('cls')
    cls_rng = torch.get_rng_state().clone()
    alignment = toy_model('alignment')
    assert torch.equal(cls_rng, torch.get_rng_state())
    assert cfg.shared_initialization_hash(cls) == cfg.shared_initialization_hash(alignment)
    assert not any(hasattr(cls, name) for name in ['align_projection', 'compare_layer', 'match_output'])
    aligned_state = alignment.state_dict()
    for name, value in cls.state_dict().items():
        torch.testing.assert_close(value, aligned_state[name], rtol=0, atol=0)

    ids = torch.tensor([
        [1, 5, 6, 2, 7, 2, 0, 0], [1, 7, 2, 5, 6, 2, 0, 0],
        [1, 8, 9, 2, 10, 2, 0, 0], [1, 10, 2, 8, 9, 2, 0, 0],
        [1, 11, 2, 12, 2, 0, 0, 0], [1, 13, 2, 14, 2, 0, 0, 0],
    ])
    batch = {'input_ids': ids, 'attention_mask': ids.ne(0).long(),
             'labels': torch.tensor([0, 1, 2, 2, 3, 3]), 'source_rows': torch.arange(6),
             # 故意给无效Pair目标：Row-only训练不应读取这些字段。
             'pair_indices': torch.tensor([[1000, 1001]]), 'pair_targets': torch.tensor([999])}
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    for model in [cls, alignment]:
        model.train()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert all(not p.requires_grad for name, p in model.deberta_model.named_parameters()
                   if 'lora_A.' not in name and 'lora_B.' not in name)
        optimizer = GradientRecorder(parameters, lr=cfg.LEARNING_RATE)
        rng = torch.get_rng_state().clone()
        loss = criterion(model(ids, batch['attention_mask'], None), batch['labels'])
        expected = torch.autograd.grad(loss, parameters, allow_unused=True)
        torch.set_rng_state(rng)
        totals, plan = cfg.train_epoch(model, [batch], optimizer, criterion, torch.device('cpu'), 1)
        for actual, wanted in zip(optimizer.recorded, expected):
            if wanted is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
        assert totals['row_ce_rows'] == 6 and totals['pair_ce_groups'] == 0
        assert plan[0]['source_rows'] == '0|1|2|3|4|5'
        assert all(torch.equal(p.detach(), before[name]) for name, p in model.named_parameters())

        # 人为设成非零，仅检验保存完整性，不是训练。
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if 'lora_B.' in name:
                        parameter.copy_(torch.linspace(-.01, .01, parameter.numel()).reshape_as(parameter))
                    elif name.startswith(('output.', 'match_output.')):
                        parameter.add_(.001)
        model.eval()
        with torch.no_grad():
            expected_logits = model(ids, batch['attention_mask'], None)
        path = folder / f'{model.head_type}.pth'
        save_trainable_weights(model, path)
        restored = toy_model(model.head_type).eval()
        load_trainable_weights(restored, path)
        with torch.no_grad():
            actual_logits = restored(ids, batch['attention_mask'], None)
        torch.testing.assert_close(expected_logits, actual_logits, rtol=0, atol=0)
    print('CPU检查通过：共享初值相同、CLS真正删对齐、仅Row CE梯度、冻结范围、非零权重精确重载。')


def check_real(folder):
    reference = json.loads((cfg.MIXED_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=reference['base_revision'], local_files_only=True,
    )
    hashes = []
    ids = torch.tensor([[1, 5, 6, 2, 7, 2], [1, 7, 2, 5, 6, 2]])
    for head_type in ['cls', 'alignment']:
        cfg.set_seed(cfg.SEED)
        model = build_model(cfg.MODEL_NAME, tokenizer, head_type, cfg.lora_settings(), reference['base_revision'])
        hashes.append(cfg.shared_initialization_hash(model))
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('真实模型：', head_type, '；可训练参数：', count)
        model.eval()
        with torch.no_grad():
            expected = model(ids, torch.ones_like(ids), None)
        model_dir = folder / head_type
        model_dir.mkdir()
        tokenizer.save_pretrained(model_dir / 'tokenizer')
        config = {'model_name': cfg.MODEL_NAME, 'head_type': head_type,
                  'base_revision': reference['base_revision'], 'lora': cfg.lora_settings()}
        (model_dir / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
        save_trainable_weights(model, model_dir / 'best_model.pth')
        del model
        restored, _, _ = load_saved_model(model_dir, torch.device('cpu'))
        with torch.no_grad():
            actual = restored(ids, torch.ones_like(ids), None)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        del restored
    assert hashes[0] == hashes[1]
    print('真实NLI DeBERTa检查通过：两组共享初值一致，专用加载器重载后输出一致；未创建优化器。')


def main(real=False):
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'row_architecture_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_toy(folder)
        if real:
            check_real(folder)
    finally:
        # 仅清理本次创建的独立检查目录，先核对绝对路径。
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('row_architecture_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
