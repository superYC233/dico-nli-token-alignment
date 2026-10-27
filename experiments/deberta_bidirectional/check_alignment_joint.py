"""检查原Alignment特征、全量梯度及完整权重重载；不执行优化器更新。"""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_bidirectional.check_bidirectional import ToyTokenizer, toy_frame
from experiments.deberta_bidirectional.data import BidirectionalDataset, REVERSE, model_inputs, write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import (
    AlignmentJointDeberta, build_model, parameter_report, save_weights, load_weights, load_saved_model,
)
from experiments.deberta_bidirectional.evaluation import predict_logits, score_predictions
from experiments.deberta_bidirectional import train_deberta_alignment_joint as cfg


class AlignmentToyTokenizer(ToyTokenizer):
    cls_token_id = 1
    sep_token_id = 2


def toy_backbone():
    torch.manual_seed(42)
    config = DebertaV2Config(
        vocab_size=128, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        relative_attention=True, pos_att_type=['p2c', 'c2p'], share_att_key=True,
        max_relative_positions=16, position_buckets=8, position_biased_input=False,
    )
    return AutoModel.from_config(config)


def check_toy(folder):
    tokenizer = AlignmentToyTokenizer()
    frame = toy_frame()
    dataset = BidirectionalDataset(frame, tokenizer, max_length=16)
    loader = DataLoader(dataset, batch_size=3, collate_fn=cfg.collate_original_units)
    batch = next(iter(loader))
    assert batch['input_ids'].shape == (6, 16)
    assert len(batch['labels']) == len(frame)
    assert torch.bincount(batch['labels'], minlength=4).tolist() == [1, 1, 2, 2]

    model = AlignmentJointDeberta(toy_backbone(), tokenizer, dimension=12).eval()
    original = DebertaDiCoModel(toy_backbone(), tokenizer, feature_size=0, use_lexical_features=False).eval()
    # 同一seed原模块权重保持相同。原末层四分类器则确实被移除了。
    for name in ['align_projection', 'compare_layer']:
        old = getattr(original, name).state_dict()
        for key, value in getattr(model, name).state_dict().items():
            torch.testing.assert_close(value, old[key], rtol=0, atol=0)
    assert not hasattr(model, 'output') and not hasattr(model, 'match_output')
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not any('lora_' in name for name, _ in model.named_parameters())

    # 从真正的原版forward抓特征，核对新模型不是仅保留了Alignment名称。
    captured = {}
    handles = [
        original.dropout.register_forward_hook(lambda module, inputs, output: captured.update(cls=output)),
        original.match_output[1].register_forward_hook(lambda module, inputs, output: captured.update(alignment=output)),
    ]
    with torch.inference_mode():
        original(batch['input_ids'], batch['attention_mask'], None)
        hidden = model.deberta_model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
        ).last_hidden_state
        actual_features = model.extract_features(hidden, batch['input_ids'], batch['attention_mask'])
        expected_features = torch.cat([captured['cls'], captured['alignment']], dim=-1)
        torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    del original

    logits = predict_logits(model, loader, torch.device('cpu'))
    torch.testing.assert_close(logits[0], logits[1, REVERSE], rtol=0, atol=0)
    torch.testing.assert_close(logits[2], logits[3, REVERSE], rtol=0, atol=0)
    torch.testing.assert_close(logits[0], logits[5], rtol=0, atol=0)

    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    outputs = model(**model_inputs(batch, torch.device('cpu')))
    outputs.retain_grad()
    loss = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))(outputs, batch['labels'])
    loss.backward()
    is_n = batch['labels'].eq(3)
    assert (outputs.grad[is_n, 3] < 0).all() and (outputs.grad[~is_n, 3] > 0).all()
    for prefix in ['deberta_model.embeddings.', 'deberta_model.encoder.layer.0.',
                   'deberta_model.encoder.layer.1.', 'align_projection.', 'compare_layer.',
                   'alignment_norm.', 'projection.', 'symmetric_output.', 'direction_weight.']:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for name, parameter in model.named_parameters() if name.startswith(prefix)), prefix
    assert all(torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters())
    for parameter in model.parameters():
        parameter.grad = None

    # 仅修改测试副本，核对encoder和Alignment一起保存；没有优化器或训练。
    with torch.no_grad():
        model.deberta_model.embeddings.word_embeddings.weight[1, 0].add_(.01)
        model.align_projection.weight[0, 0].add_(.01)
    model.eval()
    expected = predict_logits(model, loader, torch.device('cpu'))
    save_weights(model, folder / 'full_toy.pth')
    restored = AlignmentJointDeberta(toy_backbone(), tokenizer, dimension=12).eval()
    load_weights(restored, folder / 'full_toy.pth')
    torch.testing.assert_close(expected, predict_logits(restored, loader, torch.device('cpu')), rtol=0, atol=0)
    write_train_reference(frame, folder / 'toy_reference.csv')
    score_predictions(expected, frame, folder / 'toy_scores', folder / 'toy_reference.csv')

    hashes = cfg.source_hashes()
    assert hashes and all((ROOT / name).is_file() for name in hashes)
    summary = pd.DataFrame({'method': ['independent', 'pair_aware'], 'weighted_f1': [.84, .93]})
    assert cfg.select_epoch(summary, 3)['selection_score'] == .93
    print('CPU检查通过：原版Alignment特征逐值相同、全量梯度、四类N竞争、交换关系、完整重载、官方评分和启动源码。')


def check_real(folder):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=cfg.BASE_REVISION, local_files_only=True,
    )
    frame = toy_frame().iloc[[0, 1, 4]].copy()
    dataset = BidirectionalDataset(frame, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=2, collate_fn=cfg.collate_original_units)
    model = build_model(cfg.MODEL_NAME, cfg.BASE_REVISION, tokenizer, cfg.DIMENSION, cfg.HEAD_DROPOUT)
    report = pd.DataFrame(parameter_report(model))
    print(report.groupby(['group', 'trainable']).parameters.sum().to_string())
    assert report.trainable.all()
    batch = next(iter(loader))
    assert batch['input_ids'].shape[1] == cfg.MAX_LENGTH

    model.train()
    logits = model(**model_inputs(batch, torch.device('cpu')))
    nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))(logits, batch['labels']).backward()
    assert model.deberta_model.embeddings.word_embeddings.weight.grad is not None
    assert model.align_projection.weight.grad is not None
    for parameter in model.parameters():
        parameter.grad = None
    del logits
    with torch.no_grad():
        model.deberta_model.embeddings.word_embeddings.weight[1, 0].add_(.001)
        model.align_projection.weight[0, 0].add_(.001)
    model.eval()
    expected = predict_logits(model, loader, torch.device('cpu'))
    torch.testing.assert_close(expected[0], expected[1, REVERSE], rtol=0, atol=0)

    # 单条调用不提供标签或pair_id，也不需要数据集里存在反向行。
    single = BidirectionalDataset(frame.iloc[[0]][['text1', 'text2']], tokenizer, cfg.MAX_LENGTH)
    single_logits = predict_logits(model, DataLoader(single, batch_size=1, collate_fn=cfg.collate_original_units), torch.device('cpu'))
    torch.testing.assert_close(single_logits[0], expected[0], rtol=1e-4, atol=1e-5)

    model_dir = folder / 'real'
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
              'dimension': cfg.DIMENSION, 'head_dropout': cfg.HEAD_DROPOUT}
    (model_dir / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
    save_weights(model, model_dir / 'best_model.pth')
    del model, batch
    restored, _, _ = load_saved_model(model_dir, torch.device('cpu'))
    actual = predict_logits(restored, loader, torch.device('cpu'))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print('真实NLI模型检查通过：全量反向、固定128长度、单条预测、完整encoder+Alignment+联合头重载；没有参数更新。')


def main(real=False):
    torch.set_num_threads(4)
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'alignment_joint_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_toy(folder)
        if real:
            check_real(folder)
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('alignment_joint_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
