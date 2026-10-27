"""CPU检查数据/梯度/重载；--gpu额外检查真实主干，不执行参数更新。"""
import argparse
import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel as OriginalModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed
from experiments.deberta_local_semantics import config as cfg
from experiments.deberta_local_semantics.prepare_data import prepare
from experiments.deberta_local_semantics.data import LocalRowDataset
from experiments.deberta_local_semantics.model import DebertaLocalModel, build_model
from experiments.deberta_local_semantics.train_deberta_local_semantics import row_objective, build_optimizer
from experiments.deberta_local_semantics.evaluation import evaluate


def check_data():
    train, annotations, diagnostic, report = prepare()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True, use_fast=True,
    )
    rows = LocalRowDataset(train, tokenizer, annotations, cfg.MAX_LENGTH)
    old_rows = RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    assert torch.equal(rows.encoded['input_ids'], old_rows.encoded['input_ids'])
    assert torch.equal(rows.encoded['attention_mask'], old_rows.encoded['attention_mask'])
    assert torch.equal(rows.labels, old_rows.labels)
    assert RealPairDataset(rows).pairs == RealPairDataset(old_rows).pairs
    assert len(RealPairDataset(rows)) == 1380
    active = rows.local_targets.ge(0)
    assert active.sum() == 64
    assert rows.span1_masks[active].any(dim=1).all()
    assert rows.span2_masks[active].any(dim=1).all()
    assert not rows.span1_masks[~active].any() and not rows.span2_masks[~active].any()
    for index in torch.where(active)[0].tolist():
        ids = rows.encoded['input_ids'][index]
        separators = ids.eq(tokenizer.sep_token_id).long().cumsum(dim=0)
        assert separators[rows.span1_masks[index]].eq(0).all()
        assert separators[rows.span2_masks[index]].eq(1).all()
    assert set(diagnostic.instance_id).isdisjoint(train.loc[active.numpy(), 'instance_id'])
    print('数据通过：3290行/1380组；64行局部监督；16行仅留出局部标签；token输入与原版完全一致。')
    return train, annotations, rows, tokenizer


def check_model():
    tiny = DebertaV2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=128,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    backbone = DebertaV2Model(tiny)
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    set_seed(42)
    original = OriginalModel(copy.deepcopy(backbone), tokenizer, feature_size=0, use_lexical_features=False)
    reference_rng = torch.get_rng_state().clone()
    set_seed(42)
    current = DebertaLocalModel(copy.deepcopy(backbone), tokenizer)
    assert torch.equal(torch.get_rng_state(), reference_rng)
    current_state = current.state_dict()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, current_state[name], rtol=0, atol=0)

    ids = torch.tensor([[1, 4, 5, 2, 6, 7, 2, 0], [1, 6, 7, 2, 4, 5, 2, 0]])
    mask = ids.ne(0).long()
    span1 = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool)
    span2 = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]] * 2, dtype=torch.bool)
    batch = {'input_ids': ids, 'attention_mask': mask, 'label': torch.tensor([0, 1]),
             'local_target': torch.tensor([0, 1]), 'span1_mask': span1, 'span2_mask': span2}
    original.eval()
    current.eval()
    with torch.inference_mode():
        torch.testing.assert_close(current(ids, mask), original(ids, mask, None), rtol=0, atol=0)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    local_ce = nn.CrossEntropyLoss()
    original.train()
    current.train()
    before = {name: parameter.detach().clone() for name, parameter in current.named_parameters()}
    rng = torch.get_rng_state()
    old_loss = row_ce(original(ids, mask, None), batch['label'])
    old_loss.backward()
    old_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    loss, *_ = row_objective(current, batch, row_ce, local_ce, torch.device('cpu'), 0.0)
    loss.backward()
    torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), old_rng)
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(current.named_parameters())[name].grad, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in current.local_relation_head.parameters())
    print('零权重通过：原参数/预测/Row梯度/随机数状态与原版完全一致。')

    current.zero_grad(set_to_none=True)
    calls = []
    handle = current.deberta_model.register_forward_hook(lambda *args: calls.append(1))
    loss, _, local_loss, count, _ = row_objective(
        current, batch, row_ce, local_ce, torch.device('cpu'), cfg.LOCAL_WEIGHT,
    )
    loss.backward()
    handle.remove()
    assert len(calls) == 1 and count == 2 and local_loss.item() > 0
    for name in ['align_projection.weight', 'local_relation_head.1.weight',
                 'local_relation_head.3.weight', 'deberta_model.embeddings.word_embeddings.weight']:
        gradient = dict(current.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    current.zero_grad(set_to_none=True)
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    pair_loss = cfg.PAIR_WEIGHT * pair_ce(
        get_pair_logits(current(ids[:1], mask[:1]), current(ids[1:], mask[1:])), torch.tensor([0]),
    )
    pair_loss.backward()
    assert all(parameter.grad is None for parameter in current.local_relation_head.parameters())
    assert all(torch.equal(parameter, before[name]) for name, parameter in current.named_parameters())
    print('梯度通过：局部监督复用一次前向并反传到主干；Pair阶段不调用局部头；未更新参数。')

    optimizer = build_optimizer(current)
    first_ids = {id(parameter) for parameter in optimizer.param_groups[0]['params']}
    local_ids = {id(parameter) for parameter in current.local_relation_head.parameters()}
    assert first_ids.isdisjoint(local_ids)
    assert first_ids | local_ids == {id(parameter) for parameter in current.parameters()}
    assert optimizer.param_groups[0]['lr'] == cfg.LEARNING_RATE
    assert optimizer.param_groups[1]['lr'] == cfg.LOCAL_LEARNING_RATE
    assert not optimizer.state
    buffer = io.BytesIO()
    torch.save(current.state_dict(), buffer)
    buffer.seek(0)
    restored = DebertaLocalModel(copy.deepcopy(backbone), tokenizer)
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    current.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, mask), current(ids, mask), rtol=0, atol=0)
    print('保存/重载通过；主干2e-5、局部头3e-4；没有optimizer.step。')


def check_scoring():
    history = pd.read_csv(cfg.REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).set_index('instance_id').loc[history.instance_id].reset_index()
    logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)

    class HistoricalModel(nn.Module):
        def forward(self, input_ids, attention_mask, lexical_features=None):
            return logits[input_ids[:, 0]]

    batches = [{'input_ids': torch.tensor([index]), 'attention_mask': torch.ones(1, dtype=torch.long)}
               for index in range(len(dev))]
    output_dir = cfg.HERE / 'outputs/_checks/historical_scoring'
    model = HistoricalModel().train()
    rng = torch.get_rng_state().clone()
    summary = evaluate(model, DataLoader(batches, batch_size=16), dev, torch.device('cpu'), output_dir)
    assert torch.equal(torch.get_rng_state(), rng) and model.training
    scores = summary.set_index('method').weighted_f1
    assert abs(scores['pair_aware'] - 0.9381451572973859) < 1e-12
    assert abs(scores['independent'] - 0.5471771947703606) < 1e-12
    actual = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    assert actual.prediction.equals(history.prediction) and actual.pair_prediction.equals(history.pair_prediction)
    print('评分通过：旧0.938的660行独立/结构预测完全复现，单例不改答案。')


def check_gpu(train, annotations, rows, tokenizer):
    set_seed(cfg.SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device).train()
    active_rows = torch.where(rows.local_targets.ge(0))[0].tolist()[:16]
    batch = next(iter(DataLoader(torch.utils.data.Subset(rows, active_rows), batch_size=16)))
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    local_ce = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats()
    original_classifier = model.output.weight.detach().clone()
    loss, _, local_loss, count, _ = row_objective(model, batch, row_ce, local_ce, device, cfg.LOCAL_WEIGHT)
    loss.backward()
    assert count == 16 and torch.isfinite(loss) and torch.isfinite(local_loss)
    for name in ['deberta_model.embeddings.word_embeddings.weight',
                 'align_projection.weight', 'local_relation_head.3.weight']:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum() > 0 and torch.isfinite(gradient).all(), name
    model.zero_grad(set_to_none=True)
    pair_batch = next(iter(DataLoader(RealPairDataset(rows), batch_size=cfg.PAIR_BATCH_SIZE)))
    ids0, mask0, ids1, mask1, target = pair_batch
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
                                  label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    pair_loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(
        model(ids0.to(device), mask0.to(device)), model(ids1.to(device), mask1.to(device)),
    ), target.to(device))
    pair_loss.backward()
    assert torch.isfinite(pair_loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert all(parameter.grad is None for parameter in model.local_relation_head.parameters())
    assert torch.equal(model.output.weight, original_classifier)
    result = {
        'row_plus_local_loss': float(loss.detach()), 'local_loss': float(local_loss.detach()),
        'weighted_pair_loss': float(pair_loss.detach()),
        'peak_allocated_GiB_without_Adam_states': torch.cuda.max_memory_allocated() / 1024**3,
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'optimizer_steps': 0,
    }
    output_dir = cfg.HERE / 'outputs/_checks'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'gpu_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('真实GPU前后向通过：', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, annotations, rows, tokenizer = check_data()
    check_model()
    check_scoring()
    if args.gpu:
        check_gpu(train, annotations, rows, tokenizer)
    print('检查完成；没有启动训练或更新模型参数。')
