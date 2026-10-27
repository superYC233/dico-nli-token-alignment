"""检查损失拆分、梯度方向和真实数据曝光；不执行优化器更新。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.deberta_manual_aug import train_relation_pair_gate as cfg
from experiments.deberta_bidirectional.check_original_staged_replay import ToyRows, toy_model, GradientRecorder


def criteria(device='cpu'):
    pair = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
                               label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    gate = nn.CrossEntropyLoss(weight=torch.tensor(cfg.GATE_CLASS_WEIGHTS, device=device))
    return pair, gate


def check_objective():
    torch.manual_seed(9)
    logits = torch.randn(9, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 3, 3, 3])
    targets = torch.tensor([0, 1, 2])
    pair_criterion, gate_criterion = criteria()
    loss, pair_loss, gate_loss = cfg.compute_losses(logits, labels, targets, pair_criterion, gate_criterion)
    torch.testing.assert_close(loss, .5 * pair_loss + gate_loss)
    p = logits.softmax(1)
    collapsed = cfg.n_gate_logits(logits).softmax(1)
    torch.testing.assert_close(collapsed, torch.stack([p[:, :3].sum(1), p[:, 3]], dim=1))
    pair_grad = torch.autograd.grad(pair_loss, logits, retain_graph=True)[0]
    gate_grad = torch.autograd.grad(gate_loss, logits, retain_graph=True)[0]
    assert pair_grad[:, 3].eq(0).all() and pair_grad[6:].eq(0).all()
    assert gate_grad[labels.ne(3), 3].gt(0).all()
    assert gate_grad[labels.eq(3), 3].lt(0).all()
    zero, _, _ = cfg.compute_losses(logits, labels, targets, pair_criterion, gate_criterion, gate_weight=0)
    torch.testing.assert_close(zero, .5 * pair_loss)

    # 完整四类CE = N/非N项 + 非N行上的条件FBE项（保持同一个加权分母）。
    weights = torch.tensor(cfg.ROW_WEIGHTS)[labels]
    row_terms = F.cross_entropy(logits, labels, reduction='none')
    gate_terms = F.cross_entropy(cfg.n_gate_logits(logits), labels.eq(3).long(), reduction='none')
    conditional = torch.zeros_like(gate_terms)
    mask = labels.ne(3)
    conditional[mask] = F.cross_entropy(logits[mask, :3], labels[mask], reduction='none')
    torch.testing.assert_close(row_terms, gate_terms + conditional)
    expected = (weights * (gate_terms + conditional)).sum() / weights.sum()
    torch.testing.assert_close(nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))(logits, labels), expected)

    # FBE质量重新分配但总量不变：Gate不能暗中重复完整四分类。
    first = torch.tensor([[3., 0., 0., -1.], [0., 3., 0., -1.]])
    second = first[:, [1, 0, 2, 3]]
    target = torch.tensor([0, 1])
    torch.testing.assert_close(cfg.n_gate_logits(first), cfg.n_gate_logits(second))
    assert F.cross_entropy(first, target) < F.cross_entropy(second, target)
    extreme = torch.tensor([[1000., -1000., 0., 900.], [-1000., -1000., -1000., 1000.]], requires_grad=True)
    stable = gate_criterion(cfg.n_gate_logits(extreme), torch.tensor([0, 1]))
    stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()
    print('损失检查通过：分组概率正确；非N压低N、N提高N；与完整四分类不同；极值有限。')


def check_coverage(train):
    rows = ToyRows(train)
    pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(pairs.pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(12))
    batches = list(cfg.pair_gate_batches(rows, loader, torch.Generator().manual_seed(13)))
    seen = torch.cat([batch['source_rows'] for batch in batches])
    assert len(rows) == 3290 and len(pairs) == 1380 and len(batches) == 87
    assert seen.sort().values.equal(torch.arange(len(rows)))
    assert sum(batch['n_count'] for batch in batches) == 530
    assert max(len(batch['labels']) for batch in batches) == 39
    choices = torch.tensor([[0, 1], [1, 0], [2, 2]])
    for batch in batches:
        count = batch['pair_count']
        expected = choices[batch['pair_targets']]
        assert batch['labels'][:count].equal(expected[:, 0])
        assert batch['labels'][count:2 * count].equal(expected[:, 1])
        assert batch['labels'][2 * count:].eq(3).all()
    print('曝光检查通过：3290行第二阶段各一次，1380个Pair、530条N；87步、最多39行/批。')


def check_loop(train):
    full = ToyRows(train)
    pairs = cfg.RealPairDataset(full)
    selected = [index for pair in pairs.pairs[:2] for index in pair[:2]]
    selected += torch.where(full.labels.eq(3))[0][:3].tolist()
    rows = ToyRows(train.iloc[selected])
    small_pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(small_pairs.pairs, batch_size=1, shuffle=False,
                        generator=torch.Generator().manual_seed(3))
    model = toy_model()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    recorder = GradientRecorder(model)
    pair_criterion, gate_criterion = criteria()
    record, exposure = cfg.train_pair_gate(model, rows, loader, torch.Generator().manual_seed(4),
                                           recorder, pair_criterion, gate_criterion, torch.device('cpu'))
    assert record['gate_rows'] == 7 and record['gate_n_rows'] == 3 and record['pair_ce_groups'] == 2
    assert len(exposure) == len(recorder.gradients) == 2
    assert all(torch.isfinite(grad).all() for item in recorder.gradients for grad in item.values())
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    print('循环检查通过：同一次反向含两项损失；梯度有限；真实参数更新0次。')


def check_real(train):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=cfg.get_model_cache_dir(),
                                             revision=cfg.BASE_REVISION, local_files_only=True)
    rows = cfg.RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(pairs.pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=False)
    batch = next(cfg.pair_gate_batches(rows, loader, torch.Generator().manual_seed(4)))
    device = torch.device('cuda')
    model = cfg.build_model(tokenizer).to(device).train()
    model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    original = model.output.weight.detach().clone()
    pair_criterion, gate_criterion = criteria(device)
    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
    loss, pair_loss, gate_loss = cfg.compute_losses(logits, batch['labels'].to(device),
                                                  batch['pair_targets'].to(device), pair_criterion, gate_criterion)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.output.weight.grad[3].abs().sum() > 0
    assert model.deberta_model.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert torch.equal(original, model.output.weight)
    print('真实GPU检查通过：行数', len(batch['labels']), '；Pair / Gate loss：', pair_loss.item(), '/', gate_loss.item())
    print('参数：', sum(p.numel() for p in model.parameters()), '；真实优化器更新0次；没有保存模型。')
    print('峰值显存（不含Adam状态）：', round(torch.cuda.max_memory_allocated() / 1024**3, 3), 'GiB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    train, _, _ = cfg.read_data()
    check_objective()
    check_coverage(train)
    check_loop(train)
    if args.real:
        check_real(train)
