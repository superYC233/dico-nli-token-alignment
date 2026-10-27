"""公式、旧CE梯度、数据覆盖和可选GPU前后向检查；不更新模型参数。"""
import argparse
import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_pair_hinge import config as cfg
from experiments.deberta_pair_hinge.losses import hinge16, compute_joint_loss, three_combination_scores
from experiments.deberta_pair_hinge.train_deberta_pair_hinge import build_model, build_criteria, train_epoch
from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical.evaluation import pair_aware
from experiments.deberta_stage_average.train_deberta_stage_average import read_data
from experiments.deberta_manual_aug import train_relation_manual as old


def check_hinge_formula():
    generator = torch.Generator().manual_seed(71)
    first = torch.randn(32, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    second = torch.randn(32, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    labels0 = torch.arange(32) % 4
    labels1 = torch.arange(32) // 4 % 4
    actual = hinge16(first, second, labels0, labels1)
    expected = []
    # 独立逐组穷举15个负例，检查FF到NN以及金标索引。
    for i in range(len(first)):
        a, b = labels0[i].item(), labels1[i].item()
        negatives = [first[i, x] + second[i, y]
                     for x, y in itertools.product(range(4), repeat=2) if (x, y) != (a, b)]
        expected.append(F.relu(1.0 + torch.stack(negatives).max() - first[i, a] - second[i, b]))
    torch.testing.assert_close(actual, torch.stack(expected))
    actual_grad = torch.autograd.grad(actual.sum(), (first, second), retain_graph=True)
    expected_grad = torch.autograd.grad(torch.stack(expected).sum(), (first, second))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)

    # 两行都误判N：NN必须参与竞争，梯度必须压N、抬金标。
    first = torch.tensor([[1., -3., -4., 2.]], requires_grad=True)
    second = torch.tensor([[-3., 1., -4., 2.]], requires_grad=True)
    loss = hinge16(first, second, torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(loss, torch.tensor([3.]))
    loss.sum().backward()
    torch.testing.assert_close(first.grad, torch.tensor([[-1., 0., 0., 1.]]))
    torch.testing.assert_close(second.grad, torch.tensor([[0., -1., 0., 1.]]))

    # 一行间隔不足，另一行很稳：只推动较弱那行。
    first = torch.tensor([[.2, -5., -5., .1]], requires_grad=True)
    second = torch.tensor([[-5., .3, -5., -5.]], requires_grad=True)
    loss = hinge16(first, second, torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(loss, torch.tensor([.9]))
    loss.sum().backward()
    torch.testing.assert_close(second.grad, torch.zeros_like(second))
    assert first.grad[0, 0] < 0 and first.grad[0, 3] > 0

    # F/B/E共同下移：CE3看不见，hinge能看见。
    first = torch.tensor([[4., 0., 0., 1.]])
    second = torch.tensor([[0., 4., 0., 1.]])
    shift = torch.tensor([[-4., -4., -4., 0.]])
    before = three_combination_scores(first, second)
    after = three_combination_scores(first + shift, second + shift)
    torch.testing.assert_close(before.softmax(1), after.softmax(1))
    y0, y1 = torch.tensor([0]), torch.tensor([1])
    assert hinge16(first, second, y0, y1).item() == 0
    assert hinge16(first + shift, second + shift, y0, y1).item() == 3
    print('hinge通过：16组合、排除金标、含N梯度、较弱行margin、打破F/B/E共同漂移。')


def check_ce_and_singletons():
    row_ce, pair_ce = build_criteria(torch.device('cpu'))
    logits = torch.randn(8, 4, generator=torch.Generator().manual_seed(32), requires_grad=True)
    labels = torch.tensor([0, 1, 1, 0, 2, 2, 3, 3])
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    targets = torch.tensor([0, 1, 2])
    new = compute_joint_loss(logits, labels, pairs, targets, row_ce, pair_ce, pair_loss='ce3')
    original = old.compute_joint_loss(logits, labels, pairs, targets, row_ce, pair_ce)
    for a, b in zip(new, original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    g0 = torch.autograd.grad(new[0], logits, retain_graph=True)[0]
    g1 = torch.autograd.grad(original[0], logits, retain_graph=True)[0]
    torch.testing.assert_close(g0, g1, rtol=0, atol=0)

    joint = compute_joint_loss(logits, labels, pairs, targets, row_ce, pair_ce,
                              margin=cfg.HINGE_MARGIN, pair_weight=cfg.PAIR_WEIGHT)
    row_grad = torch.autograd.grad(joint[1], logits, retain_graph=True)[0]
    pair_grad = torch.autograd.grad(joint[2], logits, retain_graph=True)[0]
    total_grad = torch.autograd.grad(joint[0], logits)[0]
    torch.testing.assert_close(total_grad, row_grad + cfg.PAIR_WEIGHT * pair_grad)
    assert torch.equal(pair_grad[6:], torch.zeros_like(pair_grad[6:]))
    assert (row_grad[6:, 3] < 0).all(), 'N单例必须有抬高N的Row正监督。'

    only_n = torch.zeros(2, 4, requires_grad=True)
    loss, row, pair = compute_joint_loss(
        only_n, torch.tensor([3, 3]), torch.empty(0, 2, dtype=torch.long),
        torch.empty(0, dtype=torch.long), row_ce, pair_ce,
    )
    loss.backward()
    assert pair.item() == 0 and torch.isfinite(only_n.grad).all()
    assert (only_n.grad[:, 3] < 0).all()
    print('CE对照通过：旧CE数值和梯度完全相同；0.5仅乘一次；N单例/全N批次正常。')


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 5)
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(5, 4)

    def forward(self, input_ids, attention_mask, features):
        values = self.embedding(input_ids).mean(dim=1)
        return self.output(self.dropout(values))


class GradientRecorder:
    """形状像optimizer，但step只记录梯度，绝不修改参数。"""
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.gradients.append([p.grad.detach().clone() for p in self.model.parameters()])


def check_training_path():
    model = TinyModel()
    reference = copy.deepcopy(model)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    row_ce, pair_ce = build_criteria(torch.device('cpu'))
    batch = {
        'input_ids': torch.arange(21).reshape(7, 3), 'attention_mask': torch.ones(7, 3),
        'labels': torch.tensor([0, 1, 2, 2, 3, 3, 3]),
        'pair_indices': torch.tensor([[0, 1], [2, 3]]), 'pair_targets': torch.tensor([0, 2]),
    }
    loader = [batch, batch]
    baseline_recorder, new_recorder = GradientRecorder(reference), GradientRecorder(model)
    torch.manual_seed(93)
    old.train_mixed(reference, loader, baseline_recorder, row_ce, pair_ce, torch.device('cpu'))
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(93)
    stats = train_epoch(model, loader, new_recorder, row_ce, pair_ce, torch.device('cpu'), 'ce3')
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for before, after in zip(baseline_recorder.gradients, new_recorder.gradients):
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert stats['row_ce_rows'] == 14 and stats['pair_groups'] == 4
    assert all(torch.equal(p, initial[name]) for name, p in model.named_parameters())

    recorder = GradientRecorder(model)
    stats = train_epoch(model, loader, recorder, row_ce, pair_ce, torch.device('cpu'), 'hinge16')
    assert stats['optimizer_steps'] == 2 and stats['seen_N'] == 6
    assert all(torch.isfinite(g).all() for step in recorder.gradients for g in step)
    assert all(torch.equal(p, initial[name]) for name, p in model.named_parameters())
    print('训练路径通过：CE3与旧混合流程逐步梯度/RNG相同；hinge同一步反向；检查没有更新参数。')


def check_data():
    train, dev, _ = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    groups = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(
        groups, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED + 30),
    )
    seen, pair_count, counts, max_rows = [], 0, torch.zeros(4, dtype=torch.long), 0
    for batch in loader:
        seen.extend(batch['source_rows'].tolist())
        counts += torch.bincount(batch['labels'], minlength=4)
        pair_count += len(batch['pair_targets'])
        max_rows = max(max_rows, len(batch['labels']))
        for a, b in batch['pair_indices'].tolist():
            row0, row1 = train.iloc[batch['source_rows'][a].item()], train.iloc[batch['source_rows'][b].item()]
            assert row0.pair_id == row1.pair_id and row0.text1 == row1.text2 and row0.text2 == row1.text1
    assert sorted(seen) == list(range(3290))
    assert len(groups) == 1910 and pair_count == 1380 and len(dev) == 660
    assert counts.tolist() == [933, 933, 894, 530]
    assert len(loader) == 239 and max_rows == 16
    # 解码不因单例身份而强制N；给单例最高F就应保留F。
    singleton = train.groupby('pair_id', sort=False).filter(lambda group: len(group) == 1).iloc[:1]
    assert pair_aware(torch.tensor([[4., 0., 0., -2.]]), singleton).item() == 0
    print('数据通过：3290行各出现一次，1380组Pair完整，530条N；239步/轮，最多16行。')
    print('预测通过：单例保留模型argmax，没有统一判N。')
    return tokenizer, next(iter(loader))


def check_gpu(tokenizer, batch):
    device = torch.device('cuda')
    model = build_model(tokenizer, device)
    assert not model.use_lexical_features and all(p.requires_grad for p in model.parameters())
    row_ce, pair_ce = build_criteria(device)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
    loss, row, pair = compute_joint_loss(
        logits, batch['labels'].to(device), batch['pair_indices'].to(device),
        batch['pair_targets'].to(device), row_ce, pair_ce,
        margin=cfg.HINGE_MARGIN, pair_weight=cfg.PAIR_WEIGHT,
    )
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('GPU前后向通过：', {'rows': len(logits), 'row_ce': row.item(), 'hinge16': pair.item(),
                               'peak_allocated_GiB': round(torch.cuda.max_memory_allocated() / 1024**3, 3)})
    print('未创建优化器，没有执行参数更新；峰值不包含正式训练的Adam状态。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    check_hinge_formula()
    check_ce_and_singletons()
    check_training_path()
    tokenizer, batch = check_data()
    if args.gpu:
        check_gpu(tokenizer, batch)
    print('检查完成，没有启动训练。')
