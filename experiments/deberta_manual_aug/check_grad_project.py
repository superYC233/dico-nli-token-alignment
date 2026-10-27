"""CPU数学与批次检查；不加载DeBERTa、不下载、不更新任何模型参数。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_manual_aug import train_relation_grad_project as cfg
from experiments.deberta_manual_aug.row_priority_gradients import (
    separate_gradients, measure_gradients, projection_coefficient, assign_gradients,
)


def check_vectors(row_values, pair_values, expected, expected_coefficient):
    parameters = [nn.Parameter(torch.zeros(1, dtype=torch.float64)) for _ in row_values]
    versions = [p._version for p in parameters]
    row = tuple(None if value is None else torch.tensor([value], dtype=torch.float64) for value in row_values)
    pair = tuple(None if value is None else torch.tensor([value], dtype=torch.float64) for value in pair_values)
    names = ['output.weight'] + [f'align_projection.{index}' for index in range(len(row_values) - 1)]
    measured = measure_gradients(names, row, pair)
    coefficient = projection_coefficient(measured[0])
    assert abs(coefficient - expected_coefficient) < 1e-12
    assign_gradients(parameters, row, pair, coefficient)
    for parameter, value in zip(parameters, expected):
        if value is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, torch.tensor([value], dtype=torch.float64))
    assert versions == [p._version for p in parameters]


def check_combined_loss():
    # 相同logits，mixed分开取梯度再相加，必须等价于普通合并损失反向。
    torch.manual_seed(42)
    model = nn.Linear(5, 4).double()
    unused = nn.Parameter(torch.ones(2, dtype=torch.float64))
    parameters = list(model.parameters()) + [unused]
    names = ['output.weight', 'output.bias', 'unused']
    inputs = torch.randn(6, 5, dtype=torch.float64)
    batch = {
        'labels': torch.tensor([0, 1, 2, 2, 3, 3]),
        'pair_indices': torch.tensor([[0, 1], [2, 3]]),
        'pair_targets': torch.tensor([0, 2]),
    }
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64))
    pair_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float64),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    row_loss, pair_loss = cfg.compute_losses(model(inputs), batch, row_ce, pair_ce)
    row, pair = separate_gradients(row_loss, pair_loss, parameters, cfg.PAIR_WEIGHT)
    assign_gradients(parameters, row, pair, coefficient=0.0)
    split = [None if p.grad is None else p.grad.clone() for p in parameters]
    for p in parameters:
        p.grad = None
    row_loss, pair_loss = cfg.compute_losses(model(inputs), batch, row_ce, pair_ce)
    (row_loss + cfg.PAIR_WEIGHT * pair_loss).backward()
    for expected, parameter in zip(split, parameters):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-10, atol=1e-12)

    # N-only批次没有Pair图，依然正常产生四分类梯度。
    for p in parameters:
        p.grad = None
    batch['labels'] = torch.tensor([3, 3, 3, 3, 3, 3])
    batch['pair_indices'] = torch.empty(0, 2, dtype=torch.long)
    batch['pair_targets'] = torch.empty(0, dtype=torch.long)
    row_loss, pair_loss = cfg.compute_losses(model(inputs), batch, row_ce, pair_ce)
    assert pair_loss is None
    row, pair = separate_gradients(row_loss, pair_loss, parameters, cfg.PAIR_WEIGHT)
    assert all(g is None for g in pair)
    measured = measure_gradients(names, row, pair)
    assert not measured[0]['has_both_gradients']
    assert projection_coefficient(measured[0]) == 0
    assign_gradients(parameters, row, pair, 0.0)
    assert model.weight.grad is not None and unused.grad is None


class ToyTokenizer:
    def __call__(self, first, second, **kwargs):
        length = kwargs['max_length']
        ids = torch.zeros(len(first), length, dtype=torch.long)
        ids[:, 0] = torch.tensor([len(text) for text in first])
        ids[:, 1] = torch.tensor([len(text) for text in second])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(3, 4).double()

    def forward(self, input_ids, attention_mask, features):
        return self.output(input_ids.double())


def check_diagnostic():
    torch.manual_seed(42)
    model = ToyModel()
    before = [p.detach().clone() for p in model.parameters()]
    versions = [p._version for p in model.parameters()]
    batch = {
        'input_ids': torch.arange(18).reshape(6, 3) % 7,
        'attention_mask': torch.ones(6, 3, dtype=torch.long),
        'labels': torch.tensor([0, 1, 2, 2, 3, 3]),
        'pair_indices': torch.tensor([[0, 1], [2, 3]]),
        'pair_targets': torch.tensor([0, 2]),
        'source_rows': torch.arange(6),
    }
    train = pd.DataFrame({'instance_id': [f'train_{i}' for i in range(6)]})
    batches, layers = [], []
    totals = cfg.run_batches(
        model, [batch], train, nn.CrossEntropyLoss(), nn.CrossEntropyLoss(),
        torch.device('cpu'), 'diagnose', 0, batches, layers, 1,
    )
    assert totals['optimizer_steps'] == 0
    assert len(batches) == 1 and len(layers) == 1
    assert versions == [p._version for p in model.parameters()]
    for p, value in zip(model.parameters(), before):
        assert p.grad is None
        torch.testing.assert_close(p, value, rtol=0, atol=0)


def check_real_batch_plan():
    train, _, _ = cfg.read_train()
    dataset = GroupDataset(train, ToyTokenizer(), max_length=4)
    orders = []
    for _ in ['mixed', 'project']:
        loader = DataLoader(
            dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
            collate_fn=collate_groups,
            generator=torch.Generator().manual_seed(cfg.SEED + 30),
        )
        rows, pair_count, counts = [], 0, torch.zeros(4, dtype=torch.long)
        for batch in loader:
            rows.extend(batch['source_rows'].tolist())
            counts += torch.bincount(batch['labels'], minlength=4)
            pair_count += len(batch['pair_targets'])
            for (first, second), target in zip(batch['pair_indices'], batch['pair_targets']):
                expected = [(0, 1), (1, 0), (2, 2)][target.item()]
                indices = torch.stack([first, second])
                assert tuple(batch['labels'][indices].tolist()) == expected
        assert sorted(rows) == list(range(3290))
        assert pair_count == 1380 and len(loader) == 239
        assert counts.tolist() == [933, 933, 894, 530]
        orders.append(rows)
    assert orders[0] == orders[1]
    print('3290行/1380对/530单例、每轮239步、两臂顺序与标签曝光一致。')


def main():
    check_vectors([1, 1], [2, -3], [3.5, -1.5], -0.5)
    check_vectors([1, 2], [2, 0], [3, 2], 0.0)
    check_vectors([1, 2], [-2, -4], [1, 2], -2.0)
    check_vectors([1, 1], [-2, None], [0, 2], -1.0)
    check_vectors([1, None], [-1, 2], [1, 2], -1.0)
    check_vectors([0, 0], [1, -1], [1, -1], 0.0)
    check_vectors([1, None], [None, None], [1, None], 0.0)
    check_combined_loss()
    check_diagnostic()
    check_real_batch_plan()
    print('CPU检查全部通过：投影公式、无冲突、缺失梯度、零梯度、混合等价、N-only、无更新诊断。')


if __name__ == '__main__':
    main()
