"""CPU检查新数据、分组、每轮打乱和两种损失的梯度；不加载或训练DeBERTa。"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_manual_aug.prepare_relation_data import prepare, PREPARED_DIR
from experiments.deberta_manual_aug.train_relation_manual import (
    MODEL_NAME, MAX_LENGTH, PAIR_WEIGHT, compute_joint_loss, RealPairDataset,
)


def main():
    here = Path(__file__).resolve().parent
    for name in ['prepare_relation_data.py', 'train_relation_manual.py',
                 'train_relation_staged.py', 'train_relation_mixed.py']:
        ast.parse((here / name).read_text(encoding='utf-8'))
    train, manual, report = prepare()
    copy = pd.read_csv(PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    assert manual.label.tolist() == copy.label.tolist()
    assert report['relation_design']['all_new_words_present_in_official_train']
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    groups = GroupDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(groups.rows)
    assert len(train) == 3290 and len(groups) == 1910 and len(pairs) == 1380
    loader = DataLoader(groups, batch_size=16, shuffle=True, collate_fn=collate_groups,
                        generator=torch.Generator().manual_seed(72))
    orders = []
    for epoch in range(2):
        order, pair_count = [], 0
        counts = torch.zeros(4, dtype=torch.long)
        batches_without_n = 0
        for batch in loader:
            order.extend(batch['source_rows'].tolist())
            counts += torch.bincount(batch['labels'], minlength=4)
            pair_count += len(batch['pair_targets'])
            batches_without_n += int(not batch['labels'].eq(3).any())
            for indices in batch['pair_indices']:
                first, second = batch['source_rows'][indices].tolist()
                assert train.iloc[first].text1 == train.iloc[second].text2
                assert train.iloc[first].text2 == train.iloc[second].text1
                assert train.iloc[first].pair_id == train.iloc[second].pair_id
        assert sorted(order) == list(range(len(train))) and pair_count == len(pairs)
        assert counts.tolist() == torch.bincount(groups.rows.labels, minlength=4).tolist()
        orders.append(order)
        print(f'混合第{epoch + 1}次遍历：Row={len(order)}，Pair={pair_count}，各类={counts.tolist()}，无N批次={batches_without_n}')
    assert orders[0] != orders[1], '两轮顺序没有改变。'
    for dataset in [groups.rows, pairs]:
        row_loader = DataLoader(dataset, batch_size=16, shuffle=True,
                                generator=torch.Generator().manual_seed(52))
        first_order = list(row_loader.sampler)
        second_order = list(row_loader.sampler)
        assert sorted(first_order) == sorted(second_order) == list(range(len(dataset)))
        assert first_order != second_order

    # 验证配对关系正确但两行都想答N时：Pair CE本身没有N梯度，Row CE仍会纠正它。
    logits = torch.tensor([[10., 0., 0., 100.], [0., 10., 0., 100.], [0., 0., 0., 5.]], requires_grad=True)
    labels = torch.tensor([0, 1, 3])
    pair_indices = torch.tensor([[0, 1]])
    targets = torch.tensor([0])
    row_criterion = nn.CrossEntropyLoss()
    pair_criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    total, row_loss, pair_loss = compute_joint_loss(
        logits, labels, pair_indices, targets, row_criterion, pair_criterion,
    )
    total_grad = torch.autograd.grad(total, logits, retain_graph=True)[0]
    row_grad = torch.autograd.grad(row_loss, logits, retain_graph=True)[0]
    pair_grad = torch.autograd.grad(pair_loss, logits)[0]
    torch.testing.assert_close(total_grad, row_grad + PAIR_WEIGHT * pair_grad)
    assert pair_grad[:, 3].eq(0).all() and pair_grad[2].eq(0).all()
    assert total_grad[0, 3] > 0 and total_grad[1, 3] > 0 and total_grad[2, 3] < 0
    # 全N批次也必须能直接计算四分类损失，不伪造配对目标。
    n_logits = torch.zeros(2, 4, requires_grad=True)
    n_labels = torch.tensor([3, 3])
    n_total, n_row, n_pair = compute_joint_loss(
        n_logits, n_labels, torch.empty((0, 2), dtype=torch.long), torch.empty(0, dtype=torch.long),
        row_criterion, pair_criterion,
    )
    torch.testing.assert_close(n_total, n_row)
    assert n_pair.item() == 0
    assert torch.autograd.grad(n_total, n_logits)[0][:, 3].lt(0).all()
    print('数据隔离、Train来源、全量曝光、逐轮打乱、配对索引及Row/Pair梯度检查通过。')
    print('只编码文本并计算合成logits的梯度；没有加载或训练DeBERTa。')


if __name__ == '__main__':
    main()
