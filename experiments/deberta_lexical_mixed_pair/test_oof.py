"""CPU检查划折、样本权重和损失。不加载预训练模型，不更新模型参数。"""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.deberta_lexical_mixed_pair import config_oof as cfg
from experiments.deberta_lexical_mixed_pair.oof_data import (
    load_train_and_folds, build_weight_table, WeightedGroupDataset, collate_weighted_groups, oof_signature,
)
from experiments.deberta_lexical_mixed_pair.train_deberta_oof_hard import compute_weighted_joint_loss
from experiments.deberta_lexical_mixed_pair.train_deberta_mixed_pair import compute_joint_loss
from experiments.deberta_lexical_mixed_pair.score_train_oof import make_score_table
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor


class TinyTokenizer:
    def __call__(self, first, second, **kwargs):
        ids = [[1, 3 + len(a) % 20, 2, 3 + len(b) % 20, 2, 0] for a, b in zip(first, second)]
        return {'input_ids': torch.tensor(ids),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0]] * len(ids))}


def check_folds():
    original_reader = pd.read_csv
    reads = []
    def record_read(path, *args, **kwargs):
        reads.append(Path(path))
        return original_reader(path, *args, **kwargs)
    with patch('pandas.read_csv', side_effect=record_read):
        train, folds = load_train_and_folds()
    assert reads == [cfg.TRAIN_PATH], reads
    assert len(train) == 3042 and train['pair_id'].nunique() == 1760
    assert folds.groupby('pair_id')['fold'].nunique().eq(1).all()
    assert folds.groupby('split_group')['fold'].nunique().eq(1).all()
    covered = []
    for fold in range(1, cfg.OOF_FOLDS + 1):
        fit = folds[folds['fold'] != fold]
        held = folds[folds['fold'] == fold]
        assert set(fit['instance_id']).isdisjoint(held['instance_id'])
        assert set(fit['pair_id']).isdisjoint(held['pair_id'])
        assert set(fit['split_group']).isdisjoint(held['split_group'])
        covered.extend(held['instance_id'])
    assert sorted(covered) == sorted(train['instance_id'])
    assert int(folds['label_conflict'].sum()) == 3
    # 签名只有Train哈希，没有Dev内容；最终实验名不影响三折评分复用。
    signature = oof_signature()
    assert 'dev_sha256' not in signature
    with patch.object(cfg, 'RUN_NAME', 'another_final_run'), patch.object(cfg, 'WEIGHT_MODE', 'random'):
        assert signature == oof_signature()
    print('三折覆盖3042行；可逆对、重复文本不跨折；划折仅读取Train。')
    print('规范化无序文本组：', folds['split_group'].nunique(), '；冲突关联行：', int(folds['label_conflict'].sum()))
    return train, folds


def check_weights(train, folds):
    # 仅用于CPU逻辑检查的模拟评分，不写成真实OOF文件。
    simulated = folds.copy()
    simulated['gold_probability'] = np.linspace(0.05, 0.45, len(simulated))
    simulated['prediction'] = np.where(simulated['label'] == 'NEGATIVE_OTHER', 'E', 'N')
    # 打乱输入顺序，确认是按instance_id对齐而不是按行号拼接。
    simulated = simulated.sample(frac=1, random_state=17)
    tables = {}
    group_tables = {}
    for mode in ['hard', 'random', 'uniform']:
        table, groups = build_weight_table(train, simulated, mode)
        assert table['instance_id'].tolist() == train['instance_id'].tolist()
        for _, group in table.groupby('label'):
            assert np.isclose(group['sample_weight'].sum(), len(group))
        assert table.groupby('pair_id')['sample_weight'].nunique().eq(1).all()
        assert (table.loc[table['label_conflict'], 'raw_weight'] == 1.0).all()
        tables[mode], group_tables[mode] = table, groups
    assert (tables['uniform']['sample_weight'] == 1.0).all()
    for kind in ['FB', 'E', 'N']:
        hard = group_tables['hard'].query('kind == @kind')['raw_weight'].to_numpy()
        random = group_tables['random'].query('kind == @kind')['raw_weight'].to_numpy()
        assert np.array_equal(np.sort(hard), np.sort(random))

    exclude_id = group_tables['hard'].loc[group_tables['hard']['boosted'], 'pair_id'].iloc[0]
    with patch.object(cfg, 'EXCLUDE_PAIR_IDS', [exclude_id]):
        excluded, _ = build_weight_table(train, simulated, 'hard')
        assert (excluded.loc[excluded['pair_id'] == exclude_id, 'raw_weight'] == 1.0).all()
    print('难例/随机对照同类别同倍率；每类总权重不变；冲突组与人工排除组不额外加权。')
    return tables['hard']


def check_batches(train, weights):
    features = build_feature_tensor(train)
    dataset = WeightedGroupDataset(train, TinyTokenizer(), features, weights['sample_weight'].to_numpy())
    loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
        collate_fn=collate_weighted_groups, generator=torch.Generator().manual_seed(cfg.SEED),
    )
    covered = []
    pairs = 0
    expected = torch.tensor(weights['sample_weight'].to_numpy(), dtype=torch.float32)
    for batch in loader:
        source = batch['source_rows']
        covered.extend(source.tolist())
        pairs += len(batch['pair_targets'])
        assert torch.equal(batch['sample_weights'], expected[source])
        assert torch.equal(batch['features'], features[source])
        assert len(source) <= 16
    assert sorted(covered) == list(range(3042)) and pairs == 1282 and len(loader) == 220
    print('加权批次：每轮3042行各一次、1282对各一次、220次更新；权重与源行对齐。')


def check_loss():
    labels = torch.tensor([3, 0, 1, 2, 2])
    indices = torch.tensor([[1, 2], [3, 4]])
    targets = torch.tensor([0, 2])
    class_weights = torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64)
    row_none = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    row_mean = nn.CrossEntropyLoss(weight=class_weights)
    pair = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float64), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    logits = torch.tensor([
        [0.1, 0.2, 0.3, 1.5], [1.0, 0.1, 0.2, 0.3], [0.1, 1.2, 0.2, 0.3],
        [0.1, 0.2, 1.5, 0.3], [0.3, 0.1, 1.1, 0.2],
    ], dtype=torch.float64, requires_grad=True)
    baseline = compute_joint_loss(logits, labels, indices, targets, row_mean, pair, cfg.PAIR_WEIGHT)
    uniform = compute_weighted_joint_loss(
        logits, labels, indices, targets, torch.ones(5, dtype=torch.float64), row_none, pair, cfg.PAIR_WEIGHT,
    )
    assert all(torch.allclose(a, b) for a, b in zip(baseline, uniform))
    before = torch.autograd.grad(baseline[0], logits, retain_graph=True)[0]
    after = torch.autograd.grad(uniform[0], logits)[0]
    assert torch.allclose(before, after)

    weights = torch.tensor([1.8, 0.7, 0.7, 1.2, 1.2], dtype=torch.float64)
    weighted = compute_weighted_joint_loss(logits, labels, indices, targets, weights, row_none, pair, cfg.PAIR_WEIGHT)
    log_p = torch.log_softmax(logits, dim=1)
    coefficients = class_weights[labels] * weights
    expected = (-log_p[torch.arange(5), labels] * coefficients).sum() / coefficients.sum()
    assert torch.allclose(weighted[1], expected) and torch.equal(weighted[2], baseline[2])
    n_only = compute_weighted_joint_loss(
        torch.zeros(2, 4, dtype=torch.float64), torch.tensor([3, 3]),
        torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.ones(2, dtype=torch.float64), row_none, pair, cfg.PAIR_WEIGHT,
    )
    assert n_only[2].item() == 0.0 and torch.allclose(n_only[0], torch.tensor(np.log(4)))
    print('等权损失及梯度与原run01一致；加权分母、Pair不变和全N批次通过。')


def check_score_table(folds):
    held = folds[folds['fold'] == 1].head(12).reset_index(drop=True)
    gold = torch.tensor(held['label'].map(cfg.LABEL_TO_ID).tolist())
    logits = torch.zeros(len(held), 4)
    logits[torch.arange(len(held)), gold] = 2.0
    table = make_score_table(logits, held, 1)
    assert np.isclose(table['gold_probability'], np.exp(2)/(np.exp(2)+3)).all()
    assert np.isclose(table['gold_margin'], 2).all()
    assert table['prediction'].tolist() == [cfg.LABELS[index] for index in gold]
    # 检查评分落盘再读后，冲突列仍为布尔，不把字符串False当成True。
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    buffer.seek(0)
    restored = pd.read_csv(buffer, keep_default_na=False)
    assert restored['label_conflict'].dtype == bool
    print('OOF金标概率、margin、标签顺序和CSV布尔列通过。')


if __name__ == '__main__':
    train, folds = check_folds()
    weights = check_weights(train, folds)
    check_batches(train, weights)
    check_loss()
    check_score_table(folds)
    print('全部检查通过，没有加载预训练权重或更新模型参数。')
