"""Train内部划折、难例权重和批次整理；本文件不读取Dev。"""
import hashlib
import json
import math
import unicodedata

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from experiments.deberta_lexical_mixed_pair import config_oof as cfg
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_stage_probe.data import validate_groups


def normalize_text(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def build_fold_table(train):
    validate_groups(train)
    table = train.copy().reset_index(drop=True)
    canonical_keys = []
    canonical_labels = []
    reverse_label = {0: 1, 1: 0, 2: 2, 3: 3}
    for row in table.itertuples():
        first, second = normalize_text(row.text1), normalize_text(row.text2)
        canonical_keys.append(json.dumps(sorted([first, second]), ensure_ascii=False))
        label = cfg.LABEL_TO_ID[row.label]
        canonical_labels.append(label if first <= second else reverse_label[label])
    table['split_group'] = canonical_keys
    table['canonical_label'] = canonical_labels
    # 同一规范化文本对按同一方向比较标签，检测F/N等冲突。
    table['label_conflict'] = table.groupby('split_group')['canonical_label'].transform('nunique') > 1
    table['fold'] = -1
    splitter = StratifiedGroupKFold(n_splits=cfg.OOF_FOLDS, shuffle=True, random_state=cfg.SEED)
    for fold, (_, held_indices) in enumerate(splitter.split(
        table, table['label'].map(cfg.LABEL_TO_ID), groups=table['split_group'],
    ), start=1):
        table.loc[held_indices, 'fold'] = fold
    return table


def load_train_and_folds():
    # 只读官方Train。官方Dev的文本和标签都不用于划折或寻找难例。
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    return train, build_fold_table(train)


def oof_signature():
    # 最终RUN_NAME、加权模式和倍率不影响三折评分，所以不放入评分配置。
    names = [
        'MODEL_NAME', 'SEED', 'OOF_FOLDS', 'OOF_EPOCHS', 'USE_LEXICAL_FEATURES',
        'LEXICAL_HIDDEN_SIZE', 'LEARNING_RATE', 'LEXICAL_LEARNING_RATE', 'WEIGHT_DECAY',
        'MAX_LENGTH', 'GROUP_BATCH_SIZE', 'DEV_BATCH_SIZE', 'ROW_WEIGHTS',
        'PAIR_WEIGHTS', 'PAIR_LABEL_SMOOTHING', 'PAIR_WEIGHT',
    ]
    return {
        'parameters': {name: getattr(cfg, name) for name in names},
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'grouping': 'NFKC + casefold + whitespace; unordered text pair; keep real pairs together',
        'dtype': 'float32', 'fold_selection': 'fixed last epoch; no Dev or heldout early stopping',
    }


def build_weight_table(train, oof, mode):
    # 按instance_id对齐，不能依赖三个fold文件拼接时的行顺序。
    scores = oof.set_index('instance_id').loc[train['instance_id']].reset_index()
    assert scores['instance_id'].is_unique and len(scores) == len(train)
    for column in ['pair_id', 'text1', 'text2', 'label']:
        assert scores[column].tolist() == train[column].tolist(), column
    table = scores.copy()
    gold = table['label'].map(cfg.LABEL_TO_ID).to_numpy()
    prediction = table['prediction'].map(dict(zip(cfg.LABELS, range(4)))).to_numpy()
    wrong = gold != prediction
    # 关注E/N边界：排除仅F<->B方向判反的情况。
    table['boundary_error'] = wrong & ((gold >= 2) | (prediction >= 2))
    table['difficulty'] = 1.0 - table['gold_probability']
    groups = []
    for pair_id, group in table.groupby('pair_id', sort=False):
        first_label = cfg.LABEL_TO_ID[group.iloc[0]['label']]
        kind = 'N' if first_label == 3 else ('E' if first_label == 2 else 'FB')
        groups.append({
            'pair_id': pair_id, 'kind': kind, 'rows': len(group),
            'difficulty': float(group['difficulty'].max()),
            'boundary_error': bool(group['boundary_error'].any()),
            'label_conflict': bool(group['label_conflict'].any()),
        })
    groups = pd.DataFrame(groups)
    groups['selected_hard'] = False
    groups['raw_weight'] = 1.0
    excluded = groups['pair_id'].isin(cfg.EXCLUDE_PAIR_IDS)
    for kind in ['FB', 'E', 'N']:
        pool = groups[(groups['kind'] == kind) & ~groups['label_conflict'] & ~excluded]
        candidates = pool[pool['boundary_error']].sort_values(
            ['difficulty', 'pair_id'], ascending=[False, True], kind='stable',
        )
        count = min(len(candidates), math.ceil(len(pool) * cfg.HARD_FRACTION))
        groups.loc[candidates.head(count).index, 'selected_hard'] = True
    groups.loc[groups['selected_hard'], 'raw_weight'] = cfg.HARD_MULTIPLIER

    if mode == 'random':
        # 在FB/E/N组内打乱同一套权重，保持加权数量、类别与幅度相同。
        rng = np.random.default_rng(cfg.SEED)
        for kind in ['FB', 'E', 'N']:
            indices = groups.index[(groups['kind'] == kind) & ~groups['label_conflict'] & ~excluded]
            groups.loc[indices, 'raw_weight'] = rng.permutation(groups.loc[indices, 'raw_weight'].to_numpy())
    elif mode == 'uniform':
        groups['raw_weight'] = 1.0
    elif mode != 'hard':
        raise ValueError('WEIGHT_MODE只能是hard、random或uniform。')

    table['selected_hard'] = table['pair_id'].map(groups.set_index('pair_id')['selected_hard'])
    table['raw_weight'] = table['pair_id'].map(groups.set_index('pair_id')['raw_weight'])
    # 各类别内部归一化：每类权重之和仍等于原始样本数；N的原类别权重1.5保留。
    table['sample_weight'] = table['raw_weight'] / table.groupby('label')['raw_weight'].transform('mean')
    groups['boosted'] = groups['raw_weight'] > 1.0
    return table, groups


class WeightedGroupDataset(GroupDataset):
    def __init__(self, frame, tokenizer, features, sample_weights, max_length=128):
        super().__init__(frame, tokenizer, features, max_length)
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32)
        assert self.sample_weights.shape == (len(frame),)

    def __getitem__(self, index):
        group = super().__getitem__(index)
        group['sample_weights'] = self.sample_weights[group['source_rows']]
        return group


def collate_weighted_groups(groups):
    batch = collate_groups(groups)
    batch['sample_weights'] = torch.cat([group['sample_weights'] for group in groups])
    return batch
