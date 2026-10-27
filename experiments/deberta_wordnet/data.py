import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_wordnet import config as cfg
from experiments.deberta_wordnet.wordnet_features import FEATURE_NAMES, build_features


def validate_groups(frame):
    if not frame['instance_id'].is_unique:
        raise ValueError('instance_id 重复。')
    if not frame['label'].isin(cfg.FULL_LABELS).all():
        raise ValueError('出现未知标签。')
    targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}
    for pair_id, group in frame.groupby('pair_id'):
        if len(group) == 1:
            if group.iloc[0]['label'] != 'NEGATIVE_OTHER':
                raise ValueError(f'{pair_id}：非 N 缺少反向行。')
            continue
        if len(group) != 2:
            raise ValueError(f'{pair_id}：分组不是两行。')
        row0, row1 = group.iloc[0], group.iloc[1]
        labels = (cfg.LABEL_TO_ID[row0['label']], cfg.LABEL_TO_ID[row1['label']])
        if labels not in targets:
            raise ValueError(f'{pair_id}：不属于 FB / BF / EE，不能进入 Pair CE。')
        if row0['text1'] != row1['text2'] or row0['text2'] != row1['text1']:
            raise ValueError(f'{pair_id}：文本没有正确互换。')


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev


def build_feature_tensor(frame, use_wordnet_features=True, return_audit=False):
    # 显式只传 text1、text2，不把金标、组号或“是否成对”传给特征函数。
    values = []
    audit_rows = []
    for text1, text2 in zip(frame['text1'], frame['text2']):
        features, audit = build_features(text1, text2, use_wordnet_features)
        values.append(features)
        if return_audit:
            audit_rows.append({
                name: json.dumps(value, ensure_ascii=False) for name, value in audit.items()
            })
    features = torch.tensor(values, dtype=torch.float32)
    if features.shape != (len(frame), len(FEATURE_NAMES)):
        raise ValueError('词汇特征维度错误。')
    if not torch.isfinite(features).all():
        raise ValueError('词汇特征出现 NaN 或无穷。')
    if return_audit:
        audit_frame = frame[['instance_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
        return features, pd.concat([audit_frame, pd.DataFrame(audit_rows)], axis=1)
    return features


def save_feature_table(frame, features, path):
    table = frame[['instance_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
    for index, name in enumerate(FEATURE_NAMES):
        table[name] = features[:, index].tolist()
    table.to_csv(path, index=False, encoding='utf-8-sig')


def summarize_wordnet(frame, features, audit, split):
    by_name = {name: features[:, index] for index, name in enumerate(FEATURE_NAMES)}
    def row_count(first, second):
        return int(((by_name[first] > 0) | (by_name[second] > 0)).sum())

    counts = {}
    for column in ['residual_words_1', 'residual_words_2', 'known_words_1', 'known_words_2']:
        counts[column] = sum(len(json.loads(value)) for value in audit[column]) if column in audit else 0
    residual = counts['residual_words_1'] + counts['residual_words_2']
    known = counts['known_words_1'] + counts['known_words_2']
    return {
        'split': split, 'rows': len(frame), 'residual_word_occurrences': residual,
        'wordnet_known_occurrences': known,
        'residual_word_lookup_coverage': known / residual if residual else 0.0,
        'rows_with_synonym': row_count('wn_synonym_coverage_1', 'wn_synonym_coverage_2'),
        'rows_with_directional_hypernym': row_count('wn_specific_1_to_general_2', 'wn_specific_2_to_general_1'),
        'rows_with_antonym': row_count('wn_antonym_coverage_1', 'wn_antonym_coverage_2'),
        'rows_with_any_wordnet_relation': row_count('wn_relation_coverage_1', 'wn_relation_coverage_2'),
    }


class DiCoDataset(Dataset):
    def __init__(self, frame, tokenizer, features, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length
        if features.shape != (len(frame), len(FEATURE_NAMES)):
            raise ValueError('特征与数据没有对齐。')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row['text1'], row['text2'], truncation=True,
            padding='max_length', max_length=self.max_length, return_tensors='pt',
        )
        return (
            encoded['input_ids'].squeeze(0),
            encoded['attention_mask'].squeeze(0),
            self.features[index],
            torch.tensor(cfg.LABEL_TO_ID[row['label']], dtype=torch.long),
        )


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, features, max_length=128):
        validate_groups(frame)
        self.frame = frame.reset_index(drop=True)
        self.row_dataset = DiCoDataset(self.frame, tokenizer, features, max_length)
        self.pairs = []
        targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}
        for _, group in self.frame.groupby('pair_id'):
            if len(group) != 2:
                continue
            index0, index1 = group.index.tolist()
            labels = tuple(cfg.LABEL_TO_ID[label] for label in group['label'])
            self.pairs.append((index0, index1, targets[labels]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        index0, index1, target = self.pairs[index]
        ids0, mask0, features0, _ = self.row_dataset[index0]
        ids1, mask1, features1, _ = self.row_dataset[index1]
        # 正反方向分别使用自己的特征，不能把同一个向量重复两次。
        return (
            ids0, mask0, features0,
            ids1, mask1, features1,
            torch.tensor(target, dtype=torch.long),
        )
