import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_nli_large import config as cfg
from experiments.deberta_nli_large.lexical_features import FEATURE_NAMES, build_lexical_features


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


def build_feature_tensor(frame):
    # 显式只传 text1、text2，不把金标、组号或“是否成对”传给特征函数。
    values = [
        build_lexical_features(text1, text2)
        for text1, text2 in zip(frame['text1'], frame['text2'])
    ]
    features = torch.tensor(values, dtype=torch.float32)
    if features.shape != (len(frame), len(FEATURE_NAMES)):
        raise ValueError('词汇特征维度错误。')
    if not torch.isfinite(features).all():
        raise ValueError('词汇特征出现 NaN 或无穷。')
    return features


def save_feature_table(frame, features, path):
    table = frame[['instance_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
    for index, name in enumerate(FEATURE_NAMES):
        table[name] = features[:, index].tolist()
    table.to_csv(path, index=False, encoding='utf-8-sig')


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


class GroupDataset(Dataset):
    def __init__(self, frame, tokenizer, features, max_length=128):
        validate_groups(frame)
        self.frame = frame.reset_index(drop=True)
        # 固定长度与旧版一致；提前分词，避免每轮重新编码 Train。
        self.encoded = tokenizer(
            self.frame['text1'].tolist(), self.frame['text2'].tolist(),
            truncation=True, padding='max_length', max_length=max_length,
            return_tensors='pt',
        )
        self.features = features
        self.labels = torch.tensor(self.frame['label'].map(cfg.LABEL_TO_ID).tolist())
        self.groups = []
        self.pair_count = 0
        targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}
        for _, group in self.frame.groupby('pair_id', sort=False):
            indices = group.index.tolist()
            if len(indices) == 2:
                labels = tuple(self.labels[indices].tolist())
                target = targets[labels]
                self.pair_count += 1
            else:
                target = -1  # N 单例进入四分类训练，不进入三组合 Pair CE。
            self.groups.append((indices, target))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        indices, target = self.groups[index]
        return {
            'input_ids': self.encoded['input_ids'][indices],
            'attention_mask': self.encoded['attention_mask'][indices],
            'features': self.features[indices],
            'labels': self.labels[indices],
            'source_rows': torch.tensor(indices),
            'pair_target': target,
        }


def collate_groups(groups):
    pair_indices = []
    pair_targets = []
    offset = 0
    for group in groups:
        if group['pair_target'] >= 0:
            pair_indices.append([offset, offset + 1])
            pair_targets.append(group['pair_target'])
        offset += len(group['labels'])

    return {
        'input_ids': torch.cat([group['input_ids'] for group in groups]),
        'attention_mask': torch.cat([group['attention_mask'] for group in groups]),
        'features': torch.cat([group['features'] for group in groups]),
        'labels': torch.cat([group['labels'] for group in groups]),
        # source_rows 只用于核对数据覆盖率，不作为模型输入。
        'source_rows': torch.cat([group['source_rows'] for group in groups]),
        'pair_indices': torch.tensor(pair_indices, dtype=torch.long).reshape(-1, 2),
        'pair_targets': torch.tensor(pair_targets, dtype=torch.long),
    }
