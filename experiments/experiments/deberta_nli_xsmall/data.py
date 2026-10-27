import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_nli_xsmall import config as cfg
from experiments.deberta_lexical_stage_probe.data import validate_groups


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev


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
