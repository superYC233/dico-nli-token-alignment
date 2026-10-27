import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_joint_pair import config as cfg


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    return train, dev


class RowDataset(Dataset):
    """用于 Dev 逐行预测；提前编码，训练期间不重复分词。"""

    def __init__(self, frame, tokenizer, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.encoded = tokenizer(
            self.frame['text1'].tolist(), self.frame['text2'].tolist(),
            truncation=True, padding='max_length', max_length=max_length,
            return_tensors='pt',
        )
        self.labels = torch.tensor(
            self.frame['label'].map(cfg.LABEL_TO_ID).tolist(), dtype=torch.long,
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {
            'input_ids': self.encoded['input_ids'][index],
            'attention_mask': self.encoded['attention_mask'][index],
            'label': self.labels[index],
        }


class GroupDataset(Dataset):
    """配对的两行绑在一起；N 单例也作为一个组进入同一训练流程。"""

    def __init__(self, frame, tokenizer, max_length=128):
        self.rows = RowDataset(frame, tokenizer, max_length)
        self.groups = []
        self.pair_count = 0
        targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}

        for pair_id, group in self.rows.frame.groupby('pair_id', sort=False):
            indices = group.index.tolist()
            labels = tuple(self.rows.labels[indices].tolist())
            if len(indices) == 1 and labels == (3,):
                target = -1  # N 单例只有 Row CE，没有 Pair CE。
            elif len(indices) == 2 and labels in targets:
                first, second = group.iloc[0], group.iloc[1]
                if first['text1'] != second['text2'] or first['text2'] != second['text1']:
                    raise ValueError(f'{pair_id} 的两行文本没有互换。')
                target = targets[labels]
                self.pair_count += 1
            else:
                raise ValueError(f'{pair_id} 不是合法配对或 N 单例。')
            self.groups.append((indices, target))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        indices, target = self.groups[index]
        return {
            'input_ids': self.rows.encoded['input_ids'][indices],
            'attention_mask': self.rows.encoded['attention_mask'][indices],
            'labels': self.rows.labels[indices],
            'source_rows': torch.tensor(indices, dtype=torch.long),
            'pair_target': target,
        }


def collate_groups(groups):
    # 把本批所有组展开为行；同时记录每对的两行在这个 batch 的位置。
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
        'labels': torch.cat([group['labels'] for group in groups]),
        'source_rows': torch.cat([group['source_rows'] for group in groups]),
        'pair_indices': torch.tensor(pair_indices, dtype=torch.long).reshape(-1, 2),
        'pair_targets': torch.tensor(pair_targets, dtype=torch.long),
    }
