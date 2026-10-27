import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from experiments.deberta_reversal_kl import config as cfg
from experiments.deberta_lexical_stage_probe.data import validate_groups


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev


class RowDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.encoded = tokenizer(
            self.frame['text1'].tolist(), self.frame['text2'].tolist(),
            truncation=True, padding='max_length', max_length=max_length,
            return_tensors='pt',
        )
        self.labels = torch.tensor(self.frame['label'].map(cfg.LABEL_TO_ID).tolist())

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {
            'input_ids': self.encoded['input_ids'][index],
            'attention_mask': self.encoded['attention_mask'][index],
            'labels': self.labels[index],
        }


class GroupDataset(RowDataset):
    def __init__(self, frame, tokenizer, max_length=128):
        super().__init__(frame, tokenizer, max_length)
        self.groups = [group.index.tolist() for _, group in self.frame.groupby('pair_id', sort=False)]
        self.pair_count = sum(len(indices) == 2 for indices in self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        indices = self.groups[index]
        return {
            'input_ids': self.encoded['input_ids'][indices],
            'attention_mask': self.encoded['attention_mask'][indices],
            'labels': self.labels[indices],
            'source_rows': torch.tensor(indices),
        }


def collate_groups(groups):
    pairs = []
    offset = 0
    for group in groups:
        if len(group['labels']) == 2:
            pairs.append([offset, offset + 1])
        offset += len(group['labels'])
    return {
        'input_ids': torch.cat([group['input_ids'] for group in groups]),
        'attention_mask': torch.cat([group['attention_mask'] for group in groups]),
        'labels': torch.cat([group['labels'] for group in groups]),
        'source_rows': torch.cat([group['source_rows'] for group in groups]),
        'pair_indices': torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
    }


def make_dev_loader(dev, tokenizer, max_length, batch_size):
    return DataLoader(
        RowDataset(dev, tokenizer, max_length), batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )
