import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_abcde import config as cfg


def validate_groups(frame):
    """N 必须是单例；F/B/E 必须是文本互换、标签反转的两行。"""
    if not frame['instance_id'].is_unique:
        raise ValueError('instance_id 重复，请检查新增副本的编号。')
    if not frame['label'].isin(cfg.FULL_LABELS).all():
        raise ValueError('训练数据含未知标签。')
    for pair_id, group in frame.groupby('pair_id'):
        if len(group) == 1:
            if group.iloc[0]['label'] != 'NEGATIVE_OTHER':
                raise ValueError(f'{pair_id}：非 N 样本缺少反向行。')
            continue
        if len(group) != 2:
            raise ValueError(f'{pair_id}：出现 {len(group)} 行，应为一行或两行。')
        row0, row1 = group.iloc[0], group.iloc[1]
        labels = (row0['label'], row1['label'])
        allowed = [
            (cfg.FULL_LABELS[0], cfg.FULL_LABELS[1]),
            (cfg.FULL_LABELS[1], cfg.FULL_LABELS[0]),
            (cfg.FULL_LABELS[2], cfg.FULL_LABELS[2]),
        ]
        if labels not in allowed:
            raise ValueError(f'{pair_id}：不能把这两个标签组成 Real Pair：{labels}')
        if row0['text1'] != row1['text2'] or row0['text2'] != row1['text1']:
            raise ValueError(f'{pair_id}：两行文本没有正确互换。')


def select_n_sources(train):
    """与当前 Qwen C 同规则：只检查 Train，不读取 Dev 或模型预测。"""
    train_pairs = set(zip(train['text1'], train['text2']))
    negative = train[train['label'] == 'NEGATIVE_OTHER'].copy()
    seen_pairs = set()
    keep_indices = []
    reasons = []

    for index, row in negative.iterrows():
        pair = (row['text1'], row['text2'])
        reverse_pair = (row['text2'], row['text1'])
        if reverse_pair in train_pairs:
            reasons.append('reverse_exists_in_train')
        elif pair in seen_pairs:
            reasons.append('duplicate_source')
        else:
            seen_pairs.add(pair)
            keep_indices.append(index)
            reasons.append('keep')

    negative['selection'] = reasons
    selected = train.loc[keep_indices].copy()
    return selected, negative


def build_train_dataset(train, mode):
    """返回构造后的数据、N 筛选审计表、新增行审计表。"""
    if mode not in ('A', 'B', 'C', 'D', 'E'):
        raise ValueError('mode 只能是 A、B、C、D、E。')
    train = train.reset_index(drop=True).copy()
    validate_groups(train)
    selected_n, n_audit = select_n_sources(train)

    if mode == 'A':
        extra = train.iloc[:0].copy()
    elif mode in ('B', 'C'):
        extra = selected_n.copy()
    else:
        extra = train.copy()

    extra_audit = extra[['instance_id', 'pair_id', 'text1', 'text2', 'label']].copy()
    extra_audit = extra_audit.rename(columns={
        'instance_id': 'source_instance_id', 'pair_id': 'source_pair_id',
    })
    extra_audit['operation'] = 'copy'

    # C：只反转筛选后的 N。E：反转 F/B/E 和安全 N，其他 N 原样复制。
    if mode in ('C', 'E'):
        safe_n_ids = set(selected_n['instance_id'])
        swap_mask = (
            (extra['label'] != 'NEGATIVE_OTHER')
            | extra['instance_id'].isin(safe_n_ids)
        )
        old_text1 = extra.loc[swap_mask, 'text1'].copy()
        extra.loc[swap_mask, 'text1'] = extra.loc[swap_mask, 'text2']
        extra.loc[swap_mask, 'text2'] = old_text1
        extra.loc[swap_mask, 'label'] = extra.loc[swap_mask, 'label'].map(cfg.REVERSE_LABEL)
        extra_audit.loc[swap_mask, 'operation'] = 'swap'

    # 所有新增副本都用新组号，避免原有 F/B/E 的两行变成四行。
    # 同一个 F/B/E 组的两个新增副本，仍然共享同一个新 pair_id。
    extra['instance_id'] = extra['instance_id'] + '__extra'
    extra['pair_id'] = extra['pair_id'] + '__extra'
    extra_audit['new_instance_id'] = extra['instance_id']
    extra_audit['new_pair_id'] = extra['pair_id']

    result = pd.concat([train, extra], ignore_index=True)
    validate_groups(result)
    return result, n_audit, extra_audit


class RowDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            torch.tensor(cfg.LABEL_TO_ID[row['label']], dtype=torch.long),
        )


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        validate_groups(frame)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}

        for _, group in frame.groupby('pair_id'):
            if len(group) == 1:
                continue
            row0, row1 = group.iloc[0], group.iloc[1]
            labels = (cfg.LABEL_TO_ID[row0['label']], cfg.LABEL_TO_ID[row1['label']])
            self.pairs.append((
                row0['text1'], row0['text2'], row1['text1'], row1['text2'], targets[labels],
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        text1, text2, reverse1, reverse2, target = self.pairs[index]
        first = self.tokenizer(
            text1, text2, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt',
        )
        second = self.tokenizer(
            reverse1, reverse2, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt',
        )
        return (
            first['input_ids'].squeeze(0), first['attention_mask'].squeeze(0),
            second['input_ids'].squeeze(0), second['attention_mask'].squeeze(0),
            torch.tensor(target, dtype=torch.long),
        )
