import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_kim_gate import config as cfg
from experiments.deberta_kim_gate.lexical_features import FEATURE_NAMES, build_lexical_features
from experiments.deberta_kim_gate.relation_features import build_relation_edges, relation_tensor
import json
from tqdm import tqdm


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
        raise ValueError('原 20 维词汇特征维度错误。')
    if not torch.isfinite(features).all():
        raise ValueError('词汇特征出现 NaN 或无穷。')
    return features


def save_feature_table(frame, features, path):
    table = frame[['instance_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
    for index, name in enumerate(FEATURE_NAMES):
        table[name] = features[:, index].tolist()
    table.to_csv(path, index=False, encoding='utf-8-sig')


class DiCoDataset(Dataset):
    def __init__(self, frame, tokenizer, features, max_length=128, use_kim_features=True):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length
        if features.shape != (len(frame), len(FEATURE_NAMES)):
            raise ValueError('特征与数据没有对齐。')
        if not tokenizer.is_fast:
            raise ValueError('本版需要 fast tokenizer 提供字符偏移，不能猜测子词位置。')
        self.examples = []
        self.audit = []
        # 提前编码和查询词典；训练时只取张量，不再分词/查询 WordNet。
        for row in tqdm(self.frame.to_dict('records'), desc='准备词对知识'):
            encoded = tokenizer(
                row['text1'], row['text2'], truncation=True,
                padding='max_length', max_length=max_length, return_offsets_mapping=True,
            )
            edges, evidence = (build_relation_edges(row['text1'], row['text2'], encoded)
                               if use_kim_features else ([], []))
            self.examples.append((
                torch.tensor(encoded['input_ids'], dtype=torch.long),
                torch.tensor(encoded['attention_mask'], dtype=torch.long), edges,
            ))
            self.audit.append({
                'instance_id': row['instance_id'], 'word_pair_count': len(evidence),
                'subword_edge_count': len(edges), 'evidence': evidence,
            })

    def save_relation_audit(self, path):
        with open(path, 'w', encoding='utf-8') as output:
            for record in self.audit:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
        covered = sum(bool(record['word_pair_count']) for record in self.audit)
        print(f'含局部词典关系的样本：{covered}/{len(self.audit)}')


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        ids, mask, edges = self.examples[index]
        return (
            ids, mask, self.features[index], relation_tensor(edges, self.max_length),
            torch.tensor(cfg.LABEL_TO_ID[row['label']], dtype=torch.long),
        )


class PairDataset(Dataset):
    def __init__(self, frame, tokenizer, features, max_length=128, row_dataset=None, use_kim_features=True):
        validate_groups(frame)
        self.frame = frame.reset_index(drop=True)
        self.row_dataset = (row_dataset if row_dataset is not None else
                            DiCoDataset(self.frame, tokenizer, features, max_length, use_kim_features))
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
        ids0, mask0, features0, relations0, _ = self.row_dataset[index0]
        ids1, mask1, features1, relations1, _ = self.row_dataset[index1]
        # 正反方向分别使用自己的特征，不能把同一个向量重复两次。
        return (
            ids0, mask0, features0, relations0,
            ids1, mask1, features1, relations1,
            torch.tensor(target, dtype=torch.long),
        )
