"""只根据文本复用正反视图；pair_id、金标和组大小都不进入模型。"""
import torch
from torch.utils.data import Dataset


LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
LABEL_TO_ID = dict(zip(FULL_LABELS, range(4)))
REVERSE = [1, 0, 2, 3]


class BidirectionalDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.frame = frame.reset_index(drop=True)
        self.units = []
        unit_by_text = {}

        # canonical只是固定两个视图的排列，不改变原行标签或文本。
        # 同一文本对即使跨pair_id重复，仍只编码一次，所有原行CE都保留。
        for row_index, (text1, text2) in enumerate(zip(frame.text1, frame.text2)):
            canonical = tuple(sorted((text1, text2)))
            if canonical not in unit_by_text:
                unit_by_text[canonical] = len(self.units)
                self.units.append({'texts': canonical, 'source_rows': [], 'swapped': []})
            unit = self.units[unit_by_text[canonical]]
            unit['source_rows'].append(row_index)
            unit['swapped'].append((text1, text2) != canonical)

        first, second = [], []
        for unit in self.units:
            text1, text2 = unit['texts']
            first.extend([text1, text2])
            second.extend([text2, text1])
        self.encoded = tokenizer(
            first, second, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt',
        )
        # 允许无标签文本进行预测。标签只在train_epoch中用于CE。
        self.labels = torch.tensor(
            frame.label.map(LABEL_TO_ID).tolist() if 'label' in frame else [-100] * len(frame),
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.units)

    def __getitem__(self, index):
        unit = self.units[index]
        indices = unit['source_rows']
        return {
            'input_ids': self.encoded['input_ids'][2 * index:2 * index + 2],
            'attention_mask': self.encoded['attention_mask'][2 * index:2 * index + 2],
            'labels': self.labels[indices],
            'source_rows': torch.tensor(indices, dtype=torch.long),
            'row_swapped': torch.tensor(unit['swapped'], dtype=torch.bool),
        }


def collate_units(units):
    input_ids = torch.cat([unit['input_ids'] for unit in units])
    attention_mask = torch.cat([unit['attention_mask'] for unit in units])
    # 原句通常很短：本批裁掉全是PAD的尾部，长度补到8的倍数。
    # 两个方向都已经独立截断，裁尾不会删除任何实际token。
    longest = int(attention_mask.sum(dim=1).max())
    length = min(((longest + 7) // 8) * 8, input_ids.shape[1])
    return {
        'input_ids': input_ids[:, :length],
        'attention_mask': attention_mask[:, :length],
        'row_units': torch.cat([
            torch.full((len(unit['labels']),), index, dtype=torch.long)
            for index, unit in enumerate(units)
        ]),
        'row_swapped': torch.cat([unit['row_swapped'] for unit in units]),
        'labels': torch.cat([unit['labels'] for unit in units]),
        'source_rows': torch.cat([unit['source_rows'] for unit in units]),
    }


def model_inputs(batch, device):
    names = ['input_ids', 'attention_mask', 'row_units', 'row_swapped']
    return {name: batch[name].to(device) for name in names}


def write_train_reference(frame, path):
    # 仅给官方评分器提供反向ID，不参与模型输入或训练目标。
    reverse_ids = {}
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) == 2:
            first, second = group.instance_id.tolist()
            reverse_ids[first], reverse_ids[second] = second, first
    reference = frame.copy()
    reference['reverse_pair_id'] = reference.instance_id.map(reverse_ids).fillna('')
    reference.to_csv(path, index=False, encoding='utf-8-sig')
