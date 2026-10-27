import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_local_semantics import config as cfg


class LocalRowDataset(Dataset):
    """原Row输入加局部位置；未标注行的local_target=-1。"""

    def __init__(self, frame, tokenizer, annotations, max_length=128):
        self.frame = frame.reset_index(drop=True)
        encoded = tokenizer(
            self.frame.text1.tolist(), self.frame.text2.tolist(),
            truncation=True, padding='max_length', max_length=max_length,
            return_offsets_mapping=True, return_tensors='pt',
        )
        self.encoded = {key: encoded[key] for key in ['input_ids', 'attention_mask']}
        self.labels = torch.tensor(self.frame.label.map(cfg.LABEL_TO_ID).tolist(), dtype=torch.long)
        self.local_targets = torch.full((len(frame),), -1, dtype=torch.long)
        self.span1_masks = torch.zeros_like(encoded['input_ids'], dtype=torch.bool)
        self.span2_masks = torch.zeros_like(encoded['input_ids'], dtype=torch.bool)
        row_indices = {value: index for index, value in enumerate(self.frame.instance_id)}
        offsets = encoded['offset_mapping'].tolist()
        for annotation in annotations.itertuples(index=False):
            index = row_indices[annotation.instance_id]
            sequences = encoded.sequence_ids(index)
            for side, start, end, masks in [
                (0, annotation.span1_start, annotation.span1_end, self.span1_masks),
                (1, annotation.span2_start, annotation.span2_end, self.span2_masks),
            ]:
                positions = [position for position, (left, right) in enumerate(offsets[index])
                             if sequences[position] == side and right > start and left < end]
                if not positions or offsets[index][positions[-1]][1] < end:
                    raise ValueError(f'{annotation.instance_id}的局部短语未完整编码，不能静默截断。')
                masks[index, positions] = True
            self.local_targets[index] = cfg.LOCAL_TO_ID[annotation.local_relation]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {
            'input_ids': self.encoded['input_ids'][index],
            'attention_mask': self.encoded['attention_mask'][index],
            'label': self.labels[index],
            'local_target': self.local_targets[index],
            'span1_mask': self.span1_masks[index],
            'span2_mask': self.span2_masks[index],
        }


def diagnostic_frame(annotations):
    frame = annotations[['instance_id', 'pair_id', 'text1', 'text2', 'global_label']].copy()
    return frame.rename(columns={'global_label': 'label'}).reset_index(drop=True)
