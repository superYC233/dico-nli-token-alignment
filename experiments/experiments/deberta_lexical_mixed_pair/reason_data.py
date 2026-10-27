"""只从Train合并理由；理由和依据标注不进入编码器输入。"""
import hashlib
import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.deberta_lexical_mixed_pair import config_reason as cfg
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_stage_probe.data import validate_groups


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_spans(text, phrases):
    spans = []
    for phrase in phrases.split('|') if phrases else []:
        if text.count(phrase) != 1:
            raise ValueError(f'依据片段必须在原文里唯一出现：{phrase!r} / {text!r}')
        start = text.index(phrase)
        spans.append({'start': start, 'end': start + len(phrase), 'text': phrase})
    return json.dumps(spans, ensure_ascii=False)


def load_reason_train(train_path=cfg.TRAIN_PATH, annotation_path=cfg.ANNOTATION_PATH):
    train = pd.read_csv(train_path, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    annotations = pd.read_csv(annotation_path, keep_default_na=False)
    if not annotations.instance_id.is_unique or not set(annotations.instance_id).issubset(train.instance_id):
        raise ValueError('理由ID必须唯一，并且全部来自Train。')
    columns = ['reason', 'operation', 'evidence_text1', 'evidence_text2',
               'evidence_spans1', 'evidence_spans2', 'annotation_seed_id', 'annotation_source',
               'annotation_status', 'evidence_source']
    for column in columns:
        train[column] = ''
    groups = train.groupby('pair_id', sort=False).indices
    by_id = {value: index for index, value in enumerate(train.instance_id)}
    seen_groups = set()
    for annotation in annotations.itertuples():
        source = train.iloc[by_id[annotation.instance_id]]
        if source.pair_id in seen_groups:
            raise ValueError(f'每个训练组只需一份理由：{source.pair_id}')
        seen_groups.add(source.pair_id)
        for index in groups[source.pair_id]:
            row = train.iloc[index]
            first, second = annotation.evidence_text1, annotation.evidence_text2
            if row.instance_id != source.instance_id:
                if (row.text1, row.text2) != (source.text2, source.text1):
                    raise ValueError('只有真实互换的Train行可以复用依据标注。')
                first, second = second, first
            spans1, spans2 = locate_spans(row.text1, first), locate_spans(row.text2, second)
            if not json.loads(spans1) and not json.loads(spans2):
                raise ValueError(f'理由缺少原文依据：{source.instance_id}')
            train.loc[index, columns] = [annotation.reason, annotation.operation, first, second,
                                          spans1, spans2, annotation.instance_id,
                                          getattr(annotation, 'annotation_source', 'assistant_authored_train_only_draft'),
                                          getattr(annotation, 'annotation_status', 'supported'),
                                          getattr(annotation, 'evidence_source', 'assistant_selected')]
    train['has_reason'] = train['reason'].ne('')
    train['has_evidence'] = train.has_reason & train.annotation_status.eq('supported')
    return train


def encode_reason_targets(frame, tokenizer, max_length):
    # 原文分词方式与原mixed一致；只有offset_mapping用于构造辅助目标。
    encoded = tokenizer(
        frame.text1.tolist(), frame.text2.tolist(), truncation=True,
        padding='max_length', max_length=max_length, return_tensors='pt',
        return_offsets_mapping=True,
    )
    targets = torch.full_like(encoded['input_ids'], -100)
    for index, row in enumerate(frame.itertuples()):
        if not row.has_evidence:
            continue
        sequence_ids = encoded.sequence_ids(index)
        offsets = encoded['offset_mapping'][index].tolist()
        # 特殊token、padding和未标注行均忽略；0表示未选线索，1表示所选线索。
        for token_index, sequence in enumerate(sequence_ids):
            if sequence in (0, 1) and offsets[token_index][1] > offsets[token_index][0]:
                targets[index, token_index] = 0
        for side, field in enumerate([row.evidence_spans1, row.evidence_spans2]):
            for span in json.loads(field):
                selected = [i for i, (start, end) in enumerate(offsets)
                            if sequence_ids[i] == side and max(start, span['start']) < min(end, span['end'])]
                if not selected or max(offsets[i][1] for i in selected) < span['end']:
                    raise ValueError(f'依据被截断或未对应到token：{row.instance_id} / {span}')
                targets[index, selected] = 1
    return {key: encoded[key] for key in ['input_ids', 'attention_mask']}, targets


class ReasonGroupDataset(GroupDataset):
    def __init__(self, frame, tokenizer, features, max_length=128):
        # 复用原mixed分组；保留相同输入ID、顺序及真实Pair目标。
        super().__init__(frame, tokenizer, features, max_length)
        encoded, self.evidence_targets = encode_reason_targets(frame, tokenizer, max_length)
        if not torch.equal(self.encoded['input_ids'], encoded['input_ids']):
            raise ValueError('增加依据监督不应改变原始分词。')

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch['evidence_targets'] = self.evidence_targets[batch['source_rows']]
        return batch


def collate_reason_groups(groups):
    batch = collate_groups(groups)
    batch['evidence_targets'] = torch.cat([group['evidence_targets'] for group in groups])
    return batch


class ReasonRowDataset(Dataset):
    """Dev只包含原始文本产生的输入、词汇特征和评分标签。"""
    def __init__(self, frame, tokenizer, features, max_length=128):
        self.encoded = tokenizer(
            frame.text1.tolist(), frame.text2.tolist(), truncation=True,
            padding='max_length', max_length=max_length, return_tensors='pt',
        )
        self.features = features
        self.labels = torch.tensor(frame.label.map(cfg.LABEL_TO_ID).tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.encoded['input_ids'][index], self.encoded['attention_mask'][index],
                self.features[index], self.labels[index])


def save_prepared(train, tokenizer, targets, directory):
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / 'train_with_reasons.csv', index=False, encoding='utf-8-sig')
    annotated = train[train.has_reason]
    annotated.to_csv(directory / 'reason_examples.csv', index=False, encoding='utf-8-sig')
    train[train.has_reason & ~train.has_evidence].to_csv(
        directory / 'reason_review_needed.csv', index=False, encoding='utf-8-sig')
    rows = []
    for index in annotated.index:
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(train.loc[index, 'text1'], train.loc[index, 'text2'], truncation=True,
                      padding='max_length', max_length=cfg.MAX_LENGTH)['input_ids'])
        rows.append({'instance_id': train.loc[index, 'instance_id'],
                     'annotation_status': train.loc[index, 'annotation_status'],
                     'used_for_evidence_loss': bool(train.loc[index, 'has_evidence']),
                     'evidence_tokens': ' '.join(token for token, target in zip(tokens, targets[index]) if target == 1),
                     'positive_tokens': int(targets[index].eq(1).sum()),
                     'background_tokens': int(targets[index].eq(0).sum())})
    pd.DataFrame(rows).to_csv(directory / 'evidence_token_audit.csv', index=False, encoding='utf-8-sig')
    info = {
        'train_rows': len(train), 'reason_seed_groups': train.loc[train.has_reason, 'pair_id'].nunique(),
        'annotated_rows': len(annotated), 'unannotated_rows': int((~train.has_reason).sum()),
        'evidence_supervised_rows': int(train.has_evidence.sum()),
        'reason_status_counts': train.loc[train.has_reason, 'annotation_status'].value_counts().to_dict(),
        'evidence_label_counts': train.loc[train.has_evidence, 'label'].value_counts().to_dict(),
        'evidence_source_counts': annotated.evidence_source.value_counts().to_dict(),
        'annotated_label_counts': annotated.label.value_counts().to_dict(),
        'operation_counts': annotated.operation.value_counts().to_dict(),
        'train_sha256': file_hash(cfg.TRAIN_PATH), 'annotation_sha256': file_hash(cfg.ANNOTATION_PATH),
        'input_fields': ['text1', 'text2'], 'reason_used_as_input': False,
        'new_training_rows': 0, 'dev_used_for_annotation_selection': False,
        'annotation_source': 'assistant-authored from Train text and Train labels; not independently adjudicated',
        'scope': 'sparse discriminative cues, not complete proofs; unmarked tokens remain visible to classifier',
    }
    (directory / 'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info
