"""原C标签数据 + supported理由任务；解释不进入标签Prompt。"""
import hashlib
import json
import re

import pandas as pd
import torch
from datasets import Dataset

from experiments.qwen_lora.data_variants import build_train_dataset
from experiments.qwen_lora.reason_sft import config as cfg


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reverse_reason_reference(reason):
    # 只交换指向输入位置的短语，不改动事件本身的first/second顺序。
    replacements = {'text 1': 'Text 2', 'text 2': 'Text 1'}
    pattern = r'\bText [12]\b|\b(?:first|second) (?:phrase|text|description|side)\b'
    def replace(match):
        text = match.group(0)
        if text.casefold() in replacements:
            return replacements[text.casefold()]
        first, rest = text.split(' ', 1)
        swapped = 'second' if first.casefold() == 'first' else 'first'
        return (swapped.capitalize() if first[0].isupper() else swapped) + ' ' + rest
    return re.sub(pattern, replace, reason, flags=re.IGNORECASE)


def load_train_frame():
    original = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    reasons = pd.read_csv(cfg.REASON_PATH, keep_default_na=False)
    if not reasons.instance_id.is_unique or set(reasons.instance_id) != set(original.instance_id):
        raise ValueError('理由CSV必须与官方Train的3042个ID对应。')
    reasons = reasons.set_index('instance_id').loc[original.instance_id]
    for column in ['text1', 'text2', 'label']:
        if reasons[column].tolist() != original[column].tolist():
            raise ValueError(f'理由CSV的{column}与原Train不一致。')
    frame = build_train_dataset(Dataset.from_pandas(original, preserve_index=False), cfg.MODE).to_pandas()
    sources, texts, statuses = [], [], []
    for row in frame.itertuples():
        reverse = row.instance_id.endswith('__swap')
        source_id = row.instance_id.removesuffix('__swap') if reverse else row.instance_id
        source = reasons.loc[source_id]
        expected = (source.text2, source.text1) if reverse else (source.text1, source.text2)
        if (row.text1, row.text2) != expected or row.label != source.label:
            raise ValueError(f'C样本与来源理由不对应：{row.instance_id}')
        sources.append(source_id)
        texts.append(reverse_reason_reference(source.reason) if reverse else source.reason)
        statuses.append(source.annotation_status)
    frame['source_instance_id'] = sources
    frame['reason'] = texts
    frame['annotation_status'] = statuses
    frame['use_reason'] = frame.annotation_status.eq('supported') & frame.reason.str.strip().ne('')
    return frame


def encode_completion(tokenizer, prompt, completion, max_length=cfg.MAX_LENGTH):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + completion + tokenizer.eos_token, add_special_tokens=False)
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Prompt与答案分词边界不一致。')
    if len(ids) > max_length:
        raise ValueError(f'样本长度{len(ids)}超过{max_length}，请增加MAX_LENGTH；不会静默截断理由。')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


def build_dataset(frame, tokenizer, include_reason, label_prompt=None, reason_prompt=None, max_length=None):
    label_prompt = cfg.LABEL_PROMPT if label_prompt is None else label_prompt
    reason_prompt = cfg.REASON_PROMPT if reason_prompt is None else reason_prompt
    max_length = cfg.MAX_LENGTH if max_length is None else max_length
    examples = []
    for row in frame.itertuples():
        prompt = label_prompt.format(text1=row.text1, text2=row.text2)
        label = cfg.LABEL_MAP[row.label]
        example = encode_completion(tokenizer, prompt, label, max_length)
        target_ids = [value for value in example['labels'] if value != -100]
        if target_ids != tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]:
            raise ValueError('标签任务必须只监督一个关系token和EOS。')
        if include_reason and row.use_reason:
            explanation_prompt = reason_prompt.format(text1=row.text1, text2=row.text2)
            reason = encode_completion(tokenizer, explanation_prompt, row.reason, max_length)
            example['reason_input_ids'] = reason['input_ids']
            example['reason_labels'] = reason['labels']
        else:
            example['reason_input_ids'] = []
            example['reason_labels'] = []
        examples.append(example)
    return Dataset.from_list(examples)


class ReasonCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def pad(self, ids, labels):
        length = max(map(len, ids))
        tokens = torch.full((len(ids), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(tokens)
        targets = torch.full_like(tokens, -100)
        for index, (sequence, target) in enumerate(zip(ids, labels)):
            tokens[index, :len(sequence)] = torch.tensor(sequence)
            mask[index, :len(sequence)] = 1
            targets[index, :len(target)] = torch.tensor(target)
        return {'input_ids': tokens, 'attention_mask': mask, 'labels': targets}

    def __call__(self, examples):
        batch = self.pad([e['input_ids'] for e in examples], [e['labels'] for e in examples])
        reasons = [e for e in examples if e['reason_input_ids']]
        batch['reason_inputs'] = self.pad(
            [e['reason_input_ids'] for e in reasons], [e['reason_labels'] for e in reasons],
        ) if reasons else None
        return batch


def save_prepared(frame, dataset, directory, label_prompt=None, reason_prompt=None):
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / 'train_C_with_reasons.csv', index=False, encoding='utf-8-sig')
    frame[frame.use_reason].to_csv(directory / 'reason_task_rows.csv', index=False, encoding='utf-8-sig')
    audit = []
    for index, example in enumerate(dataset):
        audit.append({
            'instance_id': frame.iloc[index].instance_id,
            'label_input_tokens': len(example['input_ids']),
            'label_target_tokens': sum(t != -100 for t in example['labels']),
            'reason_input_tokens': len(example['reason_input_ids']),
            'reason_target_tokens': sum(t != -100 for t in example['reason_labels']),
        })
    table = pd.DataFrame(audit)
    table.to_csv(directory / 'token_audit.csv', index=False, encoding='utf-8-sig')
    info = {
        'train_rows': len(frame), 'original_rows': int((~frame.instance_id.str.endswith('__swap')).sum()),
        'reverse_N_rows': int(frame.instance_id.str.endswith('__swap').sum()),
        'reason_rows': int(frame.use_reason.sum()),
        'reason_original_rows': int((frame.use_reason & ~frame.instance_id.str.endswith('__swap')).sum()),
        'reason_reverse_N_rows': int((frame.use_reason & frame.instance_id.str.endswith('__swap')).sum()),
        'reason_label_counts': frame.loc[frame.use_reason, 'label'].value_counts().to_dict(),
        'maximum_label_length': int(table.label_input_tokens.max()),
        'maximum_reason_length': int(table.reason_input_tokens.max()),
        'train_sha256': file_hash(cfg.TRAIN_PATH), 'reason_csv_sha256': file_hash(cfg.REASON_PATH),
        'label_prompt': cfg.LABEL_PROMPT if label_prompt is None else label_prompt,
        'reason_prompt': cfg.REASON_PROMPT if reason_prompt is None else reason_prompt,
        'reason_is_label_input': False, 'dev_reason_used': False,
        'reason_annotation_source': 'assistant-authored drafts; supported subset; not independently adjudicated',
        'C_augmentation': 'reuse historical filtered reversed-N construction; no new reversed rows beyond original C',
    }
    (directory / 'data_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info
