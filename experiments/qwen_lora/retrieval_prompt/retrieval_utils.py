"""文本对检索：不读取待预测问题的金标；不把标签编码进检索向量。"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def digest(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def read_frame(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {'instance_id', 'pair_id', 'text1', 'text2', 'label'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{path} 缺少列：{required - set(frame.columns)}')
    if not frame['instance_id'].is_unique:
        raise ValueError(f'{path} 中 instance_id 不唯一。')
    if not frame['label'].isin(cfg.LABEL_MAP).all():
        raise ValueError(f'{path} 有未知标签。')
    return frame.reset_index(drop=True)


def normalized_pair(text1, text2):
    # 仅用于排除重复；保留原文用于向量和 Prompt。
    return (' '.join(text1.casefold().split()), ' '.join(text2.casefold().split()))


def clean_candidate_pool(train):
    """同一有序文本对标签冲突时，排除相关 pair_id；相同标签的重复只留一条。"""
    reverse = {'F': 'B', 'B': 'F', 'E': 'E', 'N': 'N'}
    relations = {}
    for row in train.to_dict('records'):
        key = normalized_pair(row['text1'], row['text2'])
        label = cfg.LABEL_MAP[row['label']]
        relations.setdefault(key, set()).add(label)
        relations.setdefault(key[::-1], set()).add(reverse[label])

    blocked_groups = set()
    for row in train.to_dict('records'):
        key = normalized_pair(row['text1'], row['text2'])
        if len(relations[key]) > 1:
            blocked_groups.add(row['pair_id'])

    kept, excluded, seen = [], [], set()
    for row in train.to_dict('records'):
        key = normalized_pair(row['text1'], row['text2'])
        if row['pair_id'] in blocked_groups:
            excluded.append({**row, 'reason': 'conflicting_relation'})
        elif key in seen:
            excluded.append({**row, 'reason': 'duplicate_ordered_pair'})
        else:
            seen.add(key)
            kept.append(row)
    if not kept:
        raise ValueError('候选训练集为空。')
    excluded_frame = pd.DataFrame(excluded, columns=[*train.columns, 'reason'])
    return pd.DataFrame(kept).reset_index(drop=True), excluded_frame


def eligible_indices(pool, query):
    """排除同 pair_id，以及文本相同/互换的整组示例，不使用 query 的 label。"""
    key = normalized_pair(query['text1'], query['text2'])
    blocked = {query['pair_id']}
    records = pool.to_dict('records')
    for row in records:
        candidate = normalized_pair(row['text1'], row['text2'])
        if candidate == key or candidate == key[::-1]:
            blocked.add(row['pair_id'])
    return [i for i, row in enumerate(records) if row['pair_id'] not in blocked]


def pick_examples(pool, eligible, scores, count, method, seed):
    """随机和检索使用相同候选池；每个源 pair_id 最多出现一次。"""
    if method == 'random':
        order = list(eligible)
        random.Random(seed).shuffle(order)
    elif method == 'retrieval':
        order = sorted(eligible, key=lambda i: (-float(scores[i]), i))
    else:
        raise ValueError('未知示例选择方法。')
    selected, used_groups = [], set()
    for index in order:
        group = pool.iloc[index]['pair_id']
        if group not in used_groups:
            selected.append(index)
            used_groups.add(group)
        if len(selected) == count:
            return selected
    raise ValueError(f'只有 {len(selected)} 个独立候选组，无法取 {count} 个例子。')


def render_prompt(template, query, examples):
    # 无示例组逐字复用原短 Prompt，不额外添加说明。
    question = template.format(text1=query['text1'], text2=query['text2'])
    if not examples:
        return question
    marker = 'Text 1: {text1}'
    if template.count(marker) != 1:
        raise ValueError('Prompt 模板无法定位问题的开始位置。')
    instruction, question_template = template.split(marker, 1)
    blocks = []
    for rank, example in enumerate(examples, 1):
        label = cfg.LABEL_MAP[example['label']]
        blocks.append(
            f"Example {rank}:\nText 1: {example['text1']}\n"
            f"Text 2: {example['text2']}\nAnswer: {label}\n"
        )
    target = (marker + question_template).format(
        text1=query['text1'], text2=query['text2']
    )
    return instruction + 'Examples:\n\n' + '\n'.join(blocks) + '\nNow classify this pair:\n\n' + target


def encode_texts(texts, model_name, device, batch_size, max_length):
    """按 BGE 官方用法：CLS 向量 + L2 归一化；无需 sentence-transformers。"""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc='BGE 编码短语'):
            inputs = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=False, return_tensors='pt')
            if inputs['input_ids'].shape[1] > max_length:
                raise ValueError('短语超过 BGE 长度限制，本实验不静默截断。')
            inputs = inputs.to(device)
            hidden = model(**inputs).last_hidden_state[:, 0]
            chunks.append(F.normalize(hidden.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def cached_embeddings(texts, directory):
    directory = Path(directory)
    metadata = {
        'model': cfg.EMBEDDING_MODEL, 'device': cfg.EMBEDDING_DEVICE,
        'batch_size': cfg.EMBEDDING_BATCH_SIZE, 'max_length': cfg.EMBEDDING_MAX_LENGTH,
        'pooling': 'CLS_L2', 'texts_sha256': digest(texts),
    }
    cache = directory / 'phrase_embeddings.npz'
    record = directory / 'embedding_metadata.json'
    if cache.exists() and record.exists():
        if json.loads(record.read_text(encoding='utf-8')) == metadata:
            vectors = np.load(cache, allow_pickle=False)['vectors']
            if len(vectors) == len(texts) and np.isfinite(vectors).all():
                print('复用已保存的短语向量。')
                return vectors
    vectors = encode_texts(texts, cfg.EMBEDDING_MODEL, cfg.EMBEDDING_DEVICE,
                           cfg.EMBEDDING_BATCH_SIZE, cfg.EMBEDDING_MAX_LENGTH)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vectors=vectors)
    record.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return vectors


def preparation_spec(train, queries, template):
    return {
        'version': 1, 'train_sha256': digest(train.to_dict('records')),
        'queries_sha256': digest(queries.to_dict('records')),
        'embedding_model': cfg.EMBEDDING_MODEL,
        'embedding_device': cfg.EMBEDDING_DEVICE,
        'embedding_batch_size': cfg.EMBEDDING_BATCH_SIZE,
        'embedding_max_length': cfg.EMBEDDING_MAX_LENGTH,
        'prompt_template': template, 'num_examples': cfg.NUM_EXAMPLES,
        'seed': cfg.SEED, 'max_prompt_tokens': cfg.MAX_PROMPT_TOKENS,
        'similarity': '0.5*cos(query1,train1)+0.5*cos(query2,train2)',
        'pool': 'original_train_only; conflict_filter; ordered_dedup',
    }

