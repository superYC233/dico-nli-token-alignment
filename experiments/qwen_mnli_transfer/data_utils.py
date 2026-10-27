import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from experiments.qwen_mnli_transfer import config as cfg
from experiments.qwen_mnli_transfer.common import file_hash, write_json
from experiments.generative_nli import (
    load_data, get_datasets, check_examples,
)
from experiments.qwen_lora.data_variants import build_train_dataset


NLI_LABELS = ['entailment', 'neutral', 'contradiction']


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def belongs_to_validation(premise, seed, train_rows, eval_rows):
    # 同一 premise 的不同 hypothesis 始终进入同一个分区，避免跨分区复用 premise。
    key = f'{seed}|{normalize(premise)}'.encode('utf-8')
    number = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
    return number % (train_rows + eval_rows) < eval_rows


def quotas(total):
    return {label: total // 3 + int(index < total % 3) for index, label in enumerate(NLI_LABELS)}


def select_mnli_rows(raw, tokenizer, blocked_texts, train_rows, eval_rows, seed, max_length):
    """只从 MNLI Train 抽样。标签编号通过数据集的 ClassLabel 名称解释。"""
    if train_rows < 3 or eval_rows < 3:
        raise ValueError('MNLI 训练 / 验证样本数均须至少为 3。')
    names = raw.features['label'].names
    if set(names) != set(NLI_LABELS):
        raise ValueError(f'未知的 MNLI 标签定义：{names}')
    limits = {'train': quotas(train_rows), 'validation': quotas(eval_rows)}
    counts = {'train': Counter(), 'validation': Counter()}
    selected = {'train': [], 'validation': []}
    seen_pairs = set()
    audit = Counter()

    # 保存源行号，复现时也可直接使用导出的完整样本 CSV。
    source = raw.add_column('_source_row', list(range(len(raw))))
    wanted_columns = ['premise', 'hypothesis', 'label', '_source_row']
    wanted_columns += [key for key in ['pairID', 'genre'] if key in raw.column_names]
    source = source.select_columns(wanted_columns).shuffle(seed=seed)
    for row in source:
        audit['scanned_rows'] += 1
        label_id = int(row['label'])
        if not 0 <= label_id < len(names):
            audit['invalid_label'] += 1
            continue
        premise, hypothesis = row['premise'], row['hypothesis']
        if not isinstance(premise, str) or not isinstance(hypothesis, str):
            audit['empty_text'] += 1
            continue
        text1, text2 = normalize(premise), normalize(hypothesis)
        if not text1 or not text2:
            audit['empty_text'] += 1
            continue
        if text1 in blocked_texts or text2 in blocked_texts:
            audit['exact_dico_text_overlap'] += 1
            continue
        key = (text1, text2)
        if key in seen_pairs:
            audit['duplicate_pair'] += 1
            continue
        split = 'validation' if belongs_to_validation(premise, seed, train_rows, eval_rows) else 'train'
        label = names[label_id]
        if counts[split][label] >= limits[split][label]:
            continue
        prompt = cfg.NLI_PROMPT.format(text1=premise, text2=hypothesis)
        prompt_ids = tokenizer(prompt)['input_ids']
        full_ids = tokenizer(prompt + label + tokenizer.eos_token)['input_ids']
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError('MNLI Prompt 和答案分词边界不稳定。')
        if len(full_ids) > max_length:
            # 整条排除，不能直接截断到把训练答案也删掉。
            audit['too_long'] += 1
            continue
        selected[split].append({
            'source_row': row['_source_row'], 'pair_id': str(row.get('pairID', row['_source_row'])),
            'genre': str(row.get('genre', '')), 'text1': premise, 'text2': hypothesis,
            'nli_label': label, 'prompt': prompt, 'completion': label,
        })
        seen_pairs.add(key)
        counts[split][label] += 1
        if len(selected['train']) == train_rows and len(selected['validation']) == eval_rows:
            break
    for split, target in [('train', train_rows), ('validation', eval_rows)]:
        if len(selected[split]) != target:
            raise ValueError(f'MNLI 合格样本不足：{split} 得到 {len(selected[split])}，需要 {target}。')
    summary = {
        'scanned_and_filtered': dict(audit),
        'label_counts': {split: dict(value) for split, value in counts.items()},
        'label_id_to_name': names,
        'validation_split': 'sha256 of normalized premise + seed; disjoint premises',
    }
    return DatasetDict({key: Dataset.from_list(rows) for key, rows in selected.items()}), summary


def prepare_mnli(tokenizer):
    dico_paths = [
        cfg.PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv',
        cfg.PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv',
    ]
    signature = {
        'preparation_version': 1, 'dataset': cfg.MNLI_DATASET, 'revision': cfg.MNLI_REVISION,
        'source_split': 'train', 'seed': cfg.SEED,
        'train_rows': cfg.MNLI_TRAIN_ROWS, 'eval_rows': cfg.MNLI_EVAL_ROWS,
        'max_length': cfg.MAX_LENGTH, 'prompt_template': cfg.NLI_PROMPT,
        'tokenizer_sha256': file_hash(cfg.MODEL_PATH / 'tokenizer.json'),
        'dico_file_hashes': {str(path): file_hash(path) for path in dico_paths},
    }
    manifest_path = cfg.PREPARED_DIR / 'manifest.json'
    paths = {split: cfg.PREPARED_DIR / f'{split}.csv' for split in ['train', 'validation']}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError('MNLI 准备配置已改变，请更换 MNLI_DATA_RUN_NAME 后重跑，避免误用旧数据。')
        for split, path in paths.items():
            if not path.is_file() or file_hash(path) != manifest['csv_sha256'][split]:
                raise ValueError(f'已准备的 MNLI 文件缺失或改变：{path}')
        print('复用已准备的 MNLI 样本：', cfg.PREPARED_DIR)
        return DatasetDict({key: Dataset.from_pandas(pd.read_csv(path, keep_default_na=False), preserve_index=False)
                            for key, path in paths.items()})
    if cfg.PREPARED_DIR.exists() and any(cfg.PREPARED_DIR.iterdir()):
        raise FileExistsError('MNLI 准备目录不完整，请更换 MNLI_DATA_RUN_NAME 或先检查该目录。')

    blocked_texts = set()
    for path in dico_paths:
        frame = pd.read_csv(path, usecols=['text1', 'text2'], keep_default_na=False)
        for column in ['text1', 'text2']:
            blocked_texts.update(normalize(value) for value in frame[column])
    print('首次运行需要下载 MNLI；仅使用它的 train 分区。')
    raw = load_dataset(
        cfg.MNLI_DATASET, revision=cfg.MNLI_REVISION,
        split='train', cache_dir=str(cfg.CACHE_DIR),
    )
    dataset, summary = select_mnli_rows(
        raw, tokenizer, blocked_texts, cfg.MNLI_TRAIN_ROWS, cfg.MNLI_EVAL_ROWS,
        cfg.SEED, cfg.MAX_LENGTH,
    )
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in paths.items():
        dataset[split].to_pandas().to_csv(path, index=False, encoding='utf-8-sig')
    summary['source_fingerprint'] = raw._fingerprint
    write_json(manifest_path, {
        'signature': signature, 'summary': summary,
        'csv_sha256': {key: file_hash(path) for key, path in paths.items()},
        'overlap_check': 'normalized exact individual text matches against visible DiCo Train/Dev; no labels used',
    })
    print('MNLI 标签数量：', summary['label_counts'])
    print('筛选记录：', summary['scanned_and_filtered'])
    print('样本名单及审计记录：', cfg.PREPARED_DIR)
    return dataset


def prepare_dico(prompt_template=None):
    if prompt_template is None:
        prompt_template = cfg.DICO_PROMPT
    dataset = load_data(cfg.PROJECT_ROOT, cfg.CACHE_DIR)
    original_rows = len(dataset['train'])
    dataset['train'] = build_train_dataset(dataset['train'], cfg.MODE)
    if dataset['train'] is None:
        raise ValueError('MODE 只支持 A / B / C。')
    print('DiCo 原始 / 构造后 / Dev：', original_rows, len(dataset['train']), len(dataset['dev']))
    return get_datasets(dataset, prompt_template)
