"""右键运行：官方PAWS-Wiki人工标注train，准备1000正例+1000负例。"""
import argparse
import hashlib
import json
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional import train_deberta_alignment_joint as base_cfg


SOURCE_URL = 'https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/train-00000-of-00001.parquet'
SOURCE_README = 'https://github.com/google-research-datasets/paws'
SEED = 42
PER_CLASS = 1000
MAX_LENGTH = 128
HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / 'outputs/paws_prepared_seed42'
DICO_TRAIN_PATH = base_cfg.REFERENCE_DIR / 'training_data.csv'
DICO_DEV_PATH = base_cfg.DEV_PATH


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def pair_key(text1, text2):
    return tuple(sorted((normalize(text1), normalize(text2))))


def blocklists(frame):
    phrases = {normalize(text) for column in ['text1', 'text2'] for text in frame[column]}
    pairs = {pair_key(a, b) for a, b in zip(frame.text1, frame.text2)}
    return phrases, pairs


def download_train_file(prepared_dir):
    raw_dir = Path(prepared_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / 'train-00000-of-00001.parquet'
    archive_path = raw_dir / 'paws_wiki_labeled_final.tar.gz'
    train_path = raw_dir / 'train.tsv'
    if parquet_path.exists():
        return parquet_path
    if train_path.exists():
        return train_path
    if archive_path.exists():
        write_train_tsv(archive_path, train_path)
        return train_path
    # 原Google Storage地址已拒绝匿名访问，改用同一数据集的HF训练分区。
    print('下载PAWS-Wiki labeled_final训练文件：', SOURCE_URL, flush=True)
    temporary = parquet_path.with_suffix('.download')
    request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'DiCo-NLI-PAWS-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            stream.write(block)
    temporary.replace(parquet_path)
    return parquet_path


def read_train_file(path):
    path = Path(path)
    if path.suffix == '.parquet':
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'id': str})
    frame['id'] = frame.id.astype(str)
    return frame


def write_train_tsv(archive_path, train_path):
    # 包内还含PAWS Dev/Test；只读取train.tsv，不解压其他文件。
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and PurePosixPath(member.name).name == 'train.tsv']
        if len(members) != 1:
            raise ValueError('官方标注包中应恰好有一个train.tsv。')
        with archive.extractfile(members[0]) as source:
            Path(train_path).write_bytes(source.read())


def filter_candidates(raw, tokenizer, dico_train, dico_dev, max_length=MAX_LENGTH):
    required = ['id', 'sentence1', 'sentence2', 'label']
    frame = raw[required].rename(columns={
        'id': 'source_id', 'sentence1': 'text1', 'sentence2': 'text2', 'label': 'paws_label',
    }).copy().reset_index(drop=True)
    frame['source_id'] = frame.source_id.astype(str)
    frame['paws_label'] = frame.paws_label.astype(int)
    if not frame.paws_label.isin([0, 1]).all():
        raise ValueError('PAWS只能包含原始二元标签0/1。')
    frame['source_row'] = range(2, len(frame) + 2)
    frame['source_split'] = 'train'
    frame['pair_key'] = [pair_key(a, b) for a, b in zip(frame.text1, frame.text2)]
    train_phrases, train_pairs = blocklists(dico_train)
    dev_phrases, dev_pairs = blocklists(dico_dev)
    conflicts = set(frame.groupby('pair_key').paws_label.nunique().loc[lambda values: values > 1].index)
    seen, reasons = set(), []
    for row in frame.itertuples(index=False):
        a, b = normalize(row.text1), normalize(row.text2)
        if not a or not b:
            reason = 'empty_text'
        elif a == b:
            reason = 'identical_texts'
        elif row.pair_key in conflicts:
            reason = 'conflicting_paws_labels'
        elif row.pair_key in train_pairs or a in train_phrases or b in train_phrases:
            reason = 'dico_train_overlap'
        elif row.pair_key in dev_pairs or a in dev_phrases or b in dev_phrases:
            reason = 'dico_dev_overlap'
        elif row.pair_key in seen:
            reason = 'duplicate_pair'
        else:
            reason = ''
            seen.add(row.pair_key)
        reasons.append(reason)
    frame['reason'] = reasons
    rejected = frame[frame.reason.ne('')].copy()
    candidates = frame[frame.reason.eq('')].copy().reset_index(drop=True)

    # 不截断后硬训：如果差异被截没了，原PAWS标签可能不再适用。
    lengths = []
    for start in tqdm(range(0, len(candidates), 512), desc='检查PAWS双向长度'):
        batch = candidates.iloc[start:start + 512]
        first = tokenizer(batch.text1.tolist(), batch.text2.tolist(), truncation=False,
                          padding=False, verbose=False)['input_ids']
        second = tokenizer(batch.text2.tolist(), batch.text1.tolist(), truncation=False,
                           padding=False, verbose=False)['input_ids']
        lengths.extend(max(len(a), len(b)) for a, b in zip(first, second))
    candidates['max_view_tokens'] = lengths
    too_long = candidates.max_view_tokens.gt(max_length)
    long_rows = candidates[too_long].copy()
    long_rows['reason'] = 'exceeds_max_length'
    rejected = pd.concat([rejected, long_rows], ignore_index=True)
    candidates = candidates[~too_long].copy().reset_index(drop=True)
    return candidates.drop(columns=['pair_key', 'reason']), rejected.drop(columns=['pair_key'])


def select_balanced(candidates, per_class=PER_CLASS, seed=SEED):
    selected = []
    for label in [0, 1]:
        pool = candidates[candidates.paws_label.eq(label)]
        if len(pool) < per_class:
            raise ValueError(f'PAWS标签{label}过滤后只有{len(pool)}条，少于要求的{per_class}。')
        selected.append(pool.sample(n=per_class, random_state=seed + label))
    return pd.concat(selected, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def overlap_report(selected, dico_train, dico_dev):
    phrases, pairs = blocklists(selected)
    report = {}
    for name, frame in [('dico_train', dico_train), ('dico_dev', dico_dev)]:
        other_phrases, other_pairs = blocklists(frame)
        report[f'{name}_whole_text_intersection_count'] = len(phrases & other_phrases)
        report[f'{name}_unordered_pair_intersection_count'] = len(pairs & other_pairs)
    return report


def prepare(prepared_dir=PREPARED_DIR):
    prepared_dir = Path(prepared_dir)
    csv_path = prepared_dir / 'paws_train.csv'
    manifest_path = prepared_dir / 'preparation.json'
    request = {'seed': SEED, 'per_class': PER_CLASS, 'max_length': MAX_LENGTH,
               'source_url': SOURCE_URL, 'source_split': 'train',
               'model_name': base_cfg.MODEL_NAME, 'base_revision': base_cfg.BASE_REVISION,
               'dico_train_sha256': file_hash(DICO_TRAIN_PATH),
               'dico_dev_sha256': file_hash(DICO_DEV_PATH)}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['request'] != request or file_hash(csv_path) != manifest['selected_sha256']:
            raise ValueError('PAWS准备配置或样本文件已经变化，请使用新的prepared-dir保留旧样本。')
        print('复用已准备好的PAWS训练池：', csv_path)
        return pd.read_csv(csv_path, keep_default_na=False, dtype={'source_id': str}), manifest

    train_path = download_train_file(prepared_dir)
    raw = read_train_file(train_path)
    if len(raw) != 49401:
        raise ValueError('此文件不是预期的49401行PAWS-Wiki labeled_final训练分区，请核对下载文件。')
    tokenizer = AutoTokenizer.from_pretrained(
        base_cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=base_cfg.BASE_REVISION,
        local_files_only=True,
    )
    # DiCo Dev只读取文本，用作重复排除；不读取Dev标签选择PAWS样本。
    dico_train = pd.read_csv(DICO_TRAIN_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    dico_dev = pd.read_csv(DICO_DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    candidates, rejected = filter_candidates(raw, tokenizer, dico_train, dico_dev)
    selected = select_balanced(candidates)
    overlap = overlap_report(selected, dico_train, dico_dev)
    assert all(value == 0 for value in overlap.values())
    assert selected.paws_label.value_counts().to_dict() == {0: PER_CLASS, 1: PER_CLASS}
    prepared_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(csv_path, index=False, encoding='utf-8-sig')
    rejected.to_csv(prepared_dir / 'rejected_rows.csv', index=False, encoding='utf-8-sig')
    (prepared_dir / 'overlap_report.json').write_text(json.dumps(overlap, indent=2), encoding='utf-8')
    manifest = {
        'request': request, 'source_readme': SOURCE_README, 'raw_train_sha256': file_hash(train_path),
        'raw_train_file': str(train_path), 'raw_train_format': train_path.suffix,
        'prepare_script_sha256': file_hash(Path(__file__)),
        'raw_train_rows': len(raw), 'available_counts': candidates.paws_label.value_counts().to_dict(),
        'selected_counts': selected.paws_label.value_counts().to_dict(), 'selected_rows': len(selected),
        'selected_sha256': file_hash(csv_path), 'rejection_counts': rejected.reason.value_counts().to_dict(),
        'maximum_selected_view_tokens': int(selected.max_view_tokens.max()),
        'supervision': 'binary paraphrase only: 1=equivalent, 0=not equivalent; 0 is NOT mapped to N',
        'annotation': 'PAWS-Wiki Labeled Final; human labels, train partition only',
        'overlap': overlap, 'hidden_test_checked': False,
        'normalization': 'Unicode NFKC + casefold + whitespace; exact whole text and unordered text pair',
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nPAWS训练池准备完成：', len(selected), '；等价/非等价各', PER_CLASS)
    print('过滤原因：', manifest['rejection_counts'])
    print('整段文本及文本对重复审计：', overlap)
    print('样本文件：', csv_path)
    return selected, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED_DIR)
    args = parser.parse_args()
    prepare(args.prepared_dir)
