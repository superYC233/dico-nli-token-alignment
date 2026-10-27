"""检查已保存的增强数据；只读文本，不加载模型、不训练、不改训练数据。"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRA = Path(__file__).resolve().parent / 'outputs/run01_original_scope/extra_train_rows.csv'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
NEAR_THRESHOLD = 0.85


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_texts(path):
    # 不使用 Dev / Test 的标签；无标签 Test 只需 text1、text2 两列。
    rows = []
    for index, row in enumerate(read_csv(path)):
        rows.append({
            'instance_id': row.get('instance_id') or f'row_{index}',
            'text1': row['text1'],
            'text2': row['text2'],
        })
    return rows


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def token_signature(text):
    # 比规范化更宽松：忽略标点。只作为额外候选检查，不声称保留语义。
    return ' '.join(re.findall(r'\w+', normalize(text)))


def pair_key(row, normalizer):
    return normalizer(row['text1']), normalizer(row['text2'])


def exact_matches(source, target, normalizer):
    index = defaultdict(list)
    for row in target:
        key = pair_key(row, normalizer)
        index[tuple(sorted(key))].append((key, row))

    matches = []
    for row in source:
        key = pair_key(row, normalizer)
        for target_key, other in index.get(tuple(sorted(key)), []):
            matches.append({
                'source_id': row['instance_id'],
                'target_id': other['instance_id'],
                'orientation': 'same_order' if key == target_key else 'reversed',
                'source_text1': row['text1'], 'source_text2': row['text2'],
                'target_text1': other['text1'], 'target_text2': other['text2'],
            })
    return matches


@lru_cache(maxsize=200_000)
def text_similarity(first, second):
    return SequenceMatcher(None, first, second, autojunk=False).ratio()


def near_matches(source, target):
    # 两个短语分别做字符相似度检查，同时检查交换方向。
    # 必须两边都 >= 0.85，每条增强最多保留 3 个候选；候选需要人工复核。
    target_texts = [(row, pair_key(row, normalize)) for row in target]
    matches = []
    for row in source:
        first, second = pair_key(row, normalize)
        candidates = []
        for other, (left, right) in target_texts:
            if sorted((first, second)) == sorted((left, right)):
                continue  # 精确重复已经单独报告。
            same = min(text_similarity(first, left), text_similarity(second, right))
            reverse = min(text_similarity(first, right), text_similarity(second, left))
            score = max(same, reverse)
            if score >= NEAR_THRESHOLD:
                candidates.append({
                    'source_id': row['instance_id'], 'target_id': other['instance_id'],
                    'similarity': score,
                    'orientation': 'same_order' if same >= reverse else 'reversed',
                    'source_text1': row['text1'], 'source_text2': row['text2'],
                    'target_text1': other['text1'], 'target_text2': other['text2'],
                    'review_status': 'unreviewed; similarity does not establish leakage',
                })
        candidates.sort(key=lambda item: item['similarity'], reverse=True)
        matches.extend(candidates[:3])
    return matches


def save_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_split(extra, train, holdout, split, output_dir):
    rules = {'raw': lambda text: text, 'normalized': normalize,
             'punctuation_ignored': token_signature}
    summary = {'holdout_rows': len(holdout), 'augmentation': {}, 'original_train': {}}
    exact_columns = ['source_id', 'target_id', 'orientation', 'source_text1',
                     'source_text2', 'target_text1', 'target_text2']
    for name, source in [('augmentation', extra), ('original_train', train)]:
        for rule, normalizer in rules.items():
            matches = exact_matches(source, holdout, normalizer)
            summary[name][rule] = {
                'matched_source_rows': len({row['source_id'] for row in matches}),
                'matched_holdout_rows': len({row['target_id'] for row in matches}),
                'row_match_pairs': len(matches),
            }
            save_csv(output_dir / f'{name}_vs_{split}_{rule}.csv', matches, exact_columns)

    candidates = near_matches(extra, holdout)
    save_csv(output_dir / f'augmentation_vs_{split}_near_candidates.csv', candidates,
             ['source_id', 'target_id', 'similarity', 'orientation', 'source_text1',
              'source_text2', 'target_text1', 'target_text2', 'review_status'])
    summary['near_duplicate_screen'] = {
        'minimum_phrase_character_similarity': NEAR_THRESHOLD,
        'max_candidates_per_augmentation_row': 3,
        'candidate_row_pairs': len(candidates),
        'augmentation_rows_with_candidates': len({row['source_id'] for row in candidates}),
        'review_status': 'not_manually_reviewed',
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--augmentation', type=Path, default=DEFAULT_EXTRA)
    parser.add_argument('--train-path', type=Path, default=TRAIN_PATH)
    parser.add_argument('--dev-path', type=Path, default=DEV_PATH)
    parser.add_argument('--test-path', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    extra_path = args.augmentation.resolve()
    output_dir = args.output_dir or extra_path.parent / 'overlap_audit'
    output_dir.mkdir(parents=True, exist_ok=True)

    extra = read_texts(extra_path)
    train = read_texts(args.train_path)
    dev = read_texts(args.dev_path)
    inputs = {'augmentation': extra_path, 'original_train': args.train_path.resolve(),
              'dev': args.dev_path.resolve(), 'audit_script': Path(__file__).resolve(),
              'generator_script': Path(__file__).with_name('scope_augmentation.py').resolve()}

    # 审计表记录了增强用词的 Train 来源；这里核查引用是否确实属于 Train。
    provenance_path = extra_path.with_name('augmentation_audit.csv')
    provenance = {'status': 'audit_table_not_available'}
    if provenance_path.exists():
        audit = read_csv(provenance_path)
        train_ids = {row['instance_id'] for row in train}
        extra_ids = {row['instance_id'] for row in extra}
        referenced_ids = set()
        for row in audit:
            references = row.get('word_source_ids') or row.get('control_source_instance_id', '')
            referenced_ids.update(value for value in references.split('|') if value)
        audit_ids = {row['instance_id'] for row in audit}
        provenance = {
            'status': 'checked', 'audit_rows': len(audit),
            'same_instance_ids_as_augmentation': audit_ids == extra_ids,
            'audit_instance_ids_unique': len(audit_ids) == len(audit),
            'unique_referenced_train_ids': len(referenced_ids),
            'referenced_ids_outside_train': sorted(referenced_ids - train_ids),
            'missing_source_reference_rows': sum(not (row.get('word_source_ids') or
                                                       row.get('control_source_instance_id')) for row in audit),
            'limitation': 'Train ID membership does not independently validate synthetic labels.',
        }
        inputs['provenance_table'] = provenance_path

    report = {
        'audit_time_utc': datetime.now(timezone.utc).isoformat(),
        'augmentation_rows': len(extra), 'original_train_rows': len(train),
        'normalization': 'Unicode NFKC + casefold + whitespace collapse; both pair orientations checked',
        'provenance': provenance,
        'dev': audit_split(extra, train, dev, 'dev', output_dir),
        'test': {'status': 'not_available_not_checked'},
        'limitations': [
            'Zero exact matches does not establish absence of semantic or source-corpus overlap.',
            'Near candidates require human review; this screen cannot find every paraphrase.',
            'Dev error analysis influenced method design and Dev selected checkpoints.',
            'Hashes identify these snapshots; they do not prove when or how they were created.',
            'This script audits saved data only and never filters or modifies training data.',
        ],
    }
    if args.test_path is not None:
        inputs['test'] = args.test_path.resolve()
        report['test'] = {'status': 'checked', **audit_split(
            extra, train, read_texts(args.test_path), 'test', output_dir)}
    report['files'] = {name: {'path': str(path), 'sha256': file_hash(path)}
                       for name, path in inputs.items()}
    (output_dir / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    counts = report['dev']['augmentation']
    original = report['dev']['original_train']['normalized']['matched_holdout_rows']
    lines = [
        '# 增强数据重合审计', '',
        f'- 新增增强：{len(extra)} 行；原始 Train：{len(train)} 行；Dev：{len(dev)} 行。',
        f'- 增强与 Dev 原样文本对重合（含反向）：{counts["raw"]["matched_source_rows"]} 条增强。',
        f'- 规范化后重合（含反向）：{counts["normalized"]["matched_source_rows"]} 条增强。',
        f'- 忽略标点后候选重合：{counts["punctuation_ignored"]["matched_source_rows"]} 条增强。',
        f'- 近似重合候选：{report["dev"]["near_duplicate_screen"]["candidate_row_pairs"]} 对，尚未人工复核。',
        f'- 原始 Train 与 Dev 规范化重合涉及 {original} 行 Dev，另表披露，不归因于增强。',
        f'- Test 状态：{report["test"]["status"]}。', '',
        '精确匹配同时检查正向和反向；匹配过程不使用 Dev/Test 标签。', '',
        '当前证据仅支持上述文本检查，不能声称“完全没有数据泄露”。Dev 曾用于错误分析和选模，',
        '增强标签仍需独立审核。隐藏 Test 未提供时，不能证明与其无交集。',
        '近似候选不等于泄露；相似但信息不同的短语需要人工区分。', '',
        '输入文件与生成脚本的 SHA256、规范化方法及全部计数见 overlap_report.json。',
        '哈希用于识别文件版本，不能证明构造时间或无语义重合。',
    ]
    (output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\n审计目录：', output_dir.resolve())


if __name__ == '__main__':
    main()
