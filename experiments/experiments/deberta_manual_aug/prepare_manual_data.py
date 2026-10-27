"""准备手工设计的增强草稿；严格检查新增样本与Dev的完整词、词干和文本重合。"""
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from nltk.stem import PorterStemmer


EXPERIMENT_DIR = Path(__file__).resolve().parent
PREPARED_DIR = EXPERIMENT_DIR / 'prepared'
SOURCE_PATH = EXPERIMENT_DIR / 'manual_examples.csv'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
REFERENCE_DIR = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'
FULL = {'F': 'FORWARD_ENTAILMENT', 'B': 'BACKWARD_ENTAILMENT',
        'E': 'EQUIVALENCE', 'N': 'NEGATIVE_OTHER'}
INVERSE = {'F': 'B', 'B': 'F', 'E': 'E', 'N': 'N'}
STEMMER = PorterStemmer()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def words(text):
    # 不删停用词；连字符分开；数字也算token。这里不是模型的subword词表。
    return re.findall(r'[^\W_]+', normalize(text))


def vocabulary(frame):
    return {word for column in ['text1', 'text2'] for text in frame[column] for word in words(text)}


def text_set(frame):
    return {normalize(text) for column in ['text1', 'text2'] for text in frame[column]}


def pair_key(first, second):
    return tuple(sorted((normalize(first), normalize(second))))


def pair_set(frame):
    return {pair_key(first, second) for first, second in zip(frame['text1'], frame['text2'])}


def overlap_report(extra, holdout):
    extra_words = vocabulary(extra)
    holdout_words = vocabulary(holdout)
    shared_words = sorted(extra_words & holdout_words)
    shared_stems = sorted({STEMMER.stem(word) for word in extra_words}
                          & {STEMMER.stem(word) for word in holdout_words})
    shared_phrases = sorted(text_set(extra) & text_set(holdout))
    shared_pairs = sorted(pair_set(extra) & pair_set(holdout))
    return {
        'word_intersection': shared_words, 'word_intersection_count': len(shared_words),
        'porter_stem_intersection': shared_stems,
        'porter_stem_intersection_count': len(shared_stems),
        'whole_phrase_intersection': shared_phrases,
        'whole_phrase_intersection_count': len(shared_phrases),
        'unordered_pair_intersection': [list(pair) for pair in shared_pairs],
        'unordered_pair_intersection_count': len(shared_pairs),
    }


def screen_candidates(source, dev_text, baseline):
    forbidden = vocabulary(dev_text)
    forbidden_stems = {STEMMER.stem(word) for word in forbidden}
    existing = pair_set(baseline)
    seen = set()
    accepted = []
    rejected = []
    for row in source.to_dict('records'):
        terms = set(words(row['text1']) + words(row['text2']))
        shared = sorted(terms & forbidden)
        stem_hits = sorted(word for word in terms if STEMMER.stem(word) in forbidden_stems)
        key = pair_key(row['text1'], row['text2'])
        reasons = []
        if shared:
            reasons.append('Dev完整词重合:' + '|'.join(shared))
        if stem_hits:
            reasons.append('Dev词干重合:' + '|'.join(stem_hits))
        if key in existing:
            reasons.append('与已有训练文本对重复')
        if key in seen:
            reasons.append('人工源表内部文本对重复')
        if reasons:
            rejected.append({**row, 'rejection_reason': '; '.join(reasons)})
        else:
            accepted.append(row)
            seen.add(key)
    return (pd.DataFrame(accepted, columns=source.columns),
            pd.DataFrame(rejected, columns=[*source.columns, 'rejection_reason']))


def expand_examples(source, columns):
    rows = []
    notes = []
    for row in source.itertuples(index=False):
        assert row.label in FULL
        views = [(row.text1, row.text2, row.label)]
        if row.label != 'N':
            views.append((row.text2, row.text1, INVERSE[row.label]))
        for direction, (first, second, target) in enumerate(views):
            pair_id = f'aug_manual_{row.example_id}'
            instance_id = f'{pair_id}__{direction}'
            rows.append({
                'instance_id': instance_id, 'pair_id': pair_id,
                'text1_lang': 'en', 'text2_lang': 'en',
                'text1': first, 'text2': second, 'label': FULL[target],
            })
            notes.append({
                'instance_id': instance_id, 'pair_id': pair_id,
                'source_example_id': row.example_id, 'family': row.family,
                'contrast_type': getattr(row, 'contrast_type', row.family),
                'text1': first, 'text2': second, 'target': target,
                'label_reason': row.label_reason,
                'annotation_status': 'assistant_handcrafted_draft; not independently human-adjudicated',
            })
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(notes)


def validate_extra(frame):
    assert frame['instance_id'].is_unique
    targets = {(FULL['F'], FULL['B']), (FULL['B'], FULL['F']), (FULL['E'], FULL['E'])}
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) == 1:
            assert group.iloc[0]['label'] == FULL['N']
        else:
            assert len(group) == 2 and tuple(group['label']) in targets
            first, second = group.iloc[0], group.iloc[1]
            assert first.text1 == second.text2 and first.text2 == second.text1


def make_copy_control(original, manual, dev_text, seed=42):
    # 可选预算对照：复制已有Train，不创造新文本；因此不要求复制内容与Dev词汇分离。
    # 排除原Train中与Dev整对重复的组，避免把已有重合额外放大。
    blocked = pair_set(dev_text)
    pool = {'FB': [], 'E': [], 'N': []}
    for _, group in original.groupby('pair_id', sort=False):
        if pair_key(group.iloc[0].text1, group.iloc[0].text2) in blocked:
            continue
        kind = 'N' if len(group) == 1 else ('E' if group.iloc[0].label == FULL['E'] else 'FB')
        pool[kind].append(group)
    rng = random.Random(seed)
    rows = []
    notes = []
    for target_id, target in manual.groupby('pair_id', sort=False):
        kind = 'N' if len(target) == 1 else ('E' if target.iloc[0].label == FULL['E'] else 'FB')
        selected = rng.choice(pool[kind])
        # 按目标标签排列，确保F/B反向行和原增强预算一致。
        selected_rows = selected.to_dict('records')
        if kind == 'FB' and selected_rows[0]['label'] != target.iloc[0].label:
            selected_rows.reverse()
        for direction, record in enumerate(selected_rows):
            source_id = record['instance_id']
            record = dict(record)
            record['pair_id'] = target_id.replace('aug_manual_', 'aug_manual_copy_', 1)
            record['instance_id'] = f"{record['pair_id']}__{direction}"
            rows.append(record)
            notes.append({'instance_id': record['instance_id'], 'source_instance_id': source_id,
                          'matched_manual_pair_id': target_id, 'label': record['label']})
    return pd.DataFrame(rows, columns=original.columns), pd.DataFrame(notes)


def prepare(source_path=SOURCE_PATH, prepared_dir=PREPARED_DIR):
    source_path = Path(source_path)
    prepared_dir = Path(prepared_dir)
    original = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    baseline = pd.read_csv(REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    saved = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    assert len(baseline) == 3242 and baseline.iloc[:len(original)].reset_index(drop=True).equals(original)
    effective_hash = hashlib.sha256(baseline.to_csv(index=False).encode('utf-8')).hexdigest()
    assert effective_hash == saved['augmentation']['effective_train_sha256'], '0.935参考训练快照发生变化。'
    # 过滤阶段仅读取Dev文本，不读取Dev标签；错误分析另存，不作为训练输入。
    dev_text = pd.read_csv(DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    source = pd.read_csv(source_path, keep_default_na=False)
    assert source['example_id'].is_unique
    accepted, rejected = screen_candidates(source, dev_text, baseline)
    manual, notes = expand_examples(accepted, baseline.columns)
    assert len(manual) > 0, '筛选后没有可用人工样本。'
    validate_extra(manual)
    audit = overlap_report(manual, dev_text)
    assert all(audit[key] == 0 for key in [
        'word_intersection_count', 'porter_stem_intersection_count',
        'whole_phrase_intersection_count', 'unordered_pair_intersection_count',
    ])
    combined = pd.concat([baseline, manual], ignore_index=True)
    assert combined['instance_id'].is_unique
    control, control_notes = make_copy_control(original, manual, dev_text)
    validate_extra(control)
    assert control.label.value_counts().to_dict() == manual.label.value_counts().to_dict()
    copied = pd.concat([baseline, control], ignore_index=True)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        'accepted_examples.csv': accepted, 'rejected_examples.csv': rejected,
        'manual_rows.csv': manual, 'manual_label_notes.csv': notes,
        'training_data_manual.csv': combined, 'copy_control_rows.csv': control,
        'copy_control_sources.csv': control_notes, 'training_data_copy.csv': copied,
    }
    for filename, frame in tables.items():
        frame.to_csv(prepared_dir / filename, index=False, encoding='utf-8-sig')
    old_scope = baseline.iloc[len(original):]
    report = {
        'reference_weighted_f1': 0.9354929652717138,
        'original_rows': len(original), 'old_scope_rows': len(old_scope),
        'baseline_rows': len(baseline), 'source_groups': len(source),
        'accepted_groups': len(accepted), 'rejected_groups': len(rejected),
        'new_manual_rows': len(manual), 'combined_rows': len(combined),
        'manual_class_counts': manual.label.value_counts().to_dict(),
        'new_manual_vs_dev': audit,
        'old_scope_vs_dev': overlap_report(old_scope, dev_text),
        'original_train_vs_dev': overlap_report(original, dev_text),
        'copy_control_vs_dev': overlap_report(control, dev_text),
        'copy_control_note': 'Optional control repeats existing Train; zero-vocabulary-overlap restriction applies to NEW MANUAL rows only.',
        'baseline_snapshot_canonical_sha256': effective_hash,
        'candidate_source': 'assistant-handcrafted English examples; not independently human-adjudicated',
        'normalization': 'Unicode NFKC + casefold; all Unicode letter/number tokens; no stopword removal; additionally Porter stemming',
        'development_disclosure': 'Dev errors influenced augmentation families. Dev text was used as a rejection blacklist. Dev labels are not loaded by the preparation script.',
        'test_status': 'hidden_or_not_provided; not checked; no claim of test disjointness',
        'limitations': [
            'Zero word-form/stem overlap is not zero semantic, source-corpus, or tokenizer-subword overlap.',
            'The 200 OLD scope rows and original Train do not meet the NEW vocabulary-disjoint restriction.',
            'Strict vocabulary separation forces short phrases and excludes many ordinary function words; transfer is an empirical question.',
            'Dev-guided method development and checkpoint selection remain; no fresh-test performance claim.',
        ],
        'files': {},
    }
    paths = {'manual_source': source_path, 'official_train': TRAIN_PATH, 'dev_text_source': DEV_PATH,
             'baseline_training_snapshot': REFERENCE_DIR / 'training_data.csv',
             'baseline_run_config': REFERENCE_DIR / 'run_config.json',
             'prepare_script': Path(__file__).resolve()}
    paths.update({name: prepared_dir / name for name in tables})
    report['files'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in paths.items()}
    (prepared_dir / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('参考训练集：', len(baseline), '= 原始3042 + 旧scope200')
    print('新手工设计组数 / 筛除：', len(accepted), '/', len(rejected))
    print('新增人工行数：', len(manual), '；新训练总行数：', len(combined))
    print('新增类别：', report['manual_class_counts'])
    print('新增与Dev词形 / 词干 / 整短语 / 无序文本对交集：',
          audit['word_intersection_count'], '/', audit['porter_stem_intersection_count'], '/',
          audit['whole_phrase_intersection_count'], '/', audit['unordered_pair_intersection_count'])
    print('注意：旧200条scope和可选复制对照不满足词汇零重合，已单独披露；不会混入manual_rows.csv。')
    print('样本、理由和审计：', prepared_dir)
    return combined, manual, report


if __name__ == '__main__':
    prepare()
