"""右键运行：准备768条家族增强、192条保留诊断与等预算复制对照；不训练。"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_family_aug.family_examples import build_examples
from experiments.deberta_manual_aug.prepare_manual_data import (
    FULL, INVERSE, STEMMER, words, normalize, pair_key, pair_set, text_set,
    vocabulary, overlap_report, validate_extra, make_copy_control,
)


HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / 'prepared'
BASELINE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
OFFICIAL_TRAIN = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
AUDIT_FIELDS = ['word_intersection_count', 'porter_stem_intersection_count',
                'whole_phrase_intersection_count', 'unordered_pair_intersection_count']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_material():
    # 已使用的增强源表和明确保留的诊断；不扫描模型输出，不读取Dev标签。
    candidates = [
        ROOT / 'experiments/deberta_manual_aug/manual_examples.csv',
        ROOT / 'experiments/deberta_manual_aug/boundary_examples.csv',
        ROOT / 'experiments/deberta_manual_aug/event_examples.csv',
        ROOT / 'experiments/deberta_manual_aug/semantic_contrast_examples.csv',
        ROOT / 'experiments/deberta_manual_aug/outputs/local_contrasts_prepared/extra_train_rows.csv',
        ROOT / 'experiments/deberta_manual_aug/outputs/local_contrasts_prepared/diagnostic_rows.csv',
    ]
    diagnostic_root = ROOT / 'experiments/deberta_minimal_contrasts/prepared'
    candidates.extend(sorted(diagnostic_root.glob('*/examples.csv')))
    frames, used = [], []
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, keep_default_na=False)
            if {'text1', 'text2'}.issubset(frame.columns):
                frames.append(frame[['text1', 'text2']])
                used.append(path)
    return pd.concat(frames, ignore_index=True), used


def screen_families(source, dev_text, baseline, previous):
    dev_words = vocabulary(dev_text)
    dev_stems = {STEMMER.stem(word) for word in dev_words}
    forbidden_pairs = pair_set(baseline) | pair_set(previous)
    existing_phrases = text_set(baseline) | text_set(previous)
    seen_pairs = set()
    accepted, rejected = [], []
    # 按整个家族接收/拒绝；不单独删一行，避免拆散F/B/E配对和类别预算。
    for family, group in source.groupby('family_id', sort=False):
        terms = vocabulary(group)
        reasons = []
        shared = sorted(terms & dev_words)
        stem_hits = sorted(word for word in terms if STEMMER.stem(word) in dev_stems)
        keys = [pair_key(a, b) for a, b in zip(group.text1, group.text2)]
        if shared:
            reasons.append('Dev词形:' + '|'.join(shared))
        if stem_hits:
            reasons.append('Dev词干:' + '|'.join(stem_hits))
        if set(keys) & forbidden_pairs:
            reasons.append('已有Train/增强/诊断文本对重复')
        if len(set(keys)) != len(keys) or set(keys) & seen_pairs:
            reasons.append('新源对照内部无序文本对重复')
        if group.split.iloc[0] == 'diagnostic' and text_set(group) & existing_phrases:
            reasons.append('保留诊断的完整短语已出现在旧训练/增强/诊断中')
        if reasons:
            rejected.append({'family_id': family, 'split': group.split.iloc[0],
                             'reason': '; '.join(reasons)})
        else:
            accepted.append(group)
            seen_pairs.update(keys)
    accepted = pd.concat(accepted, ignore_index=True) if accepted else source.iloc[:0].copy()
    return accepted, pd.DataFrame(rejected, columns=['family_id', 'split', 'reason'])


def expand_examples(source, columns):
    records, notes = [], []
    for row in source.itertuples(index=False):
        views = [(row.text1, row.text2, row.label)]
        if row.label != 'N':
            views.append((row.text2, row.text1, INVERSE[row.label]))
        # 一半限制对照以BF顺序入表，仍是完整的真实互换组。
        if row.contrast_type in ['restriction2', 'restriction4']:
            views.reverse()
        pair_id = 'aug_manual_family_' + row.example_id
        for direction, (first, second, label) in enumerate(views):
            instance_id = f'{pair_id}__{direction}'
            records.append({'instance_id': instance_id, 'pair_id': pair_id,
                            'text1_lang': 'en', 'text2_lang': 'en',
                            'text1': first, 'text2': second, 'label': FULL[label]})
            notes.append({'instance_id': instance_id, 'pair_id': pair_id,
                          'family_id': row.family_id, 'family_type': row.family_type,
                          'split': row.split, 'contrast_type': row.contrast_type,
                          'source_example_id': row.example_id, 'text1': first, 'text2': second,
                          'target': label, 'source_target': row.label,
                          'swapped_from_source': first == row.text2 and second == row.text1,
                          'label_reason': row.label_reason,
                          'reading_assumption': row.reading_assumption,
                          'annotation_status': 'assistant_handcrafted; not independently adjudicated'})
    return pd.DataFrame(records, columns=columns), pd.DataFrame(notes)


def word_provenance(source, official):
    exact, by_stem = {}, {}
    for row in official.itertuples(index=False):
        for column in ['text1', 'text2']:
            phrase = getattr(row, column)
            for word in words(phrase):
                evidence = (row.instance_id, column, phrase)
                exact.setdefault(word, evidence)
                by_stem.setdefault(STEMMER.stem(word), evidence)
    records = []
    for split in ['train', 'diagnostic']:
        for word in sorted(vocabulary(source[source.split.eq(split)])):
            if word in exact:
                status, evidence = 'official_train_exact_token', exact[word]
            elif STEMMER.stem(word) in by_stem:
                status, evidence = 'official_train_stem_only', by_stem[STEMMER.stem(word)]
            else:
                status, evidence = 'manually_selected_common_english', ('', '', '')
            records.append({'split': split, 'word': word, 'porter_stem': STEMMER.stem(word),
                            'source_type': status, 'train_instance_id': evidence[0],
                            'train_column': evidence[1], 'train_phrase': evidence[2]})
    return pd.DataFrame(records)


def conflict_rows(frame):
    keyed = frame.copy()
    keyed['normalized_text1'] = keyed.text1.map(normalize)
    keyed['normalized_text2'] = keyed.text2.map(normalize)
    conflicts = keyed.groupby(['normalized_text1', 'normalized_text2']).label.transform('nunique').gt(1)
    return keyed[conflicts]


def write_reference(frame, path):
    reverse_ids = {}
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) == 2:
            first, second = group.instance_id.tolist()
            reverse_ids[first], reverse_ids[second] = second, first
    reference = frame.copy()
    reference['reverse_pair_id'] = reference.instance_id.map(reverse_ids).fillna('')
    reference.to_csv(path, index=False, encoding='utf-8-sig')


def prepare():
    baseline = pd.read_csv(BASELINE_DIR / 'training_data.csv', keep_default_na=False)
    reference = json.loads((BASELINE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    canonical = hashlib.sha256(baseline.to_csv(index=False).encode('utf-8')).hexdigest()
    assert canonical == reference['augmentation']['effective_train_sha256']
    official = pd.read_csv(OFFICIAL_TRAIN, keep_default_na=False)
    dev_text = pd.read_csv(DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    previous, used_paths = old_material()
    source = build_examples()
    accepted, rejected = screen_families(source, dev_text, baseline, previous)
    assert len(accepted) > 0, '请查看整家族拒绝原因，不能用不合格样本补数量。'
    train_source = accepted[accepted.split.eq('train')]
    diagnostic_source = accepted[accepted.split.eq('diagnostic')]
    extra, train_notes = expand_examples(train_source, baseline.columns)
    diagnostic, diagnostic_notes = expand_examples(diagnostic_source, baseline.columns)
    for frame in [extra, diagnostic]:
        validate_extra(frame)
        assert frame.label.value_counts().to_dict() == {label: len(frame) // 4 for label in FULL.values()}
    assert set(train_source.family_id).isdisjoint(diagnostic_source.family_id)
    train_stems = {STEMMER.stem(word) for word in vocabulary(extra)}
    heldout_head_stems = {STEMMER.stem(word) for value in diagnostic_source.heldout_heads for word in words(value)}
    assert not train_stems & heldout_head_stems, '保留家族的指定中心词不能出现在新增训练文本中。'
    assert not pair_set(extra) & pair_set(diagnostic)
    assert not text_set(extra) & text_set(diagnostic)
    audits = {name: overlap_report(frame, dev_text)
              for name, frame in [('new_train_vs_dev', extra), ('diagnostic_vs_dev', diagnostic)]}
    assert all(report[field] == 0 for report in audits.values() for field in AUDIT_FIELDS)
    provenance = word_provenance(accepted, official)
    old_conflicts = conflict_rows(baseline)
    # 官方原标签不改；复制池排除已知冲突的整组，避免额外重复这些冲突。
    copy_pool = official[~official.pair_id.isin(old_conflicts.pair_id)].copy()
    copy, copy_notes = make_copy_control(copy_pool, extra, dev_text)
    validate_extra(copy)
    assert extra.label.value_counts().to_dict() == copy.label.value_counts().to_dict()
    group_counts = lambda frame: Counter(tuple(g.label) for _, g in frame.groupby('pair_id', sort=False))
    assert group_counts(extra) == group_counts(copy)
    combined = pd.concat([baseline, extra], ignore_index=True)
    copied = pd.concat([baseline, copy], ignore_index=True)
    assert combined.instance_id.is_unique and copied.instance_id.is_unique
    new_conflicts = conflict_rows(combined)
    assert set(new_conflicts.instance_id) == set(old_conflicts.instance_id), '新增样本不能增加同输入标签冲突。'
    assert set(conflict_rows(copied).instance_id) == set(old_conflicts.instance_id)

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    family_table = accepted.groupby(['family_id', 'split', 'family_type'], sort=False).agg(
        source_contrasts=('example_id', 'size'), heldout_heads=('heldout_heads', 'first'),
    ).reset_index()
    counts = train_notes.groupby(['family_type', 'target']).size().rename('rows').reset_index()
    tables = {
        'examples_candidates.csv': source, 'examples_source.csv': accepted,
        'rejected_families.csv': rejected, 'family_manifest.csv': family_table,
        'extra_train_rows.csv': extra, 'diagnostic_rows.csv': diagnostic,
        'manual_label_notes.csv': train_notes, 'diagnostic_label_notes.csv': diagnostic_notes,
        'training_data_manual.csv': combined, 'training_data_copy.csv': copied,
        'copy_control_rows.csv': copy, 'copy_control_sources.csv': copy_notes,
        'word_provenance.csv': provenance, 'type_label_counts.csv': counts,
        'existing_train_label_conflicts.csv': old_conflicts,
    }
    for name, frame in tables.items():
        frame.to_csv(PREPARED_DIR / name, index=False, encoding='utf-8-sig')
    write_reference(diagnostic, PREPARED_DIR / 'diagnostic_reference.csv')
    report = {
        'baseline_rows': len(baseline), 'original_rows': len(official),
        'baseline_scope_rows': 200, 'baseline_relation_rows': 48,
        'candidate_families': source.family_id.nunique(), 'rejected_families': len(rejected),
        'train_families': train_source.family_id.nunique(),
        'diagnostic_families': diagnostic_source.family_id.nunique(),
        'new_train_rows': len(extra), 'diagnostic_rows': len(diagnostic),
        'combined_train_rows': len(combined), 'class_counts_new_train': extra.label.value_counts().to_dict(),
        'combined_class_counts': combined.label.value_counts().to_dict(),
        'new_train_groups_by_order': {str(key): count for key, count in group_counts(extra).items()},
        'copy_matches_class_and_ordered_group_counts': True,
        'copy_pool_excluded_existing_conflicting_groups': old_conflicts.pair_id.nunique(),
        'train_diagnostic_pair_intersection': len(pair_set(extra) & pair_set(diagnostic)),
        'train_diagnostic_phrase_intersection': len(text_set(extra) & text_set(diagnostic)),
        'heldout_head_stem_intersection_with_new_train': [],
        'existing_conflicting_rows': len(old_conflicts), 'new_conflicting_rows': 0,
        'new_train_unordered_pair_intersection_with_baseline_or_previous': len(pair_set(extra) & (pair_set(baseline) | pair_set(previous))),
        'diagnostic_whole_phrase_intersection_with_old_training_or_previous': len(text_set(diagnostic) & (text_set(baseline) | text_set(previous))),
        **audits,
        'word_source_counts': provenance.groupby(['split', 'source_type']).size().rename('words').reset_index().to_dict('records'),
        'source_design': 'Manually selected common words and relation patterns; templates expand explicit contrast families. Not official or independently adjudicated labels.',
        'development': 'Relation families motivated by previous development analysis. Preparation reads only Dev text1/text2 as a rejection blacklist, not Dev labels.',
        'test_status': 'hidden/not provided; no assertion of zero test overlap',
        'heldout_scope': 'Held out from this augmentation and its selection; specified head stems absent from new training. Baseline may already contain individual words. Shared relation templates are intentional.',
        'quantity_scope': 'exactly n is not treated as at-least n; no rule asserting exactly n modified objects implies exactly n objects in total.',
        'previous_material_files': [str(path.relative_to(ROOT)) for path in used_paths],
        'files': {},
    }
    audit_paths = [Path(__file__), HERE / 'family_examples.py', BASELINE_DIR / 'training_data.csv',
                   OFFICIAL_TRAIN, DEV_PATH, *used_paths,
                   *[PREPARED_DIR / name for name in tables], PREPARED_DIR / 'diagnostic_reference.csv']
    report['files'] = {str(path.relative_to(ROOT)): file_hash(path) for path in audit_paths}
    (PREPARED_DIR / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('新训练家族 / 保留家族：', report['train_families'], '/', report['diagnostic_families'])
    print('新训练增强 / 保留诊断：', len(extra), '/', len(diagnostic))
    print('训练总行数：', len(baseline), '+', len(extra), '=', len(combined))
    print('新增F/B/E/N：', report['class_counts_new_train'])
    print('与Dev词形 / 词干 / 整短语 / 文本对交集：',
          [audits['new_train_vs_dev'][field] for field in AUDIT_FIELDS])
    print('整家族拒绝：', len(rejected))
    if len(rejected):
        print(rejected.to_string(index=False))
    print('源对照、标签理由、数据和审计：', PREPARED_DIR)
    return combined, extra, diagnostic, report


if __name__ == '__main__':
    prepare()
