"""原0.938的3290行 + 64行语义对照；仅读Dev文本做排除审计。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as shared
from experiments.deberta_manual_aug.prepare_boundary_data import check_relations
from experiments.deberta_manual_aug.prepare_relation_data import train_word_sources


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / 'semantic_contrast_examples.csv'
REFERENCE_DIR = HERE / 'outputs/run06_relation48_staged'
PREPARED_DIR = HERE / 'prepared_semantic64'
BASELINE_SHA256 = '7b53191544c9f0e4f82ad32ec4e414c50b6791df4f41d9a8a69e17abc96c6fd6'
DEV_SHA256 = '80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511'


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def prepare(prepared_dir=PREPARED_DIR):
    prepared_dir = Path(prepared_dir)
    baseline = pd.read_csv(REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    original = pd.read_csv(shared.TRAIN_PATH, keep_default_na=False)
    assert len(baseline) == 3290 and frame_hash(baseline) == BASELINE_SHA256
    assert baseline.iloc[:len(original)].reset_index(drop=True).equals(original)
    assert hashlib.sha256(shared.DEV_PATH.read_bytes()).hexdigest() == DEV_SHA256
    # 不加载Dev标签，也不根据Dev样本生成文本；仅用文本检查词形、词干与重复。
    dev_text = pd.read_csv(shared.DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    source = pd.read_csv(SOURCE_PATH, keep_default_na=False)
    assert len(source) == 40 and source.example_id.is_unique
    assert source.label.value_counts().to_dict() == {'N': 16, 'F': 8, 'B': 8, 'E': 8}
    # 既检查新表内部，也检查它与当前基线中旧48条人工关系的显式蕴含链。
    old_relations = pd.read_csv(HERE / 'relation_examples.csv', keep_default_na=False)
    check_relations(pd.concat([old_relations, source], ignore_index=True))
    provenance = train_word_sources(source, original)
    accepted, rejected = shared.screen_candidates(source, dev_text, baseline)
    assert len(rejected) == 0, rejected[['example_id', 'rejection_reason']].to_string(index=False)
    extra, notes = shared.expand_examples(accepted, baseline.columns)
    shared.validate_extra(extra)
    assert len(extra) == 64
    assert extra.label.value_counts().to_dict() == {label: 16 for label in shared.FULL.values()}
    assert len(shared.pair_set(extra)) == 40
    audit = shared.overlap_report(extra, dev_text)
    assert all(audit[name] == 0 for name in [
        'word_intersection_count', 'porter_stem_intersection_count',
        'whole_phrase_intersection_count', 'unordered_pair_intersection_count',
    ])
    assert not shared.pair_set(extra) & shared.pair_set(baseline)
    combined = pd.concat([baseline, extra], ignore_index=True)
    assert combined.instance_id.is_unique and len(combined) == 3354
    control, control_sources = shared.make_copy_control(original, extra, dev_text, seed=42)
    shared.validate_extra(control)
    assert control.label.value_counts().to_dict() == extra.label.value_counts().to_dict()
    copied = pd.concat([baseline, control], ignore_index=True)
    assert copied.instance_id.is_unique
    tables = {
        'accepted_examples.csv': accepted, 'manual_rows.csv': extra,
        'manual_label_notes.csv': notes, 'train_word_sources.csv': provenance,
        'training_data_manual.csv': combined, 'copy_control_rows.csv': control,
        'copy_control_sources.csv': control_sources, 'training_data_copy.csv': copied,
        'coverage_counts.csv': notes.groupby(['family', 'contrast_type', 'target']).size().rename('rows').reset_index(),
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(prepared_dir / name, index=False, encoding='utf-8-sig')
    report = {
        'reference_run': str(REFERENCE_DIR), 'reference_pair_f1': 0.9381451572973859,
        'baseline_rows': len(baseline), 'official_train_rows': len(original),
        'old_scope_rows': 200, 'old_relation_rows': 48,
        'source_groups': len(source), 'new_manual_rows': len(extra), 'combined_rows': len(combined),
        'manual_class_counts': extra.label.value_counts().to_dict(),
        'manual_real_pairs': 24, 'manual_n_singletons': 16,
        'new_manual_vs_dev': audit,
        'all_new_words_present_in_official_train': True,
        'new_pairs_already_in_baseline': 0,
        'explicit_relation_graph_with_old48': 'passed; does not replace semantic adjudication',
        'baseline_snapshot_canonical_sha256': frame_hash(baseline),
        'manual_training_canonical_sha256': frame_hash(combined),
        'copy_training_canonical_sha256': frame_hash(copied),
        'annotation': 'Assistant-created draft labels with reading assumptions; not independently human-adjudicated.',
        'source': 'Official Train vocabulary recombined into new pairs. New labels are designed, not official annotations.',
        'development_disclosure': 'Dev error categories guided the design. Preparation reads only Dev texts as a rejection blacklist; development was not independent of Dev.',
        'test_status': 'Hidden/not provided. Test overlap has not been checked.',
        'audit_scope': 'Zero overlap applies to the NEW 64 manual rows, not original Train, old248, or copy control.',
        'limitations': [
            'Literal/stem disjointness does not imply semantic, template, or tokenizer-subword disjointness.',
            'Strict word exclusion prevents many ordinary function words; headline/photo-fragment style is used.',
            'Context-dependent co-reference is not taught as unconditional equivalence.',
            'Copy control repeats official Train with the same added class/group budget; it has a separate overlap audit.',
        ],
        'copy_control_vs_dev': shared.overlap_report(control, dev_text),
        'files': {},
    }
    files = {'source': SOURCE_PATH, 'prepare': Path(__file__).resolve(),
             'baseline': REFERENCE_DIR / 'training_data.csv', 'official_train': shared.TRAIN_PATH,
             'dev_blacklist': shared.DEV_PATH}
    files.update({name: prepared_dir / name for name in tables})
    report['files'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in files.items()}
    (prepared_dir / 'overlap_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('原0.938数据3290 + 新语义对照64 =', len(combined))
    print('新增F/B/E/N各16；新增24个真实Pair、16个N单例。')
    print('新增与Dev词形 / 词干 / 整短语 / 无序文本对交集：',
          audit['word_intersection_count'], '/', audit['porter_stem_intersection_count'], '/',
          audit['whole_phrase_intersection_count'], '/', audit['unordered_pair_intersection_count'])
    print('全部新增用词可追溯到官方Train；样本与审计：', prepared_dir)
    return combined, extra, report


def build_train_dataset(mode='manual', prepared_dir=PREPARED_DIR):
    combined, extra, report = prepare(prepared_dir)
    if mode == 'manual':
        train = combined
    elif mode == 'baseline':
        train = combined.iloc[:report['baseline_rows']].copy()
    elif mode == 'copy':
        train = pd.read_csv(Path(prepared_dir) / 'training_data_copy.csv', keep_default_na=False)
    else:
        raise ValueError('AUGMENT_MODE选择manual、copy或baseline。')
    return train, report


if __name__ == '__main__':
    prepare()
