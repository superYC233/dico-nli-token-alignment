"""准备48行Train词汇重组对照；不加入上一批120，保留词形/词干/短语隔离。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as shared
from experiments.deberta_manual_aug.prepare_boundary_data import check_relations


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / 'relation_examples.csv'
PREPARED_DIR = HERE / 'prepared_relations'
REFERENCE_DIR = shared.REFERENCE_DIR


def train_word_sources(source, train):
    # 记录每个用词在官方Train的实际出处。不把重组后的整对冒充官方标注样本。
    first_occurrence = {}
    for row in train.itertuples(index=False):
        for column in ['text1', 'text2']:
            phrase = getattr(row, column)
            for word in shared.words(phrase):
                if word not in first_occurrence:
                    first_occurrence[word] = (row.instance_id, column, phrase)
    records = []
    for row in source.itertuples(index=False):
        for word in sorted(set(shared.words(row.text1) + shared.words(row.text2))):
            assert word in first_occurrence, f'{row.example_id}的{word}不在官方Train词表中。'
            instance_id, column, phrase = first_occurrence[word]
            records.append({'example_id': row.example_id, 'word': word,
                            'train_instance_id': instance_id, 'train_column': column,
                            'original_train_phrase': phrase})
    return pd.DataFrame(records)


def prepare():
    source = pd.read_csv(SOURCE_PATH, keep_default_na=False)
    assert len(source) == 30 and source.example_id.is_unique
    assert source.label.value_counts().to_dict() == {'N': 12, 'F': 6, 'B': 6, 'E': 6}
    check_relations(source)
    train = pd.read_csv(shared.TRAIN_PATH, keep_default_na=False)
    provenance = train_word_sources(source, train)
    combined, extra, report = shared.prepare(SOURCE_PATH, PREPARED_DIR)
    assert report['rejected_groups'] == 0, '有重组样本未通过隔离检查，请先看rejected_examples.csv。'
    assert len(extra) == 48 and len(combined) == 3290
    assert extra.label.value_counts().to_dict() == {label: 12 for label in shared.FULL.values()}
    provenance.to_csv(PREPARED_DIR / 'train_word_sources.csv', index=False, encoding='utf-8-sig')
    notes = pd.read_csv(PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    counts = notes.groupby(['family', 'contrast_type', 'target']).size().rename('rows').reset_index()
    counts.to_csv(PREPARED_DIR / 'relation_counts.csv', index=False, encoding='utf-8-sig')
    report['relation_design'] = {
        'source_contrasts': len(source), 'new_rows': len(extra),
        'class_counts': extra.label.value_counts().to_dict(),
        'all_new_words_present_in_official_train': True,
        'word_provenance_rows': len(provenance),
        'annotation': 'Assistant-designed relation labels with explicit reading assumptions; not independently adjudicated.',
        'source': 'Recombine words and expression patterns seen in official Train. The resulting pairs and labels are newly designed, not official labels.',
        'focus': 'Genuine restriction F/B versus meaning-preserving structure changes E, with argument/role/event N contrasts.',
        'previous_boundary_120_included': False,
        'development': 'Error families were selected using Dev analysis; Dev text is only a rejection blacklist during preparation.',
    }
    for name, path in [('relation_source', SOURCE_PATH), ('relation_prepare', Path(__file__).resolve()),
                       ('train_word_sources', PREPARED_DIR / 'train_word_sources.csv')]:
        report['files'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (PREPARED_DIR / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('新方案：旧scope基准3242 + 自然短语关系对照48 =', len(combined))
    print('新增F/B/E/N各12行；全部用词可追溯到官方Train。')
    print('样本与判据：', SOURCE_PATH)
    print('逐词Train来源：', PREPARED_DIR / 'train_word_sources.csv')
    return combined, extra, report


if __name__ == '__main__':
    prepare()
