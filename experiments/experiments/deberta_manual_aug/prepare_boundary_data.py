"""准备修订后的120行语义对照；不改旧80行或事件200行，不加载模型。"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as shared


EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = EXPERIMENT_DIR / 'boundary_examples.csv'
PREPARED_DIR = EXPERIMENT_DIR / 'prepared_boundary'
REFERENCE_DIR = shared.REFERENCE_DIR


def check_relations(source):
    """检查源表已经明确写出的蕴含链；不是用图替代人工语义审核。"""
    edges = defaultdict(set)
    for row in source.itertuples(index=False):
        first, second = shared.normalize(row.text1), shared.normalize(row.text2)
        if row.label in ['F', 'E']:
            edges[first].add(second)
        if row.label in ['B', 'E']:
            edges[second].add(first)

    def reachable(first, second):
        pending = [first]
        seen = set()
        while pending:
            current = pending.pop()
            if current == second:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(edges[current])
        return False

    for row in source.itertuples(index=False):
        first, second = shared.normalize(row.text1), shared.normalize(row.text2)
        if row.label == 'F':
            assert not reachable(second, first), f'{row.example_id}与已有关系链冲突：严格F被反向推出。'
        if row.label == 'B':
            assert not reachable(first, second), f'{row.example_id}与已有关系链冲突：严格B被反向推出。'
        if row.label == 'N':
            assert not reachable(first, second) and not reachable(second, first), (
                f'{row.example_id}与已有关系链冲突：N被其他源对照推出了蕴含。'
            )


def surface_rule(first, second):
    # 与上一版审查使用同一条规则，只做捷径诊断，不用来打标签或正式预测。
    left, right = shared.words(first), shared.words(second)
    if len(left) > len(right):
        return 'F'
    if len(left) < len(right):
        return 'B'
    if Counter(left) == Counter(right):
        return 'E' if len(left) <= 3 else 'N'
    if {left[0], right[0]} in [{'completed', 'finished'}, {'scheduled', 'timetabled'}]:
        return 'E'
    return 'N'


def design_audit(notes):
    frame = notes.copy()
    frame['words1'] = frame.text1.map(lambda text: len(shared.words(text)))
    frame['words2'] = frame.text2.map(lambda text: len(shared.words(text)))
    frame['same_word_bag'] = [Counter(shared.words(a)) == Counter(shared.words(b))
                              for a, b in zip(frame.text1, frame.text2)]
    frame['surface_prediction'] = [surface_rule(a, b) for a, b in zip(frame.text1, frame.text2)]
    frame['surface_correct'] = frame.surface_prediction.eq(frame.target)
    paired_direction = frame.target.isin(['F', 'B'])
    equivalence = frame.target.eq('E')
    return frame, {
        'source_contrasts': 72, 'expanded_rows': len(frame),
        'label_counts': frame.target.value_counts().to_dict(),
        'FBE_direction_rows_with_equal_word_counts': int((paired_direction & frame.words1.eq(frame.words2)).sum()),
        'F_rows_where_left_is_shorter': int((frame.target.eq('F') & frame.words1.lt(frame.words2)).sum()),
        'E_rows_with_different_word_counts': int((equivalence & frame.words1.ne(frame.words2)).sum()),
        'three_word_same_bag_E_rows': int((equivalence & frame.same_word_bag & frame.words1.eq(3)).sum()),
        'three_word_same_bag_N_rows': int((frame.target.eq('N') & frame.same_word_bag & frame.words1.eq(3)).sum()),
        'old_surface_rule_correct': int(frame.surface_correct.sum()),
        'old_surface_rule_accuracy': float(frame.surface_correct.mean()),
        'counts_by_contrast_type': frame.groupby('contrast_type').size().to_dict(),
        'annotation': 'Assistant-curated synthetic draft with reasons; not independently human-adjudicated.',
        'purpose': 'Vary semantic relations and break the previous length/order shortcuts; no performance guarantee.',
        'limits': [
            'Failing one surface rule is not proof that the model learned semantics or that all shortcuts are removed.',
            'Current source table is explicit; it is not produced by replacing Dev words in its error examples.',
            'Polysemy/ordinary-meaning assumptions are documented in label reasons; independent review is still absent.',
            'Context-dependent official cases and absolute-threshold versus superlative cases are not forced into synthetic gold.',
        ],
    }


def prepare():
    source = pd.read_csv(SOURCE_PATH, keep_default_na=False)
    assert len(source) == 72 and source.example_id.is_unique
    assert source.label.value_counts().to_dict() == {'F': 12, 'B': 12, 'E': 24, 'N': 24}
    check_relations(source)
    combined, manual, report = shared.prepare(SOURCE_PATH, PREPARED_DIR)
    assert report['rejected_groups'] == 0, '有源对照被隔离检查拒绝；请看rejected_examples.csv，不要带缺行数据开训。'
    assert len(manual) == 120 and len(combined) == 3362
    assert manual.label.value_counts().to_dict() == {
        shared.FULL['F']: 24, shared.FULL['B']: 24, shared.FULL['E']: 48, shared.FULL['N']: 24,
    }
    notes = pd.read_csv(PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    audit, design = design_audit(notes)
    assert design['FBE_direction_rows_with_equal_word_counts'] > 0
    assert design['F_rows_where_left_is_shorter'] > 0
    assert design['E_rows_with_different_word_counts'] > 0
    assert design['three_word_same_bag_E_rows'] > 0 and design['three_word_same_bag_N_rows'] > 0
    assert design['old_surface_rule_correct'] < len(manual)
    audit.to_csv(PREPARED_DIR / 'design_audit.csv', index=False, encoding='utf-8-sig')
    counts = notes.groupby(['family', 'contrast_type', 'target']).size().rename('rows').reset_index()
    counts.to_csv(PREPARED_DIR / 'contrast_counts.csv', index=False, encoding='utf-8-sig')
    report['boundary_design'] = design
    for name, path in [('boundary_prepare_script', Path(__file__).resolve()),
                       ('design_audit', PREPARED_DIR / 'design_audit.csv'),
                       ('contrast_counts', PREPARED_DIR / 'contrast_counts.csv')]:
        report['files'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (PREPARED_DIR / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('修订版：新增120行；F24/B24/E48/N24；总训练3362行。')
    print('同词数F/B行数：', design['FBE_direction_rows_with_equal_word_counts'])
    print('左侧更短却为F：', design['F_rows_where_left_is_shorter'])
    print('不同词数的E：', design['E_rows_with_different_word_counts'])
    print('旧表面规则正确：', design['old_surface_rule_correct'], '/', len(manual))
    print('源对照与标签理由：', SOURCE_PATH)
    return combined, manual, report


if __name__ == '__main__':
    prepare()
