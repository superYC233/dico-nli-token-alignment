"""20个动作/事件家族，每家族10行；复用严格Dev隔离审计，不改上一版80行。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as shared


EXPERIMENT_DIR = Path(__file__).resolve().parent
FAMILY_PATH = EXPERIMENT_DIR / 'event_families.csv'
SOURCE_PATH = EXPERIMENT_DIR / 'event_examples.csv'
PREPARED_DIR = EXPERIMENT_DIR / 'prepared_events'
REFERENCE_DIR = shared.REFERENCE_DIR


def build_event_examples(families):
    records = []

    def add(family, kind, first, second, label, reason):
        records.append({
            'example_id': f'event_{family}_{kind}',
            'text1': first, 'text2': second, 'label': label,
            'family': family, 'contrast_type': kind, 'label_reason': reason,
        })

    for index, row in enumerate(families.itertuples(index=False)):
        fine1 = f'{row.verb} {row.object1}'
        fine2 = f'{row.verb} {row.object2}'
        # 四种状态词也进入F/B，不能只要看到planned/completed就把它当作N/E。
        status = ['completed', 'scheduled', 'finished', 'planned'][index % 4]
        add(row.family, 'action_specificity', fine1, row.verb, 'F',
            '在同一动作的字面解释下，指定对象的动作蕴含该动作；反向不保证对象。')
        add(row.family, 'event_scope', f'{status} {row.event_specific}',
            f'{status} {row.event_general}', 'F',
            '两侧保留完全相同的状态算子，只去掉事件类型的限定；不是从计划推出实际发生。')
        add(row.family, 'manner_equivalence', f'{row.manner} {fine2}',
            f'{fine2} {row.manner}', 'E',
            '同一方式副词在简单动宾短语前后移动，仍修饰同一个动作；没有额外否定或范围算子。')
        if row.equivalence_kind == 'scheduling':
            first_state, second_state = 'scheduled', 'timetabled'
            reason = '按事件已被安排到时间表中的通常义，scheduled与timetabled表达同一排期状态；不表示已经执行。'
        else:
            first_state, second_state = 'completed', 'finished'
            reason = '在同一事件完成状态的通常义下，completed与finished均表示已经完成；不是计划或尝试。'
        add(row.family, 'event_state_equivalence',
            f'{first_state} {row.event_specific}', f'{second_state} {row.event_specific}', 'E', reason)

        if row.n_contrast == 'roles':
            first = f'{row.object1} {row.manner} {row.verb} {row.object2}'
            second = f'{row.object2} {row.manner} {row.verb} {row.object1}'
            kind = 'participant_roles'
            reason = '两个参与者交换施事与受事位置；词集合相同但关系角色相反，不能由一个方向推出另一个。'
        else:
            first, second = f'{row.manner} {fine1}', f'{row.manner} {fine2}'
            kind = 'action_object_change'
            reason = '动作和方式相同但对象不同；这两种对象不具有普遍包含关系，执行其中一种不保证另一种。'
        # N不补成同pair_id的两行；不同家族交替方向，避免固定左/右位置。
        if index % 2:
            first, second = second, first
        add(row.family, kind, first, second, 'N', reason)

        nonfactive = ['planned', 'scheduled'][index % 2]
        realized = ['finished', 'completed'][index % 2]
        first = f'{nonfactive} {row.event_specific}'
        second = f'{realized} {row.event_specific}'
        if index % 2:
            first, second = second, first
        add(row.family, 'planned_vs_completed', first, second, 'N',
            '计划或排期不保证完成；完成也不必然经过事前计划或排期。两种状态可共存，但不存在普遍互推。')

    source = pd.DataFrame(records)
    assert source.example_id.is_unique
    return source


def prepare():
    families = pd.read_csv(FAMILY_PATH, keep_default_na=False)
    assert len(families) == 20 and families.family.is_unique
    source = build_event_examples(families)
    source.to_csv(SOURCE_PATH, index=False, encoding='utf-8-sig')
    combined, manual, report = shared.prepare(SOURCE_PATH, PREPARED_DIR)
    notes = pd.read_csv(PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    family_counts = notes.groupby('family').size()
    assert len(manual) == 200 and family_counts.eq(10).all(), '有家族未通过Dev隔离；请先复核rejected_examples.csv。'
    assert manual.label.value_counts().to_dict() == {
        shared.FULL['F']: 40, shared.FULL['B']: 40, shared.FULL['E']: 80, shared.FULL['N']: 40,
    }
    summary = notes.groupby(['family', 'contrast_type', 'target']).size().rename('rows').reset_index()
    summary.to_csv(PREPARED_DIR / 'family_counts.csv', index=False, encoding='utf-8-sig')
    report['event_design'] = {
        'families': len(families), 'source_contrasts': len(source), 'expanded_rows': len(manual),
        'per_family': '2 F/B pairs + 2 E pairs + 2 N singletons = 10 rows',
        'N_structure': 'Each N contrast is a singleton; directions alternate across families.',
        'annotation': 'Assistant-designed templates with per-row reasons; not independent human gold.',
        'source_selection': 'No copying Dev examples; strict word-form and Porter-stem rejection against Dev text.',
    }
    for name, path in [('event_family_source', FAMILY_PATH), ('event_generator', Path(__file__).resolve()),
                       ('family_counts', PREPARED_DIR / 'family_counts.csv')]:
        report['files'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (PREPARED_DIR / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('动作/事件家族：20；每家族10行；新增200行；总训练3442行。')
    print('逐条标签理由：', PREPARED_DIR / 'manual_label_notes.csv')
    return combined, manual, report


if __name__ == '__main__':
    prepare()
