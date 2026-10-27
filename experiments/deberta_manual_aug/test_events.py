"""复核事件200行与原配方；不加载模型、不训练、不修改已准备数据。"""
import ast
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as shared
from experiments.deberta_manual_aug import prepare_event_data as event


def check_data():
    families = pd.read_csv(event.FAMILY_PATH, keep_default_na=False)
    source = pd.read_csv(event.SOURCE_PATH, keep_default_na=False)
    regenerated = event.build_event_examples(families)
    assert source.equals(regenerated)
    assert len(families) == 20 and len(source) == 120
    manual = pd.read_csv(event.PREPARED_DIR / 'manual_rows.csv', keep_default_na=False)
    full = pd.read_csv(event.PREPARED_DIR / 'training_data_manual.csv', keep_default_na=False)
    control = pd.read_csv(event.PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    copied = pd.read_csv(event.PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
    notes = pd.read_csv(event.PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    baseline = pd.read_csv(event.REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    assert len(manual) == len(control) == 200 and len(full) == len(copied) == 3442
    assert full.iloc[:3242].reset_index(drop=True).equals(baseline)
    assert copied.iloc[:3242].reset_index(drop=True).equals(baseline)
    assert notes.groupby('family').size().eq(10).all()
    assert set(notes.instance_id) == set(manual.instance_id)
    assert notes.label_reason.str.len().gt(0).all()
    assert notes.annotation_status.str.contains('not independently').all()
    expected = {shared.FULL['F']: 40, shared.FULL['B']: 40,
                shared.FULL['E']: 80, shared.FULL['N']: 40}
    assert manual.label.value_counts().to_dict() == control.label.value_counts().to_dict() == expected
    for frame in [manual, control, full, copied]:
        shared.validate_extra(frame)
    assert (manual.groupby('pair_id').size() == 2).sum() == 80
    assert (manual.groupby('pair_id').size() == 1).sum() == 40
    assert (full.groupby('pair_id').size() == 2).sum() == 1442
    assert math.ceil(len(full) / 16) == 216 and math.ceil(1442 / 16) == 91
    assert set(manual.instance_id).isdisjoint(baseline.instance_id)
    assert set(manual.pair_id).isdisjoint(baseline.pair_id)
    for _, group in manual.groupby(['text1', 'text2']):
        assert group.label.nunique() == 1
    dev_text = pd.read_csv(shared.DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    audit = shared.overlap_report(manual, dev_text)
    assert all(value == 0 for key, value in audit.items() if key.endswith('_count'))
    report = json.loads((event.PREPARED_DIR / 'overlap_report.json').read_text(encoding='utf-8'))
    assert audit == report['new_manual_vs_dev']
    print('20家族、200行、类别数量、N单例、反向标签、3442总行数和复制预算均通过。')
    print('新增与Dev完整词形、词干、整短语和文本对交集均为0。')


def check_contrasts():
    families = pd.read_csv(event.FAMILY_PATH, keep_default_na=False)
    source = event.build_event_examples(families)
    for family in families.family:
        examples = source[source.family == family]
        assert examples.label.value_counts().to_dict() == {'F': 2, 'E': 2, 'N': 2}
        assert len(examples) == 6
    role = source[source.contrast_type == 'participant_roles']
    assert len(role) == 8 and role.label.eq('N').all()
    for row in role.itertuples():
        assert Counter(shared.words(row.text1)) == Counter(shared.words(row.text2))
        assert row.text1 != row.text2
    manner = source[source.contrast_type == 'manner_equivalence']
    assert len(manner) == 20 and manner.label.eq('E').all()
    for row in manner.itertuples():
        assert Counter(shared.words(row.text1)) == Counter(shared.words(row.text2))
    assert source[source.contrast_type == 'planned_vs_completed'].label.eq('N').all()
    assert source[source.contrast_type == 'event_scope'].label.eq('F').all()
    print('同词集合既有E也有N；同状态范围限制与跨状态N明确区分；六种对照每家族齐全。')


def check_recipe():
    folder = Path(__file__).resolve().parent
    original = ast.parse((ROOT / 'experiments/deberta_original_aug/train_original_aug.py').read_text(encoding='utf-8'))
    current = ast.parse((folder / 'train_event_manual.py').read_text(encoding='utf-8'))
    for name in ['set_seed', 'RealPairDataset', 'train_rows', 'train_pairs', 'evaluate', 'save_weights']:
        before = next(node for node in original.body if getattr(node, 'name', None) == name)
        after = next(node for node in current.body if getattr(node, 'name', None) == name)
        assert ast.dump(before) == ast.dump(after), name
    values = {}
    for node in current.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    reference = json.loads((event.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    for name in ['seed', 'model_name', 'epochs', 'max_length', 'learning_rate', 'weight_decay',
                 'row_batch_size', 'pair_batch_size', 'dev_batch_size', 'row_weights',
                 'pair_weights', 'pair_weight', 'pair_label_smoothing']:
        assert values[name.upper()] == reference[name], name
    assert values['AUGMENT_MODE'] == 'manual' and values['RUN_NAME'] == 'run03_event_200'
    for path in folder.glob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print('训练种子、参数、两阶段循环、选模保存逻辑沿用原版；语法通过。')


if __name__ == '__main__':
    check_data()
    check_contrasts()
    check_recipe()
    print('全部检查通过，未下载或加载预训练权重，未执行训练。')
