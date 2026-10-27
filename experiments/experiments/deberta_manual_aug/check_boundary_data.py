"""只检查修订数据、隔离与原训练配方；不加载模型，不训练。"""
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_boundary_data as boundary
from experiments.deberta_manual_aug import prepare_manual_data as shared


def check_data():
    source = pd.read_csv(boundary.SOURCE_PATH, keep_default_na=False)
    manual = pd.read_csv(boundary.PREPARED_DIR / 'manual_rows.csv', keep_default_na=False)
    full = pd.read_csv(boundary.PREPARED_DIR / 'training_data_manual.csv', keep_default_na=False)
    control = pd.read_csv(boundary.PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    copied = pd.read_csv(boundary.PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
    baseline = pd.read_csv(boundary.REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    notes = pd.read_csv(boundary.PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    assert len(source) == 72 and len(manual) == len(control) == 120
    assert len(full) == len(copied) == 3362
    assert full.iloc[:3242].reset_index(drop=True).equals(baseline)
    assert copied.iloc[:3242].reset_index(drop=True).equals(baseline)
    expected = {shared.FULL['F']: 24, shared.FULL['B']: 24, shared.FULL['E']: 48, shared.FULL['N']: 24}
    assert manual.label.value_counts().to_dict() == control.label.value_counts().to_dict() == expected
    for frame in [manual, full, control, copied]:
        shared.validate_extra(frame)
    assert (manual.groupby('pair_id').size() == 2).sum() == 48
    assert (manual.groupby('pair_id').size() == 1).sum() == 24
    pair_targets = [tuple(group.label) for _, group in manual.groupby('pair_id') if len(group) == 2]
    assert pair_targets.count((shared.FULL['F'], shared.FULL['B'])) == 12
    assert pair_targets.count((shared.FULL['B'], shared.FULL['F'])) == 12
    assert pair_targets.count((shared.FULL['E'], shared.FULL['E'])) == 24
    assert (full.groupby('pair_id').size() == 2).sum() == 1410
    assert math.ceil(len(full) / 16) == 211 and math.ceil(1410 / 16) == 89
    assert set(manual.instance_id).isdisjoint(baseline.instance_id)
    assert set(manual.pair_id).isdisjoint(baseline.pair_id)
    assert set(notes.instance_id) == set(manual.instance_id)
    assert notes.label_reason.str.len().gt(0).all()
    dev_text = pd.read_csv(shared.DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    audit = shared.overlap_report(manual, dev_text)
    assert all(value == 0 for key, value in audit.items() if key.endswith('_count'))
    report = json.loads((boundary.PREPARED_DIR / 'overlap_report.json').read_text(encoding='utf-8'))
    assert report['new_manual_vs_dev'] == audit
    boundary.check_relations(source)
    print('120新增行、3362总行、FB12/BF12/EE24个新增组、24个新增N单例和复制预算通过。')
    print('新增与Dev完整词形、词干、整短语和正反文本对交集均为0。')


def check_design():
    notes = pd.read_csv(boundary.PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    audit, design = boundary.design_audit(notes)
    assert design['FBE_direction_rows_with_equal_word_counts'] > 0
    assert design['F_rows_where_left_is_shorter'] > 0
    assert design['E_rows_with_different_word_counts'] > 0
    assert design['three_word_same_bag_E_rows'] > 0
    assert design['three_word_same_bag_N_rows'] > 0
    assert design['old_surface_rule_accuracy'] < .5
    saved = pd.read_csv(boundary.PREPARED_DIR / 'design_audit.csv', keep_default_na=False)
    assert saved.equals(audit)
    # 特别检查能够识别间接矛盾，而不仅是同一文本对直接改标。
    conflict = pd.DataFrame([
        {'example_id': 'test_1', 'text1': 'specific', 'text2': 'general', 'label': 'F'},
        {'example_id': 'test_2', 'text1': 'specific', 'text2': 'equivalent', 'label': 'E'},
        {'example_id': 'test_3', 'text1': 'equivalent', 'text2': 'general', 'label': 'N'},
    ])
    try:
        boundary.check_relations(conflict)
    except AssertionError:
        pass
    else:
        raise AssertionError('应检出F/E关系链与N之间的间接冲突。')
    print('同长F/B、不同长E、短侧蕴含长侧、同长同词集合的E/N均存在。')
    print('旧表面规则：', design['old_surface_rule_correct'], '/', len(notes), '；未把此数当作语义质量证明。')


def check_recipe():
    folder = Path(__file__).resolve().parent
    before = ast.parse((folder / 'train_event_manual.py').read_text(encoding='utf-8'))
    after = ast.parse((folder / 'train_boundary_manual.py').read_text(encoding='utf-8'))
    for name in ['set_seed', 'RealPairDataset', 'train_rows', 'train_pairs', 'evaluate', 'save_weights', 'compare_with_reference']:
        old = next(node for node in before.body if getattr(node, 'name', None) == name)
        new = next(node for node in after.body if getattr(node, 'name', None) == name)
        assert ast.dump(old) == ast.dump(new), name
    settings = {}
    for node in after.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                settings[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    reference = json.loads((boundary.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    for name in ['seed', 'model_name', 'epochs', 'max_length', 'learning_rate', 'weight_decay',
                 'row_batch_size', 'pair_batch_size', 'dev_batch_size', 'row_weights',
                 'pair_weights', 'pair_weight', 'pair_label_smoothing']:
        assert settings[name.upper()] == reference[name], name
    assert settings['RUN_NAME'] == 'run04_boundary_120' and settings['AUGMENT_MODE'] == 'manual'
    for name in ['prepare_boundary_data.py', 'train_boundary_manual.py', 'check_boundary_data.py']:
        path = folder / name
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print('模型与Row/Pair训练核心、种子、参数、评估和最佳保存逻辑保持原配方；语法通过。')


if __name__ == '__main__':
    check_data()
    check_design()
    check_recipe()
    print('检查完成，未加载预训练权重，未执行训练。')
