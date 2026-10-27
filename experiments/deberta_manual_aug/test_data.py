"""数据、隔离规则与训练配方检查；不加载模型，不训练。"""
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug import prepare_manual_data as data


def check_preparation():
    # 直接复核已准备好的文件；检查过程不另建临时目录、不改动训练数据。
    combined = pd.read_csv(data.PREPARED_DIR / 'training_data_manual.csv', keep_default_na=False)
    manual = pd.read_csv(data.PREPARED_DIR / 'manual_rows.csv', keep_default_na=False)
    report = json.loads((data.PREPARED_DIR / 'overlap_report.json').read_text(encoding='utf-8'))
    assert len(manual) == 80 and len(combined) == 3322
    assert manual.label.value_counts().to_dict() == {label: 20 for label in data.FULL.values()}
    data.validate_extra(combined)
    assert combined.instance_id.is_unique
    assert (combined.groupby('pair_id').size() == 2).sum() == 1392
    assert math.ceil(len(combined) / 16) == 208
    assert math.ceil(1392 / 16) == 87
    baseline = pd.read_csv(data.REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    assert combined.iloc[:3242].reset_index(drop=True).equals(baseline)
    assert set(manual.instance_id).isdisjoint(baseline.instance_id)
    assert set(manual.pair_id).isdisjoint(baseline.pair_id)
    notes = pd.read_csv(data.PREPARED_DIR / 'manual_label_notes.csv', keep_default_na=False)
    assert set(notes.instance_id) == set(manual.instance_id)
    assert notes.label_reason.str.len().gt(0).all()
    assert notes.annotation_status.str.contains('not independently').all()
    control = pd.read_csv(data.PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    assert len(control) == len(manual)
    assert control.label.value_counts().to_dict() == manual.label.value_counts().to_dict()
    data.validate_extra(control)
    assert (control.groupby('pair_id').size() == 2).sum() == 30
    assert all(value == 0 for key, value in report['new_manual_vs_dev'].items() if key.endswith('_count'))
    dev_text = pd.read_csv(data.DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    assert data.overlap_report(manual, dev_text) == report['new_manual_vs_dev']
    preparation = ast.parse(Path(data.__file__).read_text(encoding='utf-8'))
    dev_reads = [node for node in ast.walk(preparation) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_csv'
                 and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'DEV_PATH']
    assert len(dev_reads) == 1
    assert next(ast.literal_eval(item.value) for item in dev_reads[0].keywords
                if item.arg == 'usecols') == ['text1', 'text2']
    print('80行新人工、旧3242行不变、3322行总数、标签理由、复制预算与Dev只读文本均通过。')


def check_screening():
    dev = pd.DataFrame({'text1': ['A visited station'], 'text2': ['the station']})
    baseline = pd.DataFrame({'text1': ['existing'], 'text2': ['pair']})
    source = pd.DataFrame([
        {'example_id': 'a', 'text1': 'a flask', 'text2': 'urn', 'label': 'N'},
        {'example_id': 'b', 'text1': 'visiting flasks', 'text2': 'urns', 'label': 'N'},
        {'example_id': 'c', 'text1': 'hexagonal flasks', 'text2': 'flasks', 'label': 'F'},
        {'example_id': 'd', 'text1': 'flasks', 'text2': 'hexagonal flasks', 'label': 'B'},
        {'example_id': 'e', 'text1': 'pair', 'text2': 'existing', 'label': 'N'},
    ])
    accepted, rejected = data.screen_candidates(source, dev, baseline)
    assert accepted.example_id.tolist() == ['c']
    assert set(rejected.example_id) == {'a', 'b', 'd', 'e'}
    assert '词干' in rejected.set_index('example_id').loc['b', 'rejection_reason']
    assert data.words('Ａ FLASKS—URＮS 42') == ['a', 'flasks', 'urns', '42']
    assert data.pair_key('A Flask', 'urn') == data.pair_key('URN', 'a flask')
    print('冠词也拒绝、词干变体拒绝、正反向重复拒绝、Unicode规范化通过。')


def check_recipe():
    folder = Path(__file__).resolve().parent
    original = ast.parse((ROOT / 'experiments/deberta_original_aug/train_original_aug.py').read_text(encoding='utf-8'))
    current = ast.parse((folder / 'train_original_manual.py').read_text(encoding='utf-8'))
    for name in ['set_seed', 'RealPairDataset', 'train_rows', 'train_pairs', 'evaluate', 'save_weights']:
        before = next(node for node in original.body if getattr(node, 'name', None) == name)
        after = next(node for node in current.body if getattr(node, 'name', None) == name)
        assert ast.dump(before) == ast.dump(after), name
    saved = json.loads((data.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    values = {}
    for node in current.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    for name in ['seed', 'model_name', 'epochs', 'max_length', 'learning_rate', 'weight_decay',
                 'row_batch_size', 'pair_batch_size', 'dev_batch_size', 'row_weights',
                 'pair_weights', 'pair_weight', 'pair_label_smoothing']:
        assert values[name.upper()] == saved[name], name
    assert values['AUGMENT_MODE'] == 'manual'
    for path in folder.glob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    print('0.935配方参数和两阶段核心函数未改；全部新文件语法通过。')


if __name__ == '__main__':
    check_preparation()
    check_screening()
    check_recipe()
    print('检查完成，没有加载模型或执行训练。')
