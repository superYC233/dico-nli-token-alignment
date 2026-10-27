"""比较两个训练阶段的预测；直接运行可重新生成诊断表，不加载模型。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.deberta_lexical_stage_probe import config as cfg


def error_counts(gold, prediction):
    return {
        'E_to_FB': int(((gold == 'E') & prediction.isin(['F', 'B'])).sum()),
        'FB_to_E': int((gold.isin(['F', 'B']) & (prediction == 'E')).sum()),
        'F_B_reversed': int((((gold == 'F') & (prediction == 'B'))
                             | ((gold == 'B') & (prediction == 'F'))).sum()),
        'N_to_nonN': int(((gold == 'N') & (prediction != 'N')).sum()),
        'nonN_to_N': int(((gold != 'N') & (prediction == 'N')).sum()),
    }


def align_predictions(before_path, after_path):
    before = pd.read_csv(before_path, keep_default_na=False)
    after = pd.read_csv(after_path, keep_default_na=False)
    for frame in (before, after):
        if not frame['instance_id'].is_unique:
            raise ValueError('预测中 instance_id 重复。')
    if set(before['instance_id']) != set(after['instance_id']):
        raise ValueError('两个阶段的样本编号不同。')
    before = before.set_index('instance_id').loc[after['instance_id']].reset_index()
    columns = ['instance_id', 'pair_id', 'text1', 'text2', 'gold_label']
    if not before[columns].equals(after[columns]):
        raise ValueError('对齐后，两个阶段的文本、金标或 pair_id 不同。')
    for frame in (before, after):
        if not frame['gold_label'].isin(cfg.FULL_LABELS).all():
            raise ValueError('预测文件中的金标未知。')
        for column in ['prediction', 'pair_prediction']:
            if not frame[column].isin(cfg.LABELS).all():
                raise ValueError('预测文件中出现未知预测标签。')
    return before, after


def compare_stages(before_path, after_path, before_summary, after_summary,
                   output_dir, epoch, training_phase):
    before, after = align_predictions(before_path, after_path)
    full_to_short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    gold = after['gold_label'].map(full_to_short)
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        old_prediction = before[column]
        new_prediction = after[column]
        old_correct = old_prediction == gold
        new_correct = new_prediction == gold
        changed = old_prediction != new_prediction
        fixed = ~old_correct & new_correct
        broken = old_correct & ~new_correct
        changed_wrong = changed & ~old_correct & ~new_correct

        old_metrics = before_summary.set_index('method').loc[method]
        new_metrics = after_summary.set_index('method').loc[method]
        row = {
            'epoch': epoch, 'training_phase': training_phase, 'method': method,
            'weighted_f1_before': float(old_metrics['weighted_f1']),
            'weighted_f1_after': float(new_metrics['weighted_f1']),
            'delta_f1_pp': 100 * float(new_metrics['weighted_f1'] - old_metrics['weighted_f1']),
            'changed': int(changed.sum()), 'fixed': int(fixed.sum()),
            'broken': int(broken.sum()), 'changed_wrong': int(changed_wrong.sum()),
            'net_correct': int(fixed.sum() - broken.sum()),
        }
        for label in cfg.LABELS:
            old_count = int(((gold == label) & old_correct).sum())
            new_count = int(((gold == label) & new_correct).sum())
            row[f'{label}_correct_before'] = old_count
            row[f'{label}_correct_after'] = new_count
            row[f'{label}_delta_correct'] = new_count - old_count
        for name, count in error_counts(gold, old_prediction).items():
            row[f'{name}_before'] = count
        for name, count in error_counts(gold, new_prediction).items():
            row[f'{name}_after'] = count
        records.append(row)

        table = after[['instance_id', 'pair_id', 'text1', 'text2', 'gold_label']].copy()
        table['before_prediction'] = old_prediction
        table['after_prediction'] = new_prediction
        table['before_correct'] = old_correct
        table['after_correct'] = new_correct
        folder = Path(output_dir) / method
        folder.mkdir(parents=True, exist_ok=True)
        for name, mask in [('changed', changed), ('fixed', fixed), ('broken', broken),
                           ('changed_wrong', changed_wrong)]:
            table.loc[mask].to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    return records


def compare_with_reference(results_dir):
    # 只比较相同 epoch 的 after_pair，不用 after_row 参与原协议的选模。
    reference_dir = cfg.REFERENCE_RESULTS_DIR
    if not (reference_dir / 'epoch_summary.csv').exists():
        print('未找到旧词汇分支的逐轮结果，跳过复现检查。')
        return
    saved = json.loads((results_dir / 'run_config.json').read_text(encoding='utf-8'))
    old_saved = json.loads((reference_dir / 'run_config.json').read_text(encoding='utf-8'))
    keys = [
        'model_name', 'seed', 'use_lexical_features', 'feature_names', 'feature_version',
        'lexical_hidden_size', 'train_sha256', 'dev_sha256', 'train_rows', 'dev_rows',
        'real_pairs', 'data_augmentation', 'epochs', 'row_batch_size', 'pair_batch_size',
        'dev_batch_size', 'max_length', 'learning_rate', 'lexical_learning_rate',
        'weight_decay', 'row_weights', 'pair_weights', 'pair_weight',
        'pair_label_smoothing', 'dtype', 'labels', 'training_order', 'selection',
    ]
    different_keys = [key for key in keys if saved.get(key) != old_saved.get(key)]
    rows = []
    for epoch in range(1, cfg.EPOCHS + 1):
        old_path = reference_dir / f'epoch_{epoch}/predictions_with_scores.csv'
        new_path = results_dir / f'epoch_{epoch}/after_pair/predictions_with_scores.csv'
        if not old_path.exists() or not new_path.exists():
            continue
        before, after = align_predictions(old_path, new_path)
        logit_columns = [f'logit_{label}' for label in cfg.LABELS]
        max_difference = float(np.abs(
            before[logit_columns].to_numpy() - after[logit_columns].to_numpy(),
        ).max())
        rows.append({
            'epoch': epoch,
            'independent_differences': int((before['prediction'] != after['prediction']).sum()),
            'pair_aware_differences': int((before['pair_prediction'] != after['pair_prediction']).sum()),
            'max_abs_logit_difference': max_difference,
        })
    pd.DataFrame(rows).to_csv(
        results_dir / 'reference_reproduction.csv', index=False, encoding='utf-8-sig',
    )
    (results_dir / 'reference_protocol_check.json').write_text(
        json.dumps({'same_protocol': not different_keys, 'different_keys': different_keys,
                    'reference_dir': str(reference_dir)}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    if different_keys:
        print('与旧词汇分支存在参数差异：', different_keys)
    if rows:
        print('\n===== 与旧词汇分支的同轮 after_pair 结果比较 =====')
        print(pd.DataFrame(rows).to_string(index=False))


def print_stage_table(stage_summary):
    selected = stage_summary[stage_summary['method'] == 'pair_aware']
    columns = ['epoch', 'weighted_f1', 'F_correct', 'B_correct', 'E_correct', 'N_correct']
    row = selected[selected['stage'] == 'after_row'][columns]
    pair = selected[selected['stage'] == 'after_pair'][columns]
    table = row.merge(pair, on='epoch', suffixes=('_after_row', '_after_pair'), validate='one_to_one')
    table['pair_delta_f1_pp'] = 100 * (table['weighted_f1_after_pair'] - table['weighted_f1_after_row'])
    print('\n===== 每轮两个阶段（均为 pair-aware 评分）=====')
    print(table.to_string(index=False))


def main():
    _, results_dir = cfg.get_output_dirs(cfg.USE_LEXICAL_FEATURES)
    history = pd.read_csv(results_dir / 'stage_summary.csv', keep_default_na=False)
    records = []
    previous_dir = None
    previous_summary = None
    for epoch in sorted(history['epoch'].unique()):
        row_dir = results_dir / f'epoch_{epoch}/after_row'
        pair_dir = results_dir / f'epoch_{epoch}/after_pair'
        if not (pair_dir / 'summary.csv').exists():
            continue
        row_summary = pd.read_csv(row_dir / 'summary.csv')
        pair_summary = pd.read_csv(pair_dir / 'summary.csv')
        if previous_dir is not None:
            records.extend(compare_stages(
                previous_dir / 'predictions_with_scores.csv', row_dir / 'predictions_with_scores.csv',
                previous_summary, row_summary, row_dir.parent / 'row_changes', epoch, 'row_ce',
            ))
        records.extend(compare_stages(
            row_dir / 'predictions_with_scores.csv', pair_dir / 'predictions_with_scores.csv',
            row_summary, pair_summary, row_dir.parent / 'pair_changes', epoch, 'pair_ce',
        ))
        previous_dir, previous_summary = pair_dir, pair_summary
    pd.DataFrame(records).to_csv(results_dir / 'phase_effects.csv', index=False, encoding='utf-8-sig')
    print_stage_table(history)
    compare_with_reference(results_dir)
    print('诊断目录：', results_dir)


if __name__ == '__main__':
    main()
