"""从已保存的 CSV 分析阶段变化，不加载模型或训练。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# 这里只复用逐条对齐、修好/破坏与错误类型计数；不使用旧词汇实验的训练配置。
from experiments.deberta_lexical_stage_probe.analyze_stages import (
    align_predictions, compare_stages,
)


def compare_reference(results_dir, reference_dir):
    old_config_path = reference_dir / 'run_config.json'
    new_config_path = results_dir / 'run_config.json'
    if not old_config_path.exists() or not new_config_path.exists():
        return {'status': 'reference_or_current_config_not_available'}
    old = json.loads(old_config_path.read_text(encoding='utf-8'))
    new = json.loads(new_config_path.read_text(encoding='utf-8'))
    keys = ['model_name', 'seed', 'epochs', 'max_length', 'dtype', 'learning_rate',
            'weight_decay', 'row_batch_size', 'pair_batch_size', 'dev_batch_size',
            'row_weights', 'pair_weights', 'pair_weight', 'pair_label_smoothing',
            'freeze_embeddings', 'freeze_first_n_layers', 'use_lexical_features',
            'real_pairs', 'row_steps_per_epoch', 'pair_steps_per_epoch',
            'trainable_parameters', 'total_parameters', 'training_order', 'selection',
            'train_sha256', 'dev_sha256']
    differences = {key: {'reference': old.get(key), 'current': new.get(key)}
                   for key in keys if old.get(key) != new.get(key)}
    old_hash = old.get('augmentation', {}).get('effective_train_sha256')
    new_hash = new.get('augmentation', {}).get('effective_train_sha256')
    if old_hash != new_hash:
        differences['effective_train_sha256'] = {'reference': old_hash, 'current': new_hash}

    rows = []
    for folder in sorted(results_dir.glob('epoch_*')):
        epoch = int(folder.name.split('_')[-1])
        old_path = reference_dir / f'epoch_{epoch}/predictions_with_scores.csv'
        new_path = folder / 'after_pair/predictions_with_scores.csv'
        if not old_path.exists() or not new_path.exists():
            continue
        before, after = align_predictions(old_path, new_path)
        columns = ['logit_F', 'logit_B', 'logit_E', 'logit_N']
        rows.append({
            'epoch': epoch,
            'independent_differences': int((before.prediction != after.prediction).sum()),
            'pair_aware_differences': int((before.pair_prediction != after.pair_prediction).sum()),
            'max_abs_logit_difference': float(np.abs(before[columns].to_numpy() -
                                                       after[columns].to_numpy()).max()),
        })
    if rows:
        pd.DataFrame(rows).sort_values('epoch').to_csv(
            results_dir / 'reference_reproduction.csv', index=False, encoding='utf-8-sig',
        )
    result = {
        'status': 'checked', 'reference_dir': str(reference_dir),
        'same_recorded_protocol': not differences, 'protocol_differences': differences,
        'compared_epochs': len(rows),
        'all_compared_predictions_match': all(row['independent_differences'] == 0 and
                                             row['pair_aware_differences'] == 0 for row in rows) if rows else None,
        'limitation': 'RNG isolation does not guarantee identical GPU floating-point results.',
    }
    (results_dir / 'reference_protocol_check.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    return result


def summarize_run(results_dir, reference_dir):
    results_dir, reference_dir = Path(results_dir), Path(reference_dir)
    history = pd.read_csv(results_dir / 'stage_summary.csv')
    phase_effects = pd.read_csv(results_dir / 'phase_effects.csv')
    records = []
    for epoch in sorted(history.epoch.unique()):
        current = history[history.epoch == epoch]
        if set(current.stage) != {'after_row', 'after_pair'}:
            continue  # 中途停下的最后半轮不拿来作前后比较。
        row = current[current.stage == 'after_row'].set_index('method')
        pair = current[current.stage == 'after_pair'].set_index('method')
        effects = phase_effects[(phase_effects.epoch == epoch) &
                                (phase_effects.training_phase == 'pair_ce')].set_index('method')
        change = effects.loc['independent']
        records.append({
            'epoch': int(epoch),
            'row_ind_f1': float(row.loc['independent', 'weighted_f1']),
            'pair_ind_f1': float(pair.loc['independent', 'weighted_f1']),
            'pair_ind_delta_pp': float(change.delta_f1_pp),
            'row_nonN_to_N': int(change.nonN_to_N_before),
            'pair_nonN_to_N': int(change.nonN_to_N_after),
            'row_N_correct': int(row.loc['independent', 'N_correct']),
            'pair_N_correct': int(pair.loc['independent', 'N_correct']),
            'row_pair_f1': float(row.loc['pair_aware', 'weighted_f1']),
            'pair_pair_f1': float(pair.loc['pair_aware', 'weighted_f1']),
            'pair_fixed': int(change.fixed), 'pair_broken': int(change.broken),
        })
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError('尚没有完整的 Row / Pair 前后结果。')
    table.to_csv(results_dir / 'stage_comparison.csv', index=False, encoding='utf-8-sig')
    reproduction = compare_reference(results_dir, reference_dir)
    worst = table.loc[table.pair_ind_delta_pp.idxmin()]
    best_index = int(table.pair_pair_f1.idxmax())
    selected = table.loc[best_index]
    lines = [
        '# 原版增强模型：Row / Pair 阶段诊断', '',
        f'- 完整比较轮数：{len(table)}。',
        f'- Pair CE 后独立 F1 下降的轮数：{int((table.pair_ind_delta_pp < 0).sum())}。',
        f'- Pair CE 后独立 F1 最大下降/最小变化：第 {int(worst.epoch)} 轮，{worst.pair_ind_delta_pp:+.2f} 个百分点。',
        f'- 原协议选出的最佳轮：第 {int(selected.epoch)} 轮 after_pair，pair-aware F1={selected.pair_pair_f1:.6f}。',
        f'- 该轮独立 F1：Row 后 {selected.row_ind_f1:.6f} → Pair 后 {selected.pair_ind_f1:.6f}。',
        f'- 该轮非 N→N：Row 后 {int(selected.row_nonN_to_N)} → Pair 后 {int(selected.pair_nonN_to_N)}。',
        f'- 该轮 N 正确数：Row 后 {int(selected.row_N_correct)} → Pair 后 {int(selected.pair_N_correct)}。', '',
        '## 怎样读结果', '',
        '1. 若 Row 后独立判断较好，Pair 后明显下降且非N→N增加，说明本次 Pair 更新阶段损害了四分类表现。',
        '2. 若 Row 后已经较差，问题在进入该轮 Pair 阶段前就存在；第二轮起可看 row_changes 检查 Row 是否修复了上一轮的退化。',
        '3. 若两阶段独立判断都较好，本次未复现历史问题，先检查 reference_reproduction.csv。',
        '4. 阶段差异不能单独证明过拟合、标签噪声或某个梯度机制；这些原因还需独立对照。', '',
        '## 文件', '',
        '- stage_comparison.csv：每轮两个阶段的独立 F1、结构 F1、非N→N、N 正确数。',
        '- phase_effects.csv：Row / Pair 更新各自修好和破坏哪些类别、多少行。',
        '- epoch_*/pair_changes/independent/broken.csv：本轮 Pair 更新改错的具体短语。',
        '- epoch_*/row_changes：上一轮 Pair 后 → 本轮 Row 后；第一轮没有此对照。',
        '- 每个 after_row / after_pair 都保存逐条分数、两种解码和混淆矩阵。',
        '- 仅 after_pair 的 pair-aware F1 参与权重保存；after_row 仅用于诊断。', '',
        '## 与历史 0.935493 运行的复现核对', '',
        f'- {json.dumps(reproduction, ensure_ascii=False)}', '',
        '额外评估已隔离随机数；若同轮预测不同，这份报告只能诊断本次训练，不能视作历史权重的逐步回放。',
        '本报告是同一训练轨迹的阶段观测，不是额外随机种子的复验。',
    ]
    (results_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n===== 每轮 Row / Pair 前后对比 =====')
    print(table.to_string(index=False))
    print('\n与历史运行的记录配置一致：', reproduction.get('same_recorded_protocol'))
    print('同轮独立/结构预测全部一致：', reproduction.get('all_compared_predictions_match'))
    return table


def analyze_saved(results_dir, reference_dir):
    # 可独立重跑分析；不读权重，只使用已经保存的预测和分数。
    results_dir = Path(results_dir)
    records, effects = [], []
    previous_dir, previous_summary = None, None
    epoch_dirs = sorted(results_dir.glob('epoch_*'), key=lambda path: int(path.name.split('_')[-1]))
    for folder in epoch_dirs:
        epoch = int(folder.name.split('_')[-1])
        row_dir, pair_dir = folder / 'after_row', folder / 'after_pair'
        if not (row_dir / 'summary.csv').exists() or not (pair_dir / 'summary.csv').exists():
            continue
        row_summary = pd.read_csv(row_dir / 'summary.csv')
        pair_summary = pd.read_csv(pair_dir / 'summary.csv')
        for stage, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, stage=stage)
                records.append(record)
        if previous_dir is not None:
            effects.extend(compare_stages(
                previous_dir / 'predictions_with_scores.csv', row_dir / 'predictions_with_scores.csv',
                previous_summary, row_summary, folder / 'row_changes', epoch, 'row_ce',
            ))
        effects.extend(compare_stages(
            row_dir / 'predictions_with_scores.csv', pair_dir / 'predictions_with_scores.csv',
            row_summary, pair_summary, folder / 'pair_changes', epoch, 'pair_ce',
        ))
        previous_dir, previous_summary = pair_dir, pair_summary
    if not records:
        raise FileNotFoundError('没有完整阶段结果，请先运行 train_stage_probe.py。')
    pd.DataFrame(records).to_csv(results_dir / 'stage_summary_rebuilt.csv', index=False, encoding='utf-8-sig')
    # 保留训练日志里的 row_loss/pair_loss，重建分析不覆盖原 stage_summary。
    if not (results_dir / 'stage_summary.csv').exists():
        pd.DataFrame(records).to_csv(results_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(effects).to_csv(results_dir / 'phase_effects.csv', index=False, encoding='utf-8-sig')
    return summarize_run(results_dir, reference_dir)


if __name__ == '__main__':
    from experiments.deberta_original_aug_stage_probe.train_stage_probe import RESULTS_DIR, REFERENCE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    analyze_saved(args.results_dir, args.reference_dir)
