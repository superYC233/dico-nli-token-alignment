"""零训练分析：先看历史 seed42，完成新种子后自动加入配对比较。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.deberta_lexical_reliability import config as cfg
from experiments.deberta_lexical_reliability.protocol import ensure_protocol


def weighted_f1(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    correct = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = support + predicted
    f1 = np.divide(2 * correct, denominator, out=np.zeros_like(correct), where=denominator > 0)
    total = support.sum(axis=-1)
    return np.divide((f1 * support).sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)


def load_prediction(path, dev):
    frame = pd.read_csv(path, keep_default_na=False)
    if not frame['instance_id'].is_unique or set(frame['instance_id']) != set(dev['instance_id']):
        raise ValueError(f'{path} 的样本编号不正确。')
    frame = frame.set_index('instance_id').loc[dev['instance_id']]
    if 'gold_label' in frame and not np.array_equal(frame['gold_label'].to_numpy(), dev['label'].to_numpy()):
        raise ValueError('预测文件的金标与官方 Dev 不同。')
    if 'pair_id' in frame and not np.array_equal(frame['pair_id'].to_numpy(), dev['pair_id'].to_numpy()):
        raise ValueError('预测文件的 pair_id 与官方 Dev 不同。')
    column = 'pair_prediction' if 'pair_prediction' in frame else 'label'
    mapping = dict(cfg.LABEL_TO_ID)
    mapping.update({label: index for index, label in enumerate(cfg.LABELS)})
    predictions = frame[column].map(mapping)
    if predictions.isna().any():
        raise ValueError('预测文件出现未知类别。')
    return predictions.to_numpy(dtype=np.int64)


def group_confusion(gold, predictions, groups):
    result = np.zeros((int(groups.max()) + 1, 4, 4), dtype=np.float64)
    np.add.at(result, (groups, gold, predictions), 1)
    return result


def paired_group_bootstrap(group_matrices, repeats, seed):
    # [模型数, 组数, 4, 4]。所有模型共享完全相同的组重采样。
    matrices = np.asarray(group_matrices, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[-2:] != (4, 4):
        raise ValueError('bootstrap 输入形状错误。')
    if repeats <= 0:
        raise ValueError('bootstrap 次数必须为正。')
    model_count, group_count = matrices.shape[:2]
    by_group = matrices.transpose(1, 0, 2, 3).reshape(group_count, model_count * 16)
    rng = np.random.default_rng(seed)
    scores = np.empty((repeats, model_count), dtype=np.float64)
    for start in range(0, repeats, 200):
        size = min(200, repeats - start)
        selected = rng.integers(0, group_count, size=(size, group_count))
        sampled = by_group[selected].sum(axis=1).reshape(size, model_count, 4, 4)
        scores[start:start + size] = weighted_f1(sampled)
    return scores


def load_runs(dev, protocol_hash):
    historical = cfg.PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42'
    original_config = json.loads((cfg.PROJECT_ROOT / 'artifacts/final/config.json').read_text(encoding='utf-8'))
    runs = [{
        'seed': 42, 'source': 'historical_selected',
        'baseline_path': cfg.PROJECT_ROOT / 'artifacts/final/dev_predictions_pair_aware.csv',
        'lexical_path': historical / 'epoch_3/predictions_with_scores.csv',
        'baseline_epoch': original_config['selected_epoch'], 'lexical_epoch': 3,
        'baseline_final_f1': None, 'lexical_final_f1': None,
    }]
    pending = []
    for seed in cfg.SEEDS:
        record = {'seed': seed, 'source': 'new_paired'}
        complete = True
        for arm in ['baseline', 'lexical']:
            folder = cfg.RESULTS_ROOT / cfg.RUN_NAME / f'{arm}_seed_{seed}'
            marker = folder / 'completed.json'
            if not marker.is_file():
                pending.append(f'{arm}_seed_{seed}')
                complete = False
                continue
            info = json.loads(marker.read_text(encoding='utf-8'))
            if info['protocol_sha256'] != protocol_hash or info['seed'] != seed or info['arm'] != arm:
                raise ValueError('新种子的运行记录与冻结协议不同。')
            saved = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
            if saved['seed'] != seed or saved['use_lexical_features'] != (arm == 'lexical'):
                raise ValueError('运行种子或版本不正确。')
            history = pd.read_csv(folder / 'epoch_summary.csv')
            paired = history[history['method'] == 'pair_aware']
            if sorted(paired['epoch'].tolist()) != list(range(1, cfg.EPOCHS + 1)):
                raise ValueError('轮次记录不完整。')
            best = paired.sort_values(['weighted_f1', 'epoch'], ascending=[False, True]).iloc[0]
            record[f'{arm}_epoch'] = int(best['epoch'])
            record[f'{arm}_path'] = folder / f"epoch_{int(best['epoch'])}" / 'predictions_with_scores.csv'
            record[f'{arm}_reported_f1'] = float(best['weighted_f1'])
            record[f'{arm}_final_f1'] = float(paired.loc[paired['epoch'] == cfg.EPOCHS, 'weighted_f1'].iloc[0])
        if complete:
            runs.append(record)
    return runs, pending


def main(repeats=cfg.BOOTSTRAP_REPEATS):
    protocol_hash = ensure_protocol()
    output = cfg.RESULTS_ROOT / cfg.RUN_NAME / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    gold = dev['label'].map(cfg.LABEL_TO_ID).to_numpy(dtype=np.int64)
    groups, group_ids = pd.factorize(dev['pair_id'], sort=False)
    runs, pending = load_runs(dev, protocol_hash)
    matrices, rows = [], []
    for run in runs:
        row = {key: run[key] for key in ['seed', 'source', 'baseline_epoch', 'lexical_epoch',
                                        'baseline_final_f1', 'lexical_final_f1']}
        predictions = {}
        for arm in ['baseline', 'lexical']:
            predictions[arm] = load_prediction(run[f'{arm}_path'], dev)
            grouped = group_confusion(gold, predictions[arm], groups)
            matrices.append(grouped)
            score = float(weighted_f1(grouped.sum(axis=0)))
            if f'{arm}_reported_f1' in run and abs(score - run[f'{arm}_reported_f1']) > 1e-12:
                raise ValueError('重算 F1 与训练时报告不同。')
            row[f'{arm}_f1'] = score
            row[f'{arm}_correct'] = int((predictions[arm] == gold).sum())
            row[f'{arm}_E_correct'] = int(((gold == 2) & (predictions[arm] == 2)).sum())
            row[f'{arm}_N_correct'] = int(((gold == 3) & (predictions[arm] == 3)).sum())
        before, after = predictions['baseline'] == gold, predictions['lexical'] == gold
        row['fixed'] = int((~before & after).sum())
        row['broken'] = int((before & ~after).sum())
        row['delta_pp'] = 100 * (row['lexical_f1'] - row['baseline_f1'])
        rows.append(row)
    table = pd.DataFrame(rows)
    # 每个种子两份模型；383 个组按整体抽样，不把可逆组两行拆开。
    print(f'Dev：{len(dev)} 行，{len(group_ids)} 个 pair_id；bootstrap：{repeats} 次。')
    samples = paired_group_bootstrap(matrices, repeats, cfg.BOOTSTRAP_SEED)
    intervals, deltas = [], []
    for i, run in enumerate(runs):
        delta = 100 * (samples[:, 2 * i + 1] - samples[:, 2 * i])
        lower, upper = np.percentile(delta, [2.5, 97.5])
        lex_lower, lex_upper = np.percentile(samples[:, 2 * i + 1], [2.5, 97.5])
        intervals.append({
            'comparison': f"seed_{run['seed']}_lexical_minus_baseline",
            'delta_pp': rows[i]['delta_pp'], 'lower_pp': lower, 'upper_pp': upper,
            'lexical_f1_lower': lex_lower, 'lexical_f1_upper': lex_upper,
            'scope': 'fixed fitted models and current Dev; no training-seed variance or selection correction',
        })
        if run['source'] == 'new_paired':
            deltas.append(delta)
    if len(deltas) == len(cfg.SEEDS):
        mean_delta = np.stack(deltas).mean(axis=0)
        lower, upper = np.percentile(mean_delta, [2.5, 97.5])
        intervals.append({
            'comparison': 'PRIMARY_new_seeds_mean_delta',
            'delta_pp': table.loc[table['source'] == 'new_paired', 'delta_pp'].mean(),
            'lower_pp': lower, 'upper_pp': upper,
            'scope': 'two fixed seed pairs; shared Dev resampling; not a population CI over training seeds',
        })
    interval_table = pd.DataFrame(intervals)
    table.to_csv(output / 'per_seed.csv', index=False, encoding='utf-8-sig')
    interval_table.to_csv(output / 'bootstrap.csv', index=False, encoding='utf-8-sig')
    print(table[['seed', 'source', 'baseline_epoch', 'lexical_epoch', 'baseline_f1', 'lexical_f1',
                 'delta_pp', 'fixed', 'broken']].to_string(index=False))
    print('\n===== 固定模型的 Dev 组抽样区间，未做多重比较校正 =====')
    print(interval_table[['comparison', 'delta_pp', 'lower_pp', 'upper_pp']].to_string(index=False))
    summaries = []
    for scope, selected in [('all_available_descriptive', table),
                            ('new_seeds_only', table[table['source'] == 'new_paired'])]:
        if selected.empty:
            continue
        summaries.append({
            'scope': scope, 'seed_count': len(selected),
            'baseline_mean': selected['baseline_f1'].mean(),
            'baseline_std': selected['baseline_f1'].std(ddof=1),
            'lexical_mean': selected['lexical_f1'].mean(),
            'lexical_std': selected['lexical_f1'].std(ddof=1),
            'paired_delta_mean_pp': selected['delta_pp'].mean(),
            'paired_delta_std_pp': selected['delta_pp'].std(ddof=1),
        })
    pd.DataFrame(summaries).to_csv(output / 'seed_summary.csv', index=False, encoding='utf-8-sig')
    print('\n待完成：', ', '.join(pending) if pending else '四次新训练均已完成')
    print('seed42 是已参与选配方的历史结果；新种子均值以 7、21 两对完整跑完为准。')
    print('置信区间不消除反复使用 Dev 的影响；两对新种子也不足以保证稳定优势。')
    print('分析目录：', output)
    return table, interval_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=cfg.BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    main(args.repeats)
