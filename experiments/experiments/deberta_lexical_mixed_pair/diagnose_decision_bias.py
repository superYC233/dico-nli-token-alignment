"""只读已有 CSV，诊断分数偏置与分类错误；不加载模型、不训练。

Dev 最优门槛只用于诊断，不作为新模型成绩或正式部署参数。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / 'outputs'
OUTPUT_DIR = RESULTS_ROOT / 'decision_diagnosis'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
MODELS = {
    'legacy_09306': PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe/outputs/run01_rr/lexical_seed_42/epoch_3/after_pair/predictions_with_scores.csv',
    'mixed_pair': RESULTS_ROOT / 'run01_mixed_pair/epoch_3/predictions_with_scores.csv',
    'mixed_row_only': RESULTS_ROOT / 'run02_mixed_row_only/epoch_10/predictions_with_scores.csv',
    'mixed_low_lex_lr': RESULTS_ROOT / 'run03_mixed_lexlr_1e_4/epoch_3/predictions_with_scores.csv',
}


def weighted_f1(gold, prediction):
    counts = np.bincount(4 * gold + prediction, minlength=16).reshape(4, 4)
    support = counts.sum(axis=1)
    denominator = support + counts.sum(axis=0)
    per_class = np.divide(
        2.0 * np.diag(counts), denominator,
        out=np.zeros(4), where=denominator != 0,
    )
    return float((per_class * support).sum() / len(gold))


def n_bias_curve(logits, gold, paired, paired_mask):
    best_fbe = logits[:, :3].argmax(axis=1)
    max_fbe = logits[:, :3].max(axis=1)
    # N 与最强 F/B/E 分数交叉的位置。每个相邻交叉点之间只取一次即可。
    crossing = np.unique(max_fbe - logits[:, 3])
    biases = np.unique(np.r_[
        crossing[0] - 1, (crossing[:-1] + crossing[1:]) / 2, crossing[-1] + 1, 0.0,
    ])
    records = []
    for bias in biases:
        independent = np.where(logits[:, 3] + bias > max_fbe, 3, best_fbe)
        # 修改 N 不影响 FB/BF/EE 的组合分数，所以配对行的结构答案保持不变。
        structured = np.where(paired_mask, paired, independent)
        records.append({
            'n_bias': float(bias),
            'independent_f1': weighted_f1(gold, independent),
            'pair_aware_f1': weighted_f1(gold, structured),
            'N_correct': int(((gold == 3) & (independent == 3)).sum()),
            'nonN_to_N': int(((gold != 3) & (independent == 3)).sum()),
        })
    return pd.DataFrame(records)


def two_bias_oracle(logits, gold, paired, paired_mask):
    # Dev 诊断搜索：F 加 d、B 减 d，再扫描 N 偏置。
    # F/B 的全局反向偏置同样在 FB、BF 组合中抵消，EE 不变。
    best = None
    for direction_bias in np.linspace(-6.0, 6.0, 121):
        adjusted = logits.copy()
        adjusted[:, 0] += direction_bias
        adjusted[:, 1] -= direction_bias
        curve = n_bias_curve(adjusted, gold, paired, paired_mask)
        row = curve.loc[curve.independent_f1.idxmax()].to_dict()
        if best is None or row['independent_f1'] > best['independent_f1']:
            best = {'direction_bias': float(direction_bias), **row}
    return best


def save_errors(frame, gold, name):
    table = frame.copy()
    table['gold_short'] = [LABELS[value] for value in gold]
    summary = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        wrong = table[table[column] != table['gold_short']].copy()
        wrong['error_type'] = wrong['gold_short'] + '_to_' + wrong[column]
        wrong.to_csv(OUTPUT_DIR / f'{name}_{method}_errors.csv', index=False, encoding='utf-8-sig')
        for error_type, count in wrong['error_type'].value_counts().items():
            summary.append({'model': name, 'method': method, 'error_type': error_type, 'rows': int(count)})
    return summary


def audit_data():
    train = pd.read_csv(PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv', keep_default_na=False)
    dev = pd.read_csv(PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv', keep_default_na=False)
    label_counts = train.groupby(['text1', 'text2'])['label'].nunique()
    conflicts = label_counts[label_counts > 1].index
    train_pairs = pd.MultiIndex.from_frame(train[['text1', 'text2']])
    conflicting_rows = train[train_pairs.isin(conflicts)].sort_values(['text1', 'text2', 'label'])
    conflicting_rows.to_csv(OUTPUT_DIR / 'train_exact_label_conflicts.csv', index=False, encoding='utf-8-sig')
    overlap = set(zip(train.text1, train.text2)) & set(zip(dev.text1, dev.text2))
    result = {
        'train_rows': len(train), 'dev_rows': len(dev),
        'train_ordered_text_pairs_with_conflicting_labels': len(conflicts),
        'train_rows_in_exact_conflicts': len(conflicting_rows),
        'train_dev_shared_exact_ordered_text_pairs': len(overlap),
    }
    (OUTPUT_DIR / 'data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    error_summaries = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (name, path) in zip(axes.flat, MODELS.items()):
        frame = pd.read_csv(path, keep_default_na=False)
        logits = frame[['logit_' + label for label in LABELS]].to_numpy()
        gold = frame.gold_label.map(dict(zip(FULL_LABELS, range(4)))).to_numpy(dtype=int)
        independent = frame.prediction.map(dict(zip(LABELS, range(4)))).to_numpy(dtype=int)
        paired = frame.pair_prediction.map(dict(zip(LABELS, range(4)))).to_numpy(dtype=int)
        paired_mask = frame.groupby('pair_id').pair_id.transform('size').eq(2).to_numpy()
        assert np.array_equal(logits.argmax(axis=1), independent), name
        assert np.isclose(weighted_f1(gold, independent), f1_score(gold, independent, average='weighted'))

        curve = n_bias_curve(logits, gold, paired, paired_mask)
        baseline = curve[curve.n_bias.eq(0.0)].iloc[0]
        assert np.isclose(baseline.independent_f1, weighted_f1(gold, independent))
        assert np.isclose(baseline.pair_aware_f1, weighted_f1(gold, paired))
        curve.to_csv(OUTPUT_DIR / f'{name}_dev_diagnostic_n_curve.csv', index=False, encoding='utf-8-sig')

        # N 对其余三类的对数优势；不依赖是否成对。
        n_score = logits[:, 3] - np.logaddexp.reduce(logits[:, :3], axis=1)
        n_gold = gold == 3
        fbe_mask = ~n_gold
        fbe_only = logits[fbe_mask, :3].argmax(axis=1)
        raw_best = curve.loc[curve.independent_f1.idxmax()]
        target_98 = curve[curve.N_correct >= 98]
        target_98 = target_98.loc[target_98.nonN_to_N.idxmin()]
        joint_oracle = two_bias_oracle(logits, gold, paired, paired_mask)
        override = np.where(paired_mask, paired, 3)
        summaries.append({
            'model': name, 'source_predictions': str(path),
            'baseline_independent_f1': baseline.independent_f1,
            'baseline_pair_aware_f1': baseline.pair_aware_f1,
            'baseline_N_correct': int(baseline.N_correct),
            'baseline_nonN_to_N': int(baseline.nonN_to_N),
            'N_AUROC': roc_auc_score(n_gold, n_score),
            'N_average_precision': average_precision_score(n_gold, n_score),
            'FBE_correct_when_N_column_removed_diagnostic': int((fbe_only == gold[fbe_mask]).sum()),
            'dev_oracle_N_bias_best_independent_f1': raw_best.independent_f1,
            'dev_oracle_two_bias_best_independent_f1': joint_oracle['independent_f1'],
            'dev_oracle_two_bias_direction': joint_oracle['direction_bias'],
            'dev_oracle_two_bias_N': joint_oracle['n_bias'],
            'dev_oracle_at_least_98_N_min_nonN_to_N': int(target_98.nonN_to_N),
            'structure_diagnostic_all_singletons_N_f1': weighted_f1(gold, override),
        })
        error_summaries.extend(save_errors(frame, gold, name))

        recall = curve.N_correct / int(n_gold.sum())
        axis.plot(recall, curve.independent_f1, label='Independent', color='tab:blue')
        axis.plot(recall, curve.pair_aware_f1, label='Pair-aware', color='tab:orange')
        baseline_recall = baseline.N_correct / int(n_gold.sum())
        axis.scatter([baseline_recall], [baseline.independent_f1], color='tab:blue', marker='o', zorder=3)
        axis.scatter([baseline_recall], [baseline.pair_aware_f1], color='tab:orange', marker='s', zorder=3)
        axis.set_title(name)
        axis.set_xlabel('N recall as the N bias changes')
        axis.set_ylabel('Weighted F1')
        axis.set_xlim(0, 1.01)
        axis.set_ylim(0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(loc='lower left')

    figure.suptitle('Dev-only diagnostic: raising N recall can improve structure scores while hurting independent predictions')
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(OUTPUT_DIR / 'n_bias_tradeoff.png', dpi=160)
    plt.close(figure)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / 'decision_diagnosis_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(error_summaries).to_csv(OUTPUT_DIR / 'baseline_error_counts.csv', index=False, encoding='utf-8-sig')
    audit = audit_data()
    notes = '''# 判定偏置诊断说明

本目录只读已有预测，不加载模型、不训练、不修改正式预测。
基准成绩逐条验证为原 logits 的 argmax，并核对官方 Weighted F1。

`dev_oracle` 字段使用当前 Dev 金标寻找诊断最优点，不是独立评估成绩，
不能把这些点作为新模型结果或据此确定正式门槛。两偏置扫描范围为
F 加 d、B 减 d，d 在 [-6, 6]、步长 0.1；N 偏置扫描所有分数交叉点之间的区间。
扫描只检查常数偏置这一类改动，不能证明所有校准方法都无效。

`structure_diagnostic_all_singletons_N_f1` 保持配对预测不动，把单例全设为 N，
用于揭示当前数据结构的影响，不代表学到更多语义，也不保证隐藏测试集同样组织。

AUROC/AP 评价当前固定模型在 Dev 上对 N 的排序，不能单独证明跨数据集语义理解。
移除 N 列后评价 F/B/E 只用于拆解错误，不是四分类正式结果。
数据冲突检查使用完全相同的有序文本字符串；不是对所有语境歧义的穷尽审查。

`*_errors.csv` 全部来自原始未调整预测。图中的圆点/方点为原模型的独立/结构分数。
'''
    (OUTPUT_DIR / 'README.md').write_text(notes, encoding='utf-8')
    print(summary.drop(columns=['source_predictions']).to_string(index=False))
    print('\n数据审计：', audit)
    print('\n诊断输出：', OUTPUT_DIR)
    print('dev_oracle 是使用 Dev 金标的诊断结果，不是新的正式模型成绩。')


if __name__ == '__main__':
    main()
