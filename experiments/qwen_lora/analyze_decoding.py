from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 让脚本能导入项目根目录里的官方评分包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_functions import score_files


# 短标签 → 官方完整标签
FULL_LABELS = {
    'F': 'FORWARD_ENTAILMENT',
    'B': 'BACKWARD_ENTAILMENT',
    'E': 'EQUIVALENCE',
    'N': 'NEGATIVE_OTHER',
}

SCORE_COLUMNS = ['logit_F', 'logit_B', 'logit_E', 'logit_N']
def pair_aware_decode(df):
    df = df.copy()

    # 保留原预测，调整后的答案放进新列
    df['pair_prediction'] = df['prediction']

    # 三种合法组合，顺序与下面三个分数一致
    combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]

    for pair_id, group in df.groupby('pair_id'):
        # 单例保持原预测
        if len(group) != 2:
            continue

        row0 = group.iloc[0]
        row1 = group.iloc[1]

        # 分别计算 FB、BF、EE 的联合分数
        scores = [
            row0['logit_F'] + row1['logit_B'],
            row0['logit_B'] + row1['logit_F'],
            row0['logit_E'] + row1['logit_E'],
        ]

        # 找出最高分对应的组合；并列时取排在前面的
        best_index = scores.index(max(scores))
        label0, label1 = combinations[best_index]

        # 写回这两行的新预测
        df.loc[group.index[0], 'pair_prediction'] = label0
        df.loc[group.index[1], 'pair_prediction'] = label1

    return df

def score_one_method(df, prediction_column, output_dir):
    """把指定列转换成官方格式，调用官方评分脚本。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not df[prediction_column].isin(FULL_LABELS).all():
        raise ValueError(f'{prediction_column} 中存在非法标签')

    predictions = pd.DataFrame({
        'instance_id': df['instance_id'],
        'label': df[prediction_column].map(FULL_LABELS),
    })

    prediction_path = output_dir / 'official_predictions.csv'
    predictions.to_csv(prediction_path, index=False)

    reference_path = (
        PROJECT_ROOT
        / 'data/dev/dico_nli_dev_track1_reference.csv'
    )

    return score_files(
        reference_path,
        prediction_path,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    # 1. 读取已经保存的 Qwen B 分数
    input_path = (
        SCRIPT_DIR
        / 'outputs/readout_comparison/qwen_B_seed42'
        / 'predictions_with_scores.csv'
    )

    output_dir = input_path.parent / 'decoding_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, keep_default_na=False)
    df = df.reset_index(drop=True)

    # 2. 执行配对规则，原 prediction 列保持不变
    df = pair_aware_decode(df)

    short_labels = {v: k for k, v in FULL_LABELS.items()}
    df['gold_short'] = df['gold_label'].map(short_labels)

    if df['gold_short'].isna().any():
        raise ValueError('gold_label 中存在无法识别的标签')

    df.to_csv(
        output_dir / 'all_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # 找出属于双行组的样本，用于单独评估 paired 部分
    pair_sizes = df.groupby('pair_id')['pair_id'].transform('size')
    paired_mask = pair_sizes == 2

    print('总样本数：', len(df))
    print('paired 行数：', int(paired_mask.sum()))

    # 3. 分别计算独立预测、配对预测的成绩
    reports = {}
    summary_rows = []

    methods = [
        ('independent', 'prediction'),
        ('pair_aware', 'pair_prediction'),
    ]

    for method_name, prediction_column in methods:
        report = score_one_method(
            df,
            prediction_column,
            output_dir / method_name,
        )
        reports[method_name] = report

        n_metrics = report.classification.per_label[FULL_LABELS['N']]

        row = {
            'method': method_name,
            'weighted_f1': report.weighted_f1,
            'macro_f1': report.classification.macro_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons,
            'hard_cons': report.hard_cons,
            'N_precision': n_metrics.precision,
            'N_recall': n_metrics.recall,
        }

        # 保存四个类别各自的 F1
        for short_label, full_label in FULL_LABELS.items():
            row[f'{short_label}_f1'] = (
                report.classification.per_label[full_label].f1
            )

        # 只评估 554 条 paired 行
        paired_gold = df.loc[paired_mask, 'gold_short']
        paired_predictions = df.loc[paired_mask, prediction_column]

        row['paired_accuracy'] = accuracy_score(
            paired_gold,
            paired_predictions,
        )

        # 对 F/B/E 三类取宏平均；预测成 N 仍会计入对应类别的漏判
        row['paired_macro_f1_FBE'] = f1_score(
            paired_gold,
            paired_predictions,
            labels=['F', 'B', 'E'],
            average='macro',
            zero_division=0,
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / 'metrics_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )

    print('\n===== 成绩对比 =====')
    print(summary.round(6).to_string(index=False))

    # 4. 统计规则修好了哪些、又改错了哪些
    old_correct = df['prediction'] == df['gold_short']
    new_correct = df['pair_prediction'] == df['gold_short']
    changed = df['prediction'] != df['pair_prediction']

    fixed = ~old_correct & new_correct
    broken = old_correct & ~new_correct
    changed_wrong = changed & ~old_correct & ~new_correct

    raw_consistency = reports['independent'].consistency

    change_counts = pd.DataFrame([{
        'changed': int(changed.sum()),
        'fixed': int(fixed.sum()),
        'broken': int(broken.sum()),
        'changed_wrong': int(changed_wrong.sum()),
        'incompatible_pairs_before': (
            raw_consistency.reversible_pairs
            - raw_consistency.soft_consistent_pairs
        ),
    }])

    change_counts.to_csv(
        output_dir / 'change_counts.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # 保存对应样本，方便后续查看
    for filename, mask in [
        ('fixed.csv', fixed),
        ('broken.csv', broken),
        ('changed_wrong.csv', changed_wrong),
    ]:
        df.loc[mask].to_csv(
            output_dir / filename,
            index=False,
            encoding='utf-8-sig',
        )

    print('\n===== 预测改动统计 =====')
    print(change_counts.to_string(index=False))

    # 5. 描述原始四候选分数的尖锐度
    # 这里只在 F/B/E/N 四个候选内归一化，不代表校准后的正确概率
    scores = df[SCORE_COLUMNS].to_numpy(dtype=np.float64)

    # 每行减去最大值，避免 exp 计算溢出
    shifted_scores = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    max_probabilities = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    confidence_summary = pd.DataFrame([{
        'mean_max_probability': float(max_probabilities.mean()),
        'median_max_probability': float(np.median(max_probabilities)),
        'mean_probability_margin': float(margins.mean()),
        'fraction_max_probability_above_0p9': float(
            (max_probabilities > 0.9).mean()
        ),
    }])

    confidence_summary.to_csv(
        output_dir / 'confidence_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )

    print('\n===== 四候选分数统计 =====')
    print(confidence_summary.round(6).to_string(index=False))

    print('\n结果目录：', output_dir)