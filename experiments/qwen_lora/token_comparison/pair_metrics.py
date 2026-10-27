"""把新分类头的四列分数交给已有 pair-aware 和官方评分函数。"""
from pathlib import Path

import numpy as np

from baseline import score_predictions
from evaluate_lora_pair import pair_aware_decode
from pair_data import LABELS


def make_prediction_frame(frame, logits):
    logits = np.asarray(logits)
    if logits.shape != (len(frame), 4) or not np.isfinite(logits).all():
        raise ValueError('预测分数必须是有限数值，形状应为 [样本数, 4]。')
    result = frame[['instance_id', 'pair_id', 'label']].copy().reset_index(drop=True)
    result = result.rename(columns={'label': 'gold_label'})
    result['prediction'] = [LABELS[i] for i in logits.argmax(axis=1)]
    for index, label in enumerate(LABELS):
        result[f'logit_{label}'] = logits[:, index]
    return pair_aware_decode(result)


def score_prediction_frame(frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    reports = {}
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        records = frame[['instance_id', column]].rename(columns={column: 'prediction'}).to_dict('records')
        directory = output_dir / method
        directory.mkdir(parents=True, exist_ok=True)
        reports[method] = score_predictions(records, str(directory))

    gold = frame['gold_label'].map({
        'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
        'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'
    })
    correct = frame['pair_prediction'] == gold
    n_correct = int((correct & (gold == 'N')).sum())
    fbe_correct = int((correct & (gold != 'N')).sum())
    pair_report = reports['pair_aware']
    metrics = {
        'pair_f1': pair_report.weighted_f1,
        'independent_f1': reports['independent'].weighted_f1,
        'N_f1': pair_report.classification.per_label['NEGATIVE_OTHER'].f1,
        'N_correct': n_correct,
        'FBE_correct': fbe_correct,
        'soft_cons': pair_report.soft_cons,
        'hard_cons': pair_report.hard_cons,
    }
    print(f"pair-aware F1={metrics['pair_f1']:.6f}；N 正确={n_correct}；F/B/E 正确={fbe_correct}")
    return metrics
