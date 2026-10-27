from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from evaluation_functions import score_files
from experiments.deberta_abcde import config as cfg


def get_pair_logits(first, second):
    # 顺序固定为 FB、BF、EE；用于原版三组合 Pair CE。
    # 这不是包含 N 的“合法组合概率约束”。
    return torch.stack([
        first[:, 0] + second[:, 1],
        first[:, 1] + second[:, 0],
        first[:, 2] + second[:, 2],
    ], dim=1)


def pair_aware(logits, frame):
    if logits.shape != (len(frame), 4) or not torch.isfinite(logits).all():
        raise ValueError('Dev logits 的形状或数值不正确。')
    frame = frame.reset_index(drop=True)
    predictions = logits.argmax(dim=1).clone()
    combinations = [(0, 1), (1, 0), (2, 2)]
    for pair_id, group in frame.groupby('pair_id'):
        if len(group) == 1:
            continue  # 单例保留模型预测，不按组大小强制判 N。
        if len(group) != 2:
            raise ValueError(f'{pair_id} 出现超过两行。')
        index0, index1 = group.index.tolist()
        scores = get_pair_logits(logits[index0:index0 + 1], logits[index1:index1 + 1])
        choice = scores.argmax(dim=1).item()
        predictions[index0], predictions[index1] = combinations[choice]
    return predictions


def predict_logits(model, dev_loader, device):
    was_training = model.training
    model.eval()
    outputs = []
    try:
        with torch.inference_mode():
            for input_ids, attention_mask, _ in tqdm(dev_loader, desc='Dev 预测'):
                logits = model(input_ids.to(device), attention_mask.to(device))
                outputs.append(logits.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(outputs, dim=0)


def score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    independent = logits.argmax(dim=1)
    paired = pair_aware(logits, dev)

    frame = dev[['instance_id', 'pair_id', 'label']].copy().reset_index(drop=True)
    frame = frame.rename(columns={'label': 'gold_label'})
    frame['prediction'] = [cfg.LABELS[i] for i in independent.tolist()]
    frame['pair_prediction'] = [cfg.LABELS[i] for i in paired.tolist()]
    for index, label in enumerate(cfg.LABELS):
        frame[f'logit_{label}'] = logits[:, index].tolist()
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')

    rows = []
    for method, predictions in [('independent', independent), ('pair_aware', paired)]:
        score_dir = output_dir / method
        score_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = score_dir / 'official_predictions.csv'
        pd.DataFrame({
            'instance_id': frame['instance_id'],
            'label': [cfg.FULL_LABELS[i] for i in predictions.tolist()],
        }).to_csv(prediction_path, index=False)
        report = score_files(gold_path, prediction_path, output_dir=score_dir)
        n_metrics = report.classification.per_label['NEGATIVE_OTHER']
        rows.append({
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_precision': n_metrics.precision, 'N_recall': n_metrics.recall,
            'N_f1': n_metrics.f1, 'N_correct': n_metrics.true_positive,
            'FBE_correct': sum(
                report.classification.per_label[label].true_positive for label in cfg.FULL_LABELS[:3]
            ),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary
