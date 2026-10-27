"""恢复原始行顺序；沿用项目官方评分，分别报告独立和pair-aware。"""
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from experiments.deberta_bidirectional.data import LABELS, FULL_LABELS, model_inputs
from experiments.deberta_lexical.evaluation import score_logits


def predict_logits(model, loader, device):
    was_training = model.training
    values = torch.empty((len(loader.dataset.frame), 4), dtype=torch.float32)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        # 评估不改变后续训练的dropout随机数。
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            for batch in tqdm(loader, desc='双向联合预测'):
                logits = model(**model_inputs(batch, device))
                values[batch['source_rows']] = logits.float().cpu()
    finally:
        model.train(was_training)
    return values


def score_predictions(logits, frame, output_dir, gold_path):
    output_dir = Path(output_dir)
    summary = score_logits(logits, frame, output_dir, gold_path=gold_path)
    predictions = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    gold = predictions.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        index = summary.index[summary.method.eq(method)][0]
        predicted = predictions[column]
        summary.loc[index, 'nonN_to_N'] = int((gold.ne('N') & predicted.eq('N')).sum())
        for label in LABELS:
            summary.loc[index, f'{label}_predicted'] = int(predicted.eq(label).sum())
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')

    before = predictions.prediction.eq(gold)
    after = predictions.pair_prediction.eq(gold)
    changed = predictions.prediction.ne(predictions.pair_prediction)
    changes = {'changed': int(changed.sum()), 'fixed': int((~before & after).sum()),
               'broken': int((before & ~after).sum()),
               'changed_wrong': int((changed & ~before & ~after).sum())}
    (output_dir / 'decoding_changes.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    print('独立非N→N：', int(summary.loc[summary.method.eq('independent'), 'nonN_to_N'].iloc[0]))
    print('pair-aware改动：', changes)
    return summary


def evaluate(model, loader, frame, device, output_dir, gold_path):
    return score_predictions(predict_logits(model, loader, device), frame, output_dir, gold_path)
