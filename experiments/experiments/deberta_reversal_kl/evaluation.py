from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from evaluation_functions import score_files
from experiments.deberta_reversal_kl import config as cfg


def decode_twins(logits, frame, allow_n):
    predictions = logits.argmax(dim=1).clone()
    combinations = [(0, 1), (1, 0), (2, 2)]
    if allow_n:
        combinations.append((3, 3))
    for _, group in frame.reset_index(drop=True).groupby('pair_id', sort=False):
        if len(group) == 1:
            continue
        first, second = group.index.tolist()
        scores = torch.stack([logits[first, a] + logits[second, b] for a, b in combinations])
        a, b = combinations[scores.argmax().item()]
        predictions[first], predictions[second] = a, b
    return predictions


def predict_logits(model, loader, device):
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc='Dev预测'):
                output = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                )
                values.append(output.logits.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(values)


def score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        'independent': logits.argmax(dim=1),
        # 作者风格：保留N竞争；配对行可能选NN，因此SoftCons不保证为1。
        'twin': decode_twins(logits, dev, allow_n=True),
        # 仅作旧结果对照：配对行只选FB/BF/EE；单例仍由模型预测。
        'pair_aware': decode_twins(logits, dev, allow_n=False),
    }
    frame = dev[['instance_id', 'pair_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
    frame = frame.rename(columns={'label': 'gold_label'})
    for method, column in [('independent', 'prediction'), ('twin', 'twin_prediction'), ('pair_aware', 'pair_prediction')]:
        frame[column] = [cfg.LABELS[i] for i in predictions[method].tolist()]
    for index, label in enumerate(cfg.LABELS):
        frame[f'logit_{label}'] = logits[:, index].tolist()
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')

    gold = frame['gold_label'].map(dict(zip(cfg.FULL_LABELS, cfg.LABELS)))
    records = []
    for method, predicted in predictions.items():
        score_dir = output_dir / method
        score_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = score_dir / 'official_predictions.csv'
        pd.DataFrame({
            'instance_id': frame['instance_id'],
            'label': [cfg.FULL_LABELS[i] for i in predicted.tolist()],
        }).to_csv(prediction_path, index=False)
        report = score_files(gold_path, prediction_path, output_dir=score_dir)
        n = report.classification.per_label['NEGATIVE_OTHER']
        short_prediction = pd.Series([cfg.LABELS[i] for i in predicted.tolist()])
        row = {
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_precision': n.precision, 'N_recall': n.recall, 'N_f1': n.f1,
            'N_correct': n.true_positive,
            'non_N_to_N': int((gold.ne('N') & short_prediction.eq('N')).sum()),
            'N_to_non_N': int((gold.eq('N') & short_prediction.ne('N')).sum()),
        }
        for short, full in zip(cfg.LABELS[:3], cfg.FULL_LABELS[:3]):
            metrics = report.classification.per_label[full]
            row[f'{short}_correct'] = metrics.true_positive
            row[f'{short}_f1'] = metrics.f1
        row['FBE_correct'] = sum(row[f'{label}_correct'] for label in cfg.LABELS[:3])
        records.append(row)
        confusion = pd.crosstab(gold.rename('gold'), short_prediction.rename('prediction'))
        confusion.reindex(index=cfg.LABELS, columns=cfg.LABELS, fill_value=0).to_csv(
            score_dir / 'confusion_matrix.csv', encoding='utf-8-sig',
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary
