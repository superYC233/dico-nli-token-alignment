import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from evaluation_functions import score_files
from experiments.deberta_wordnet import config as cfg


def get_pair_logits(first, second):
    # 与原版相同的三组合 Pair CE，不是额外的合法概率约束。
    return torch.stack([
        first[:, 0] + second[:, 1],
        first[:, 1] + second[:, 0],
        first[:, 2] + second[:, 2],
    ], dim=1)


def pair_aware(logits, frame):
    if logits.shape != (len(frame), 4) or not torch.isfinite(logits).all():
        raise ValueError('预测分数的形状或数值不正确。')
    frame = frame.reset_index(drop=True)
    predictions = logits.argmax(dim=1).clone()
    combinations = [(0, 1), (1, 0), (2, 2)]
    for pair_id, group in frame.groupby('pair_id'):
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise ValueError(f'{pair_id}：不是两行分组。')
        index0, index1 = group.index.tolist()
        scores = get_pair_logits(logits[index0:index0 + 1], logits[index1:index1 + 1])
        choice = scores.argmax(dim=1).item()
        predictions[index0], predictions[index1] = combinations[choice]
    return predictions


def predict_logits(model, dev_loader, device):
    was_training = model.training
    model.eval()
    all_logits = []
    try:
        with torch.inference_mode():
            for input_ids, attention_mask, features, _ in tqdm(dev_loader, desc='Dev 预测'):
                logits = model(
                    input_ids.to(device), attention_mask.to(device), features.to(device),
                )
                all_logits.append(logits.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(all_logits, dim=0)


def score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    independent = logits.argmax(dim=1)
    paired = pair_aware(logits, dev)
    frame = dev[['instance_id', 'pair_id', 'text1', 'text2', 'label']].copy().reset_index(drop=True)
    frame = frame.rename(columns={'label': 'gold_label'})
    frame['prediction'] = [cfg.LABELS[i] for i in independent.tolist()]
    frame['pair_prediction'] = [cfg.LABELS[i] for i in paired.tolist()]
    for index, label in enumerate(cfg.LABELS):
        frame[f'logit_{label}'] = logits[:, index].tolist()
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')

    summary = []
    full_to_short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    gold_short = frame['gold_label'].map(full_to_short)
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
        row = {
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_precision': n_metrics.precision, 'N_recall': n_metrics.recall,
            'N_f1': n_metrics.f1, 'N_correct': n_metrics.true_positive,
        }
        for short, full in zip(cfg.LABELS[:3], cfg.FULL_LABELS[:3]):
            metrics = report.classification.per_label[full]
            row[f'{short}_correct'] = metrics.true_positive
            row[f'{short}_f1'] = metrics.f1
        row['FBE_correct'] = row['F_correct'] + row['B_correct'] + row['E_correct']
        summary.append(row)

        predicted = pd.Series([cfg.LABELS[i] for i in predictions.tolist()], name='prediction')
        confusion = pd.crosstab(gold_short.rename('gold'), predicted)
        confusion = confusion.reindex(index=cfg.LABELS, columns=cfg.LABELS, fill_value=0)
        confusion.to_csv(score_dir / 'confusion_matrix.csv', encoding='utf-8-sig')

    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary


def compare_with_original(prediction_path, output_dir, baseline_path=None,
                          baseline_column='label', baseline_name='原 DeBERTa'):
    # 只在完成预测后做分析；历史预测不进入特征、训练或解码。
    if baseline_path is None:
        baseline_path = cfg.PROJECT_ROOT / 'artifacts/final/dev_predictions_pair_aware.csv'
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f'未找到{baseline_name}的历史预测，跳过逐条比较。')
        return
    old = pd.read_csv(baseline_path, keep_default_na=False)
    old = old[['instance_id', baseline_column]].rename(columns={baseline_column: 'old_prediction'})
    current = pd.read_csv(prediction_path, keep_default_na=False)
    if set(old['instance_id']) != set(current['instance_id']):
        raise ValueError('历史预测和当前 Dev 的 instance_id 不一致。')
    full_to_short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    old['old_prediction'] = old['old_prediction'].map(lambda label: full_to_short.get(label, label))
    if not old['old_prediction'].isin(cfg.LABELS).all():
        raise ValueError('历史预测包含未知标签。')
    merged = current.merge(old, on='instance_id', validate='one_to_one')
    gold = merged['gold_label'].map(full_to_short)
    old_correct = merged['old_prediction'] == gold
    new_correct = merged['pair_prediction'] == gold
    fixed = merged[~old_correct & new_correct]
    broken = merged[old_correct & ~new_correct]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
    fixed.to_csv(output_dir / 'fixed.csv', index=False, encoding='utf-8-sig')
    broken.to_csv(output_dir / 'broken.csv', index=False, encoding='utf-8-sig')
    print(f'相对{baseline_name}：修好 {len(fixed)} 条，新增错误 {len(broken)} 条。')


def compare_with_lexical(prediction_path, output_dir):
    root = cfg.PROJECT_ROOT
    best_path = root / 'model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_metrics.json'
    if not best_path.exists():
        print('未找到 0.930569 词汇分支版的最佳轮记录，跳过该对照。')
        return
    epoch = json.loads(best_path.read_text(encoding='utf-8'))['best_epoch']
    baseline_path = (
        root / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42'
        / f'epoch_{epoch}' / 'predictions_with_scores.csv'
    )
    compare_with_original(
        prediction_path, output_dir, baseline_path=baseline_path,
        baseline_column='pair_prediction', baseline_name='0.930569 词汇分支版',
    )
