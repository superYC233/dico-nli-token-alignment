import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from experiments.deberta_local_semantics import config as cfg
from experiments.deberta_manual_aug.train_boundary_manual import evaluate as original_evaluate


def evaluate(model, loader, dev, device, output_dir):
    summary = original_evaluate(model, loader, dev, device, output_dir)
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    gold_n = frame.gold_label.eq('NEGATIVE_OTHER')
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        predicted_n = frame[column].eq('N')
        selected = summary.method.eq(method)
        summary.loc[selected, 'N_predicted'] = int(predicted_n.sum())
        summary.loc[selected, 'non_N_to_N'] = int((~gold_n & predicted_n).sum())
        summary.loc[selected, 'N_to_non_N'] = int((gold_n & ~predicted_n).sum())
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary[['method', 'N_predicted', 'non_N_to_N', 'N_to_non_N']].to_string(index=False))
    return summary


def evaluate_local(model, loader, annotations, device, output_dir):
    was_training = model.training
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    predictions, losses = [], []
    with torch.random.fork_rng(devices=devices):
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                _, tokens = model(
                    batch['input_ids'].to(device), batch['attention_mask'].to(device), return_tokens=True,
                )
                logits = model.local_logits(
                    tokens, batch['span1_mask'].to(device), batch['span2_mask'].to(device),
                )
                losses.extend(F.cross_entropy(
                    logits, batch['local_target'].to(device), reduction='none',
                ).cpu().tolist())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
    model.train(was_training)
    detail = annotations.copy().reset_index(drop=True)
    detail['local_prediction'] = [cfg.LABELS[value] for value in predictions]
    detail['correct'] = detail.local_prediction.eq(detail.local_relation)
    detail['local_ce'] = losses
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    metrics = {
        'rows': len(detail), 'accuracy': float(detail.correct.mean()),
        'mean_ce': float(detail.local_ce.mean()),
        'scope': 'Local-label holdout only. All inputs remain in ordinary Train Row CE; not unseen-text generalization.',
    }
    (output_dir / 'summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"局部标注留出诊断：{detail.correct.sum()}/{len(detail)}，不用于选模。")
    return metrics


def compare_with_reference(prediction_path, output_dir):
    old = pd.read_csv(cfg.REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    joined = current.merge(
        old[['instance_id', 'prediction', 'pair_prediction']],
        on='instance_id', suffixes=('', '_reference'), validate='one_to_one',
    )
    gold = joined.gold_label.map(short)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before = joined[column + '_reference'].eq(gold)
        after = joined[column].eq(gold)
        fixed, broken = joined[~before & after], joined[before & ~after]
        fixed.to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        broken.to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
        counts.append({'method': method, 'fixed': len(fixed), 'broken': len(broken)})
        print(f'相对0.938145的{method}：修好 {len(fixed)}，破坏 {len(broken)}。')
    pd.DataFrame(counts).to_csv(output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
