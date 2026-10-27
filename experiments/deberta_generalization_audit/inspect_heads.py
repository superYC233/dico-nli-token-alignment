"""只读检查现有模型：CLS 与 Token Alignment 分别给了什么答案。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from experiments.deberta_row_ce_frozen import config as cfg
from experiments.deberta_row_ce_frozen.train_deberta_row_ce import build_model
from experiments.deberta_lexical.evaluation import pair_aware
from experiments.deberta_generalization_audit.export_predictions import OUT, BASELINE_MODEL_DIR, predict, sha256


def main():
    manifest = json.loads((OUT / 'inference_manifest.json').read_text(encoding='utf-8'))
    checkpoint = BASELINE_MODEL_DIR / 'best_model.pth'
    assert sha256(checkpoint) == manifest['checkpoint_sha256'], '权重已变化，请另开审计。'
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_DIR / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    values = {'cls': [], 'alignment': []}

    def collect(name):
        def hook(module, inputs, output):
            values[name].append(output.detach().float().cpu())
        return hook

    # 抓取同一次前向的两条分支；不改变前向计算，不重新训练。
    handles = [model.output.register_forward_hook(collect('cls')),
               model.match_output.register_forward_hook(collect('alignment'))]
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    try:
        full = predict(dev, tokenizer, model, device)
    finally:
        for handle in handles:
            handle.remove()
    old = pd.read_csv(OUT / 'dev_predictions.csv', keep_default_na=False)
    assert (full.instance_id == old.instance_id).all()
    assert (full.prediction == old.prediction).all()
    assert (full.pair_prediction == old.pair_prediction).all()
    labels = np.array(['F', 'B', 'E', 'N'])
    gold = full.gold.to_numpy()
    cls = torch.cat(values['cls'])
    alignment = torch.cat(values['alignment'])
    np.testing.assert_allclose(
        (cls + alignment).numpy(),
        full[[f'logit_{c}' for c in labels]].to_numpy(), atol=1e-6,
    )
    records = full[['instance_id', 'pair_id', 'text1', 'text2', 'gold',
                    'prediction', 'pair_prediction']].copy()
    summary = {}
    for name, logits in [('cls', cls), ('alignment', alignment), ('full', cls + alignment)]:
        predictions = labels[logits.argmax(1).numpy()]
        paired = labels[pair_aware(logits, dev).numpy()]
        records[name + '_prediction'] = predictions
        for index, label in enumerate(labels):
            records[name + '_logit_' + label] = logits[:, index].numpy()
        # 正确类相对最高错误类的分数差；大于 0 表示该分支答对。
        z = logits.numpy()
        gold_indices = np.array([list(labels).index(g) for g in gold])
        other = z.copy()
        other[np.arange(len(z)), gold_indices] = -np.inf
        records[name + '_gold_margin'] = z[np.arange(len(z)), gold_indices] - other.max(1)
        summary[name] = {
            'independent_f1': f1_score(gold, predictions, average='weighted'),
            'pair_f1': f1_score(gold, paired, average='weighted'),
            'correct': int(sum(gold == predictions)),
            'N_correct': int(sum((gold == 'N') & (predictions == 'N'))),
            'FBE_correct': int(sum((gold != 'N') & (predictions == gold))),
            'predicted_N': int(sum(predictions == 'N')),
        }
    cp = records.cls_prediction.to_numpy()
    ap = records.alignment_prediction.to_numpy()
    fp = records.prediction.to_numpy()
    summary['cls_vs_full'] = {
        'fixed': int(sum((cp != gold) & (fp == gold))),
        'broken': int(sum((cp == gold) & (fp != gold))),
    }
    summary['wrong_full_breakdown'] = {
        'cls_correct_alignment_wrong': int(sum((fp != gold) & (cp == gold) & (ap != gold))),
        'alignment_correct_cls_wrong': int(sum((fp != gold) & (ap == gold) & (cp != gold))),
        'both_wrong': int(sum((fp != gold) & (ap != gold) & (cp != gold))),
    }
    summary['note'] = ('Inference ablation of jointly trained heads; '
                       'not separately retrained CLS-only/alignment-only controls.')
    records.to_csv(OUT / 'head_decomposition.csv', index=False, encoding='utf-8-sig')
    (OUT / 'head_decomposition_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
