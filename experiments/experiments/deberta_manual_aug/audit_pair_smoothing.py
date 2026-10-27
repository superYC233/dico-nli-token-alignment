"""用已保存的Row/Pair分数检查平滑梯度；不加载模型、不训练。"""
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'experiments/deberta_stage_average/outputs/run01_late_average/epoch_2'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs/run06_relation48_staged/branch_audit'
LABEL_IDS = {'FORWARD_ENTAILMENT': 0, 'BACKWARD_ENTAILMENT': 1,
             'EQUIVALENCE': 2, 'NEGATIVE_OTHER': 3}


def main():
    summaries = []
    for phase in ['after_row', 'after_pair']:
        frame = pd.read_csv(SOURCE_DIR / phase / 'train/predictions_with_scores.csv')
        logits = torch.tensor(frame[['logit_F', 'logit_B', 'logit_E', 'logit_N']].to_numpy(),
                              dtype=torch.float64)
        pairs, targets, pair_ids = [], [], []
        for pair_id, group in frame.groupby('pair_id'):
            if len(group) != 2:
                continue
            pairs.append(group.index.tolist())
            labels = tuple(group.gold_label.map(LABEL_IDS))
            targets.append({(0, 1): 0, (1, 0): 1, (2, 2): 2}[labels])
            pair_ids.append(pair_id)
        pairs = torch.tensor(pairs)
        targets = torch.tensor(targets)
        first, second = logits[pairs[:, 0]], logits[pairs[:, 1]]
        scores = torch.stack([first[:, 0] + second[:, 1], first[:, 1] + second[:, 0],
                              first[:, 2] + second[:, 2]], dim=1).requires_grad_()
        weights = torch.tensor([1.0, 1.0, 1.25], dtype=scores.dtype)
        plain_loss = F.cross_entropy(scores, targets, weight=weights)
        smooth_loss = F.cross_entropy(scores, targets, weight=weights, label_smoothing=0.05)
        # 梯度只针对保存的组合分数；没有神经网络参数或优化器。
        plain_grad = torch.autograd.grad(plain_loss, scores, retain_graph=True)[0]
        smooth_grad = torch.autograd.grad(smooth_loss, scores)[0]
        probability = scores.softmax(dim=1).detach()
        correct = scores.argmax(dim=1).eq(targets)
        target_grad = smooth_grad.gather(1, targets[:, None]).squeeze(1)
        cosine = (plain_grad * smooth_grad).sum(1) / (
            plain_grad.norm(dim=1) * smooth_grad.norm(dim=1)
        ).clamp_min(1e-30)
        summaries.append({
            'phase': phase, 'pairs': len(targets), 'pair_correct': int(correct.sum()),
            'pair_accuracy': float(correct.double().mean()),
            'weighted_CE_without_smoothing': float(plain_loss.detach()),
            'weighted_CE_with_smoothing': float(smooth_loss.detach()),
            'gold_score_down_pairs': int(target_grad.gt(0).sum()),
            'correct_gold_score_down_pairs': int((target_grad.gt(0) & correct).sum()),
            'opposing_gradient_pairs': int(cosine.lt(0).sum()),
            'median_gradient_cosine': float(cosine.median()),
            'median_max_probability': float(probability.max(1).values.median()),
            'scope': 'Saved eval logits; not actual dropout training gradients; no model optimization',
        })
        pd.DataFrame({'pair_id': pair_ids, 'correct': correct.tolist(),
                      'gold_score_gradient_smoothed': target_grad.tolist(),
                      'gradient_cosine': cosine.tolist()}).to_csv(
                          OUTPUT_DIR / f'{phase}_smoothing_gradients.csv',
                          index=False, encoding='utf-8-sig')
    result = pd.DataFrame(summaries)
    result.to_csv(OUTPUT_DIR / 'smoothing_summary.csv', index=False, encoding='utf-8-sig')
    print(result.to_string(index=False))


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
