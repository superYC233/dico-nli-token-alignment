"""只读核算：现有阶段分数、梯度日志和解析反例；不加载或训练模型。"""
import json
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs/run06_row_parameter_anchor'
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_IDS = dict(zip(FULL_LABELS, range(4)))
LABELS = 'FBEN'


def tensor_logits(frame):
    return torch.tensor(frame[[f'logit_{c}' for c in LABELS]].to_numpy(), dtype=torch.float64)


def margins(logits, targets):
    competitors = logits.clone()
    competitors[torch.arange(len(targets)), targets] = -torch.inf
    return logits[torch.arange(len(targets)), targets] - competitors.max(dim=1).values


def inspect_stages():
    records = []
    example = []
    for epoch in range(1, 11):
        before = pd.read_csv(RESULTS_DIR / f'epoch_{epoch}/after_row/predictions_with_scores.csv',
                             keep_default_na=False)
        after = pd.read_csv(RESULTS_DIR / f'epoch_{epoch}/after_pair/predictions_with_scores.csv',
                            keep_default_na=False)
        assert before.instance_id.tolist() == after.instance_id.tolist()
        targets = torch.tensor(before.gold_label.map(LABEL_IDS).to_numpy())
        first, second = tensor_logits(before), tensor_logits(after)
        ce_before = F.cross_entropy(first, targets, reduction='none')
        ce_after = F.cross_entropy(second, targets, reduction='none')
        right_before = first.argmax(1).eq(targets)
        right_after = second.argmax(1).eq(targets)
        reference_margin, current_margin = margins(first, targets), margins(second, targets)
        # 只给原来判对的行设置正margin下限；原来判错的行不锁住。
        floors = reference_margin.clamp(max=1.0)
        damage = torch.where(right_before, (floors - current_margin).clamp_min(0), 0)
        broken = right_before & ~right_after
        assert bool((damage[broken] > 0).all())
        for k, label in enumerate(LABELS):
            mask = targets.eq(k)
            records.append({
                'epoch': epoch, 'class': label, 'rows': int(mask.sum()),
                'row_ce': float(ce_before[mask].mean()),
                'pair_ce': float(ce_after[mask].mean()),
                'ce_change': float((ce_after[mask] - ce_before[mask]).mean()),
                'correct_before': int((right_before & mask).sum()),
                'correct_after': int((right_after & mask).sum()),
                'broken': int((broken & mask).sum()),
                'fixed': int((~right_before & right_after & mask).sum()),
                'margin_damage_mean': float(damage[mask].mean()),
                'broken_detected_by_margin': int((broken & mask & damage.gt(0)).sum()),
            })
        if epoch == 9:
            selected = before.pair_id.eq(before.iloc[0].pair_id)
            for phase, frame in [('after_row', before), ('after_pair', after)]:
                item = frame.loc[selected].copy()
                zs = tensor_logits(item)
                group_scores = torch.stack([
                    zs[0, 0] + zs[1, 1], zs[0, 1] + zs[1, 0], zs[0, 2] + zs[1, 2],
                ])
                example.append({
                    'phase': phase,
                    'rows': item[['text1', 'text2', 'prediction',
                                 'logit_F', 'logit_B', 'logit_E', 'logit_N']].to_dict('records'),
                    'group_scores': group_scores.tolist(),
                    'first_row_gold_probability': float(zs[0].softmax(0)[0]),
                    'first_row_F_minus_N': float(zs[0, 0] - zs[0, 3]),
                })
    table = pd.DataFrame(records)
    table.to_csv(RESULTS_DIR / 'applicability_stage_losses.csv', index=False, encoding='utf-8-sig')
    declining = table[table.correct_after < table.correct_before]
    missed = declining[declining.ce_change <= 0]
    return {
        'scope': 'Dev is used for retrospective diagnosis only, not training or selecting constraint budgets',
        'class_decline_events': len(declining),
        'class_ce_increase_detected': int((declining.ce_change > 0).sum()),
        'class_ce_increase_missed': missed[['epoch', 'class', 'ce_change',
                                          'correct_before', 'correct_after']].to_dict('records'),
        'broken_row_events': int(table.broken.sum()),
        'margin_detected_broken_row_events': int(table.broken_detected_by_margin.sum()),
        'example': example,
    }


def inspect_gradient_log():
    path = ROOT / 'experiments/deberta_manual_aug/outputs/run25_row_pair_ema_bloop/gradient_batches.csv'
    log = pd.read_csv(path)
    return {
        'source': str(path),
        'scope': 'earlier BLOOP run, 3290 rows and no Lexical branch; not the current anchor run',
        'batches': len(log),
        'row_pair_negative_count': int(log.row_pair_cosine.lt(0).sum()),
        'row_pair_negative_fraction': float(log.row_pair_cosine.lt(0).mean()),
        'row_pair_cosine_median': float(log.row_pair_cosine.median()),
        'row_dot_assigned_negative_count': int(log.row_dot_assigned_gradient.lt(0).sum()),
        'row_dot_assigned_negative_fraction': float(log.row_dot_assigned_gradient.lt(0).mean()),
        'scope_of_dot_product': 'current minibatch raw gradient dot assigned gradient, not actual AdamW update',
    }


def hand_calculations():
    # 两条真实标签相同的样本：平均CE下降不保证正确数不下降。
    before_p = torch.tensor([[.6, .001, .001, .398], [.6, .001, .001, .398]], dtype=torch.float64)
    after_p = torch.tensor([[.999, .0001, .0001, .0008], [.49, .001, .001, .508]], dtype=torch.float64)
    target = torch.zeros(2, dtype=torch.long)
    mean_ce_counterexample = {
        'before_correct': int(before_p.argmax(1).eq(target).sum()),
        'after_correct': int(after_p.argmax(1).eq(target).sum()),
        'before_mean_ce': float(-before_p[:, 0].log().mean()),
        'after_mean_ce': float(-after_p[:, 0].log().mean()),
    }

    # 使用现有Pair权重和平滑的解析最优概率；没有执行优化器更新。
    eps = .05
    weights = torch.tensor([1., 1., 1.25], dtype=torch.float64)
    q = torch.tensor([1 - eps + eps / 3, eps / 3, eps / 3], dtype=torch.float64)
    effective_q = q * weights
    effective_q = effective_q / effective_q.sum()
    gap_fb_bf = float((effective_q[0] / effective_q[1]).log())
    gap_fb_ee = float((effective_q[0] / effective_q[2]).log())
    a = (3 + gap_fb_bf / 2 + gap_fb_ee / 2) / 3
    b, e = a - gap_fb_bf / 2, a - gap_fb_ee / 2
    z_before = torch.tensor([[3., 0., 0., 2.8], [0., 3., 0., 2.8]], dtype=torch.float64)
    z_after = torch.tensor([[a, b, e, 2.8], [b, a, e, 2.8]], dtype=torch.float64)

    def pair_scores(z):
        return torch.stack([z[0, 0] + z[1, 1], z[0, 1] + z[1, 0], z[0, 2] + z[1, 2]])

    pair_results = []
    for phase, z in [('before', z_before), ('conditional_optimum', z_after)]:
        scores = pair_scores(z)
        pair_results.append({
            'phase': phase, 'logits': z.tolist(),
            'independent': [LABELS[j] for j in z.argmax(1).tolist()],
            'pair_choice': ['FB', 'BF', 'EE'][int(scores.argmax())],
            'weighted_smoothed_pair_loss': float(F.cross_entropy(
                scores[None], torch.tensor([0]), weight=weights, label_smoothing=eps)),
            'mean_row_loss': float(F.cross_entropy(z, torch.tensor([0, 1]))),
        })
    assert torch.allclose(pair_scores(z_after).softmax(0), effective_q)
    assert z_before.argmax(1).tolist() == [0, 1] and z_after.argmax(1).tolist() == [3, 3]
    assert pair_results[1]['weighted_smoothed_pair_loss'] < pair_results[0]['weighted_smoothed_pair_loss']

    # 真实loss不变方向的直接检查。
    shift = torch.tensor([4., -4., 0., 0.], dtype=torch.float64)
    assert torch.equal(pair_scores(z_before + shift), pair_scores(z_before))

    # 同一AdamW状态跨阶段：N的当前梯度为0，上一阶段动量仍能更新它。
    beta1, beta2, lr, old_grad = .9, .999, 2e-5, -.2
    m1, v1 = (1 - beta1) * old_grad, (1 - beta2) * old_grad**2
    m2, v2 = beta1 * m1, beta2 * v1
    displacement = -lr * (m2 / (1 - beta1**2)) / (math.sqrt(v2 / (1 - beta2**2)) + 1e-8)
    return {
        'mean_ce_is_not_accuracy': mean_ce_counterexample,
        'smoothed_pair_conditional_optimum': {
            'note': 'constructed logit-space counterexample using current weights and smoothing, not a prediction of actual model training',
            'label_smoothing': eps, 'class_weights': weights.tolist(),
            'effective_target_probability': effective_q.tolist(),
            'FB_minus_BF_optimum': gap_fb_bf, 'FB_minus_EE_optimum': gap_fb_ee,
            'states': pair_results,
        },
        'zero_pair_gradient_does_not_freeze_adam': {
            'pair_step_N_gradient': 0., 'previous_Row_N_gradient': old_grad,
            'lr': lr, 'second_step_N_parameter_displacement': displacement,
            'note': 'scalar example with no weight decay; does not quantify contribution in the actual checkpoint',
        },
        'local_direction_cannot_guarantee_finite_improvement': {
            'Row_loss': 'theta^2', 'Pair_loss': '(theta-1)^2', 'reference_theta': 0.,
            'Row_gradient_at_reference': 0.,
            'Row_loss_at_theta_0p1': .01, 'Pair_loss_at_theta_0p1': .81,
            'note': 'Row gradient is zero at its minimum, so first-order gradient protection alone cannot prevent a finite Row increase',
        },
    }


def main():
    previous = pd.read_csv(ROOT / 'experiments/deberta_manual_aug/outputs/run09_pair_no_smoothing/best_summary.csv')
    summary = {
        'training_performed': False, 'model_loaded': False,
        'stages': inspect_stages(), 'gradients': inspect_gradient_log(),
        'hand_calculations': hand_calculations(),
        'previous_zero_smoothing_control': previous[['method', 'weighted_f1', 'F_correct',
                                                     'B_correct', 'E_correct', 'N_correct']].to_dict('records'),
        'decision_limit': 'No probability of Dev F1 improvement can be estimated from these diagnostics alone.',
    }
    path = RESULTS_DIR / 'applicability_hand_calculations.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('结果：', path)


if __name__ == '__main__':
    main()
