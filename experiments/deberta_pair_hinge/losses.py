import torch
import torch.nn.functional as F


def three_combination_scores(first, second):
    return torch.stack([
        first[:, 0] + second[:, 1],  # FB
        first[:, 1] + second[:, 0],  # BF
        first[:, 2] + second[:, 2],  # EE
    ], dim=1)


def hinge16(first, second, first_labels, second_labels, margin=1.0):
    """返回每组hinge：金标组合对抗其余15个组合，包含所有含N负例。"""
    # 顺序：FF、FB、FE、FN、BF……NN；不是只在四个合法组合中选。
    scores = (first.unsqueeze(2) + second.unsqueeze(1)).flatten(start_dim=1)
    gold_indices = first_labels * 4 + second_labels
    gold_scores = scores.gather(1, gold_indices[:, None]).squeeze(1)

    # 必须排除金标自身，否则已经满足margin时仍会产生恒定损失。
    negative_scores = scores.scatter(1, gold_indices[:, None], -torch.inf)
    hardest_negative = negative_scores.max(dim=1).values
    return F.relu(margin + hardest_negative - gold_scores)


def compute_joint_loss(logits, labels, pair_indices, pair_targets,
                       row_criterion, pair_criterion, pair_loss='hinge16',
                       margin=1.0, pair_weight=0.5):
    # 所有行都算四分类CE，N单例也直接更新模型。
    row_loss = row_criterion(logits, labels)
    if len(pair_targets) == 0:
        pair_value = logits.new_zeros(())
    else:
        index0, index1 = pair_indices[:, 0], pair_indices[:, 1]
        first, second = logits[index0], logits[index1]
        if pair_loss == 'ce3':
            pair_value = pair_criterion(three_combination_scores(first, second), pair_targets)
        elif pair_loss == 'hinge16':
            per_pair = hinge16(first, second, labels[index0], labels[index1], margin)
            # 与加权CE相同的金标组权重及加权均值分母。
            weights = pair_criterion.weight[pair_targets]
            pair_value = (per_pair * weights).sum() / weights.sum()
        else:
            raise ValueError('pair_loss只能为ce3或hinge16。')
    return row_loss + pair_weight * pair_value, row_loss, pair_value
