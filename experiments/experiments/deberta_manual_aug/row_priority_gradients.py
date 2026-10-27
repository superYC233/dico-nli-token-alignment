"""分别计算Row/Pair梯度；只在冲突时投影Pair，Row梯度不投影。"""
import math

import torch


def parameter_group(name):
    if name.startswith('deberta_model.encoder.layer.'):
        layer = int(name.split('.')[3])
        return f'encoder_layer_{layer:02d}'
    if name.startswith('deberta_model.embeddings.'):
        return 'embeddings'
    if name.startswith('deberta_model.'):
        return 'encoder_other'
    if name.startswith('output.'):
        return 'cls_head'
    return 'alignment_head'


def separate_gradients(row_loss, pair_loss, parameters, pair_weight):
    # 两次反向共用同一次前向和同一份dropout；不会写入parameter.grad。
    row_gradients = torch.autograd.grad(
        row_loss, parameters, retain_graph=pair_loss is not None, allow_unused=True,
    )
    if pair_loss is None:
        pair_gradients = (None,) * len(parameters)
    else:
        # 这里已经乘了0.5，后面组合时不能再乘一次。
        pair_gradients = torch.autograd.grad(
            pair_weight * pair_loss, parameters, allow_unused=True,
        )
    return row_gradients, pair_gradients


def measure_gradients(names, row_gradients, pair_gradients):
    """全模型与分层统计；实际投影只使用全模型的一个系数。"""
    groups = list(dict.fromkeys(parameter_group(name) for name in names))
    positions = {group: index for index, group in enumerate(groups)}
    first = next(g for g in row_gradients if g is not None)
    values = torch.zeros(len(groups), 3, dtype=torch.float64, device=first.device)
    with torch.no_grad():
        for name, row, pair in zip(names, row_gradients, pair_gradients):
            index = positions[parameter_group(name)]
            if row is not None:
                values[index, 0] += torch.linalg.vector_norm(row).double().square()
            if pair is not None:
                values[index, 1] += torch.linalg.vector_norm(pair).double().square()
            if row is not None and pair is not None:
                values[index, 2] += torch.dot(row.reshape(-1), pair.reshape(-1)).double()
        # 先按参数求平方和/内积，再求全模型值；不能先分别归一化各层。
        values = torch.cat([values.sum(dim=0, keepdim=True), values], dim=0).cpu().tolist()

    records = []
    for group, (row_square, pair_square, dot) in zip(['all'] + groups, values):
        row_norm = math.sqrt(row_square)
        pair_norm = math.sqrt(pair_square)
        both = row_norm > 0 and pair_norm > 0
        cosine = max(-1.0, min(1.0, dot / (row_norm * pair_norm))) if both else None
        records.append({
            'group': group, 'row_grad_norm': row_norm,
            'weighted_pair_grad_norm': pair_norm, 'row_pair_dot': dot,
            'cosine': cosine, 'has_both_gradients': both,
            'conflict': both and dot < 0,
            'pair_to_row_norm_ratio': pair_norm / row_norm if row_norm > 0 else None,
        })
    return records


def projection_coefficient(global_record):
    # gp' = gp - c * gr；没有冲突或Row梯度为0时，c=0。
    if not global_record['conflict']:
        return 0.0
    return global_record['row_pair_dot'] / global_record['row_grad_norm'] ** 2


def assign_gradients(parameters, row_gradients, pair_gradients, coefficient):
    """g = gr + (gp - c*gr)。只写梯度，不在此处更新模型参数。"""
    with torch.no_grad():
        for parameter, row, pair in zip(parameters, row_gradients, pair_gradients):
            if row is None:
                parameter.grad = pair
            else:
                # 即使该参数的Pair梯度为None，全局投影也有-c*gr这一项。
                gradient = row.mul_(1.0 - coefficient)
                if pair is not None:
                    gradient.add_(pair)
                parameter.grad = gradient
