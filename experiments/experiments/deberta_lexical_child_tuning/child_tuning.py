"""Child-Tuning-D梯度掩码：只从Train的Row CE估计，保留原AdamW。"""
import numpy as np
import torch
from tqdm import tqdm


ENCODER_PREFIX = 'deberta_model.encoder.layer.'


def select_fisher_mask(fisher, reserve_p):
    # 全部encoder层共用一个分位数，不是每层分别取20%。
    total = sum(value.numel() for value in fisher.values())
    all_scores = np.empty(total, dtype=np.float32)
    offset = 0
    for value in fisher.values():
        size = value.numel()
        all_scores[offset:offset + size] = value.detach().cpu().numpy().reshape(-1)
        offset += size
    threshold = float(np.percentile(all_scores, (1 - reserve_p) * 100, overwrite_input=True))
    masks = {name: value >= threshold for name, value in fisher.items()}
    records = []
    for name, mask in masks.items():
        records.append({
            'parameter': name, 'coordinates': mask.numel(),
            'retained': int(mask.sum().item()),
        })
    retained = sum(row['retained'] for row in records)
    stats = {
        'reserve_p': reserve_p, 'threshold': threshold,
        'eligible_coordinates': total, 'retained_coordinates': retained,
        'actual_retained_fraction': retained / total,
        'encoder_prefix': ENCODER_PREFIX,
    }
    return masks, stats, records


def calculate_fisher_mask(model, loader, row_criterion, device, reserve_p=0.2,
                         clip_norm=1.0, seed=142):
    if not 0 < reserve_p <= 1:
        raise ValueError('RESERVE_P必须在(0,1]内。')
    parameters = {
        name: parameter for name, parameter in model.named_parameters()
        if name.startswith(ENCODER_PREFIX) and parameter.requires_grad
    }
    if not parameters:
        raise ValueError('未找到DeBERTa encoder.layer参数，不能生成Child-D掩码。')
    fisher = {name: torch.zeros_like(parameter, dtype=torch.float32)
              for name, parameter in parameters.items()}
    was_training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    counts = torch.zeros(4, dtype=torch.long)
    model.zero_grad(set_to_none=True)
    # Fisher遍历使用自己的随机数，完成后还原正式训练的CPU/CUDA dropout状态。
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        model.train()
        try:
            for ids, attention_mask, features, labels in tqdm(loader, desc='Train Row CE：估计Fisher'):
                logits = model(ids.to(device), attention_mask.to(device), features.to(device))
                loss = row_criterion(logits, labels.to(device))
                loss.backward()
                for name, parameter in parameters.items():
                    if parameter.grad is not None:
                        # 与作者实现一致：逐参数裁剪，再累积批梯度平方。
                        torch.nn.utils.clip_grad_norm_([parameter], clip_norm)
                        gradient = parameter.grad.float()
                        fisher[name].addcmul_(gradient, gradient, value=1 / len(loader))
                counts += torch.bincount(labels.cpu(), minlength=4)
                model.zero_grad(set_to_none=True)
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
    masks, stats, records = select_fisher_mask(fisher, reserve_p)
    stats.update({
        'batches': len(loader), 'train_rows': int(counts.sum()),
        'label_counts_F_B_E_N': counts.tolist(), 'seed': seed,
        'fisher_grad_clip': clip_norm,
        'objective': 'Train weighted four-class Row CE only; no Pair CE or Dev',
        'estimator': 'mean squared clipped batch gradients, not exact per-example Fisher',
        'parameters_updated_during_fisher': False,
    })
    return masks, stats, records


@torch.no_grad()
def apply_gradient_mask(model, masks):
    # 在Adam的一阶/二阶动量更新之前遮罩；不更换优化器，不改weight decay。
    for name, parameter in model.named_parameters():
        if name in masks and parameter.grad is not None:
            parameter.grad.mul_(masks[name])
