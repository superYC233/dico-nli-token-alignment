"""只检查 CPU 损失计算与配置，不加载预训练模型，不更新模型参数。"""
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from experiments.deberta_lexical_mixed_pair import config_rdrop as cfg
from experiments.deberta_lexical_mixed_pair.train_deberta_rdrop import (
    compute_joint_loss, compute_rdrop_loss, symmetric_kl,
)


def check_config():
    source = cfg.EXPERIMENT_DIR / 'outputs/run01_mixed_pair/run_config.json'
    baseline = json.loads(source.read_text(encoding='utf-8'))
    for config_name, saved_name in [
        ('MODEL_NAME', 'model_name'), ('SEED', 'seed'), ('EPOCHS', 'epochs'),
        ('LEARNING_RATE', 'learning_rate'), ('LEXICAL_LEARNING_RATE', 'lexical_learning_rate'),
        ('WEIGHT_DECAY', 'weight_decay'), ('GROUP_BATCH_SIZE', 'group_batch_size'),
        ('DEV_BATCH_SIZE', 'dev_batch_size'), ('MAX_LENGTH', 'max_length'),
        ('ROW_WEIGHTS', 'row_weights'), ('PAIR_WEIGHTS', 'pair_weights'),
        ('PAIR_WEIGHT', 'pair_weight'), ('PAIR_LABEL_SMOOTHING', 'pair_label_smoothing'),
        ('USE_LEXICAL_FEATURES', 'use_lexical_features'),
        ('LEXICAL_HIDDEN_SIZE', 'lexical_hidden_size'),
    ]:
        assert getattr(cfg, config_name) == baseline[saved_name], config_name
    assert cfg.SELECT_BY == 'independent' and cfg.RDROP_WEIGHT == 1.0
    assert cfg.RUN_NAME == 'run05_mixed_rdrop'
    print('run01参数配方、R-Drop权重和独立F1选模配置通过。')


def check_kl():
    first = torch.tensor([[0.7, 0.1, 0.1, 0.1]]).log()
    second = torch.tensor([[0.1, 0.7, 0.1, 0.1]]).log()
    expected = torch.tensor(0.6 * math.log(7))
    result = symmetric_kl(first, second)
    assert torch.allclose(result, expected, atol=1e-6)
    assert torch.allclose(result, symmetric_kl(second, first), atol=1e-6)
    assert torch.allclose(symmetric_kl(first, first), torch.tensor(0.0), atol=1e-6)
    # 重复样本不能把KL变成三倍，检查按样本平均的口径。
    assert torch.allclose(result, symmetric_kl(first.repeat(3, 1), second.repeat(3, 1)))

    # 只改变N分数，也必须影响四类KL；用小矩阵检查两边梯度，不训练模型。
    first = torch.tensor([[2.0, 0.0, 0.0, 0.0]], requires_grad=True)
    second = torch.tensor([[2.0, 0.0, 0.0, 4.0]], requires_grad=True)
    kl = symmetric_kl(first, second)
    gradients = torch.autograd.grad(kl, (first, second))
    assert kl.item() > 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient[0, 3].abs().item() > 0 for gradient in gradients)
    print('对称KL、批次均值、N列和两次输出的梯度通过。')


def check_combined_loss():
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    labels = torch.tensor([3, 0, 1, 3, 2, 2])
    pair_indices = torch.tensor([[1, 2], [4, 5]])
    pair_targets = torch.tensor([0, 2])
    first = torch.tensor([
        [0.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 2.0, 0.0],
    ])
    second = first + torch.tensor([0.2, -0.1, 0.3, 0.0])
    total, row, pair, kl = compute_rdrop_loss(
        first, second, labels, pair_indices, pair_targets,
        row_criterion, pair_criterion, cfg.PAIR_WEIGHT, cfg.RDROP_WEIGHT,
    )
    first_total, first_row, first_pair = compute_joint_loss(
        first, labels, pair_indices, pair_targets, row_criterion, pair_criterion, cfg.PAIR_WEIGHT,
    )
    second_total, second_row, second_pair = compute_joint_loss(
        second, labels, pair_indices, pair_targets, row_criterion, pair_criterion, cfg.PAIR_WEIGHT,
    )
    assert torch.allclose(row, 0.5 * (first_row + second_row))
    assert torch.allclose(pair, 0.5 * (first_pair + second_pair))
    assert torch.allclose(total, 0.5 * (first_total + second_total) + cfg.RDROP_WEIGHT * kl)

    # 两次输出一致时KL为0，应恢复原联合损失。
    identical = compute_rdrop_loss(
        first, first, labels, pair_indices, pair_targets,
        row_criterion, pair_criterion, cfg.PAIR_WEIGHT, cfg.RDROP_WEIGHT,
    )
    assert torch.allclose(identical[0], first_total, atol=1e-6)

    # 全N批次也正常参与Row CE和KL；没有虚构的Pair监督。
    empty_indices = torch.empty(0, 2, dtype=torch.long)
    empty_targets = torch.empty(0, dtype=torch.long)
    n_labels = torch.tensor([3, 3])
    plain = torch.zeros(2, 4)
    better = plain.clone()
    better[:, 3] = 3.0
    n_loss = compute_rdrop_loss(
        plain, plain, n_labels, empty_indices, empty_targets,
        row_criterion, pair_criterion, cfg.PAIR_WEIGHT, cfg.RDROP_WEIGHT,
    )
    better_loss = compute_rdrop_loss(
        better, better, n_labels, empty_indices, empty_targets,
        row_criterion, pair_criterion, cfg.PAIR_WEIGHT, cfg.RDROP_WEIGHT,
    )
    assert n_loss[2].item() == 0.0 and better_loss[2].item() == 0.0
    assert better_loss[0] < n_loss[0]
    print('两次CE平均、Pair系数、KL合计和全N批次监督通过。')


if __name__ == '__main__':
    check_config()
    check_kl()
    check_combined_loss()
    print('全部检查通过，没有加载预训练权重或更新模型参数。')
