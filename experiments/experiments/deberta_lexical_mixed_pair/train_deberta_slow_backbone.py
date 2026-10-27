"""右键运行：降低 DeBERTa 主干学习率，按独立 F1 保存最佳权重。"""
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config_slow_backbone as cfg
from experiments.deberta_lexical_mixed_pair.data import read_data, GroupDataset, collate_groups
# 模型、特征和评分直接复用已核对过的词汇版，不另写一套。
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor, save_feature_table, DiCoDataset
from experiments.deberta_lexical_stage_probe.evaluation import (
    get_pair_logits, predict_logits, score_logits, compare_with_original,
)


def build_optimizer(model):
    backbone_parameters = []
    head_parameters = []
    lexical_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith('deberta_model.'):
            # 已有 NLI 预训练知识：用较小学习率更新，没有冻结参数。
            backbone_parameters.append(parameter)
        elif name.startswith('lexical_output.'):
            lexical_parameters.append(parameter)
        else:
            # 新初始化的 CLS 分类头、投影、词级比较和 match_output。
            head_parameters.append(parameter)
    return torch.optim.AdamW([
        {'name': 'backbone', 'params': backbone_parameters, 'lr': cfg.BACKBONE_LEARNING_RATE},
        {'name': 'heads', 'params': head_parameters, 'lr': cfg.LEARNING_RATE},
        {'name': 'lexical', 'params': lexical_parameters, 'lr': cfg.LEXICAL_LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)


def compute_joint_loss(logits, labels, pair_indices, pair_targets,
                       row_criterion, pair_criterion, pair_weight):
    # 所有行都参与四分类 CE，包括 N 单例。分母始终是 F/B/E/N 四类。
    row_loss = row_criterion(logits, labels)
    if len(pair_targets) > 0:
        first = logits[pair_indices[:, 0]]
        second = logits[pair_indices[:, 1]]
        pair_loss = pair_criterion(get_pair_logits(first, second), pair_targets)
    else:
        # 全 N 批次没有配对目标，只做四分类训练。
        pair_loss = logits.new_zeros(())
    loss = row_loss + pair_weight * pair_loss
    return loss, row_loss, pair_loss


def train_epoch(model, loader, optimizer, row_criterion, pair_criterion, device):
    model.train()
    total_row = 0.0
    total_pair = 0.0
    label_counts = torch.zeros(4, dtype=torch.long)
    pair_count = 0
    batches_without_n = 0
    for batch in tqdm(loader, desc='四类混合联合训练'):
        labels = batch['labels'].to(device)
        optimizer.zero_grad(set_to_none=True)
        # 只做一次前向，同一份输出同时用于单行和配对监督。
        logits = model(
            batch['input_ids'].to(device), batch['attention_mask'].to(device),
            batch['features'].to(device),
        )
        loss, row_loss, pair_loss = compute_joint_loss(
            logits, labels, batch['pair_indices'].to(device), batch['pair_targets'].to(device),
            row_criterion, pair_criterion, cfg.PAIR_WEIGHT,
        )
        loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        total_pair += pair_loss.item()
        label_counts += torch.bincount(batch['labels'], minlength=4)
        pair_count += len(batch['pair_targets'])
        batches_without_n += int(not (batch['labels'] == 3).any())
    row_mean = total_row / len(loader)
    pair_mean = total_pair / len(loader)
    return {
        'row_loss': row_mean, 'pair_loss': pair_mean,
        'total_loss': row_mean + cfg.PAIR_WEIGHT * pair_mean,
        'train_rows': int(label_counts.sum()), 'train_pairs': pair_count,
        'optimizer_steps': len(loader), 'batches_without_N': batches_without_n,
        **{f'seen_{label}': int(count) for label, count in zip(cfg.LABELS, label_counts)},
    }


def make_dev_loader(dev, tokenizer, features, max_length, batch_size):
    return DataLoader(
        DiCoDataset(dev, tokenizer, features, max_length),
        batch_size=batch_size, shuffle=False,
        # Dev 迭代器使用独立随机数，不改变训练的 shuffle 或 dropout 序列。
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )


def main():
    set_seed(cfg.SEED)
    device = torch.device('cuda')
    train, dev = read_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    train_dataset = GroupDataset(train, tokenizer, train_features, cfg.MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
        collate_fn=collate_groups, generator=torch.Generator().manual_seed(cfg.SEED),
    )
    dev_loader = make_dev_loader(dev, tokenizer, dev_features, cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE)
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES,
    ).to(device)
    optimizer = build_optimizer(model)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    save_feature_table(train, train_features, cfg.RESULTS_DIR / 'train_features.csv')
    save_feature_table(dev, dev_features, cfg.RESULTS_DIR / 'dev_features.csv')
    source_dir = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe'
    run_config = {
        'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED, 'dtype': 'float32',
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'epochs': cfg.EPOCHS, 'learning_rate': cfg.LEARNING_RATE,
        'backbone_learning_rate': cfg.BACKBONE_LEARNING_RATE,
        'head_learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT, 'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'max_length': cfg.MAX_LENGTH, 'group_batch_size': cfg.GROUP_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'labels': cfg.FULL_LABELS,
        'train_rows': len(train), 'dev_rows': len(dev), 'train_groups': len(train_dataset),
        'train_pairs': train_dataset.pair_count, 'steps_per_epoch': len(train_loader),
        'selection': cfg.SELECT_BY, 'data_augmentation': False,
        'row_exposure': 'each original row exactly once per epoch',
        'pair_exposure': 'each real pair exactly once per epoch',
        'training': 'mixed groups; one forward; Row CE + pair_weight * Pair CE',
        'comparison_run': 'run01_mixed_pair',
        'training_change': 'only DeBERTa backbone LR: 2e-5 -> 5e-6; heads 2e-5; lexical 1e-3',
        'selection_change': 'save by independent F1; compare baseline using its independent-selected epoch 2',
        'comparison_note': 'same mixed data schedule as run01; compare best independent F1 over 10 epochs for both runs',
        'loss_reporting': 'mean of per-step criterion values; batches without pairs contribute zero Pair CE',
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'shared_source_sha256': {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ['model.py', 'lexical_features.py', 'data.py', 'evaluation.py']
        },
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )
    print('模型：DeBERTa + Token Alignment + 原词汇分支')
    print('Train / Dev：', len(train), '/', len(dev))
    print('训练组数 / 真实 Pair：', len(train_dataset), '/', train_dataset.pair_count)
    print('每步最多行数 / 每轮步数：', cfg.GROUP_BATCH_SIZE * 2, '/', len(train_loader))
    print('损失：完整四分类 CE +', cfg.PAIR_WEIGHT, '* Pair CE')
    for group in optimizer.param_groups:
        print('参数组：', group['name'], '；学习率：', group['lr'])
    print('本次按 independent Dev Weighted F1 保存最佳权重。')
    print('模型目录：', cfg.MODEL_DIR)
    print('结果目录：', cfg.RESULTS_DIR)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_epoch(model, train_loader, optimizer, row_criterion, pair_criterion, device)
        print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
        print('本轮各类训练行数：', {label: losses[f'seen_{label}'] for label in cfg.LABELS})
        epoch_dir = cfg.RESULTS_DIR / f'epoch_{epoch}'
        logits = predict_logits(model, dev_loader, device)
        summary = score_logits(logits, dev, epoch_dir, gold_path=cfg.GOLD_PATH)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(
            cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
        )
        score = float(summary.loc[summary['method'] == cfg.SELECT_BY, 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            torch.save(model.state_dict(), cfg.MODEL_DIR / 'best_model.pth')
            info = {
                'best_epoch': epoch, 'selection': cfg.SELECT_BY, 'weighted_f1': score,
                'same_checkpoint_metrics': summary.to_dict('records'),
            }
            (cfg.MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，{cfg.SELECT_BY} F1={score:.6f}')
    best_prediction = cfg.RESULTS_DIR / f'epoch_{best_epoch}/predictions_with_scores.csv'
    compare_with_original(best_prediction, cfg.RESULTS_DIR / 'comparison_with_original')
    baseline_history = pd.read_csv(cfg.EXPERIMENT_DIR / 'outputs/run01_mixed_pair/epoch_summary.csv')
    baseline = baseline_history[baseline_history['method'] == 'independent']
    baseline_best = baseline.loc[baseline['weighted_f1'].idxmax()]
    print(
        f"旧混合版独立 F1 基准：第 {int(baseline_best['epoch'])} 轮，"
        f"F1={baseline_best['weighted_f1']:.6f}"
    )
    baseline_metrics = baseline_history[baseline_history['epoch'] == int(baseline_best['epoch'])].copy()
    baseline_metrics['run'] = 'run01_mixed_pair'
    current_metrics = pd.read_csv(cfg.RESULTS_DIR / 'best_summary.csv')
    current_metrics['run'] = cfg.RUN_NAME
    current_metrics['epoch'] = best_epoch
    comparison = pd.concat([baseline_metrics, current_metrics], ignore_index=True)
    comparison = comparison[['run', 'epoch', 'method', 'weighted_f1', 'N_correct', 'F_correct', 'B_correct', 'E_correct']]
    comparison.to_csv(cfg.RESULTS_DIR / 'comparison_independent_selected.csv', index=False, encoding='utf-8-sig')
    print('\n两边均按 independent 选模，列出各自同一权重的两种成绩：')
    print(comparison.to_string(index=False))
    print(f'\n最佳第 {best_epoch} 轮，{cfg.SELECT_BY} F1={best_f1:.6f}')
    print('最佳权重：', cfg.MODEL_DIR / 'best_model.pth')
    print('该权重的独立及结构成绩：', cfg.RESULTS_DIR / 'best_summary.csv')
    print('内存里是最后一轮；复测最佳权重请运行 evaluate_slow_backbone.py。')


if __name__ == '__main__':
    main()
