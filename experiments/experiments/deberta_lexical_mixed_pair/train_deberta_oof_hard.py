"""读取OOF难例权重，训练最终的单个base模型。首次建议运行run_oof_all.py。"""
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
from experiments.deberta_lexical_mixed_pair import config_oof as cfg
from experiments.deberta_lexical_mixed_pair.oof_data import (
    WeightedGroupDataset, collate_weighted_groups, build_weight_table, oof_signature,
)
# 模型、特征和评分直接复用已核对过的词汇版，不另写一套。
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor, save_feature_table, DiCoDataset
from experiments.deberta_lexical_stage_probe.evaluation import (
    get_pair_logits, predict_logits, score_logits,
)


def build_optimizer(model):
    original_parameters = []
    lexical_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith('lexical_output.'):
            lexical_parameters.append(parameter)
        else:
            original_parameters.append(parameter)
    return torch.optim.AdamW([
        {'params': original_parameters, 'lr': cfg.LEARNING_RATE},
        {'params': lexical_parameters, 'lr': cfg.LEXICAL_LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)


def compute_weighted_joint_loss(logits, labels, pair_indices, pair_targets, sample_weights,
                                row_criterion, pair_criterion, pair_weight):
    # criterion保留原类别权重，reduction设为none；这里只新增样本权重。
    row_values = row_criterion(logits, labels)
    denominator = (row_criterion.weight[labels] * sample_weights).sum()
    row_loss = (row_values * sample_weights).sum() / denominator
    if len(pair_targets) > 0:
        first = logits[pair_indices[:, 0]]
        second = logits[pair_indices[:, 1]]
        pair_loss = pair_criterion(get_pair_logits(first, second), pair_targets)
    else:
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
        loss, row_loss, pair_loss = compute_weighted_joint_loss(
            logits, labels, batch['pair_indices'].to(device), batch['pair_targets'].to(device),
            batch['sample_weights'].to(device), row_criterion, pair_criterion, cfg.PAIR_WEIGHT,
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


def build_model(tokenizer, device):
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES,
    )
    return model.to(device)


def make_train_loader(frame, tokenizer, features, sample_weights):
    dataset = WeightedGroupDataset(frame, tokenizer, features, sample_weights, cfg.MAX_LENGTH)
    return DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
        collate_fn=collate_weighted_groups, generator=torch.Generator().manual_seed(cfg.SEED),
    )


def make_criteria(device):
    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, device=device), reduction='none',
    )
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    return row_criterion, pair_criterion


def main():
    set_seed(cfg.SEED)
    device = torch.device('cuda')
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    saved_signature = json.loads((cfg.OOF_DIR / 'oof_config.json').read_text(encoding='utf-8'))
    assert saved_signature == oof_signature(), 'OOF评分配置或Train已变，请更换OOF_RUN_NAME重新评分。'
    oof = pd.read_csv(cfg.OOF_DIR / 'train_oof_scores.csv', keep_default_na=False)
    weight_table, weight_groups = build_weight_table(train, oof, cfg.WEIGHT_MODE)

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    weight_path = cfg.RESULTS_DIR / 'train_sample_weights.csv'
    weight_table.to_csv(weight_path, index=False, encoding='utf-8-sig')
    weight_groups.to_csv(cfg.RESULTS_DIR / 'group_weights.csv', index=False, encoding='utf-8-sig')
    mass = weight_table.groupby('label').agg(rows=('instance_id', 'size'), weight_sum=('sample_weight', 'sum'))
    mass.to_csv(cfg.RESULTS_DIR / 'class_weight_totals.csv', encoding='utf-8-sig')
    print('最终加权模式：', cfg.WEIGHT_MODE)
    print('Train / Dev：', len(train), '/', len(dev))
    print('每类样本数量与归一化后的总权重：')
    print(mass.to_string())
    print('分组加权统计：')
    print(weight_groups.groupby('kind').agg(groups=('pair_id', 'size'), boosted=('boosted', 'sum')).to_string())

    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    train_loader = make_train_loader(train, tokenizer, train_features, weight_table['sample_weight'].to_numpy())
    dev_loader = make_dev_loader(dev, tokenizer, dev_features, cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE)
    model = build_model(tokenizer, device)
    optimizer = build_optimizer(model)
    row_criterion, pair_criterion = make_criteria(device)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    source_dir = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe'
    run_config = {
        'run_name': cfg.RUN_NAME, 'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED, 'dtype': 'float32',
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'epochs': cfg.EPOCHS, 'learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT, 'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'max_length': cfg.MAX_LENGTH, 'group_batch_size': cfg.GROUP_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'labels': cfg.FULL_LABELS,
        'train_rows': len(train), 'dev_rows': len(dev), 'train_groups': len(train_loader.dataset),
        'train_pairs': train_loader.dataset.pair_count, 'steps_per_epoch': len(train_loader),
        'selection': cfg.SELECT_BY, 'data_augmentation': False,
        'weight_mode': cfg.WEIGHT_MODE, 'hard_fraction': cfg.HARD_FRACTION,
        'hard_multiplier': cfg.HARD_MULTIPLIER, 'exclude_pair_ids': cfg.EXCLUDE_PAIR_IDS,
        'weight_normalization': 'each class sample-weight sum equals its original row count',
        'sample_weight_scope': 'Row CE only; Pair CE unchanged',
        'row_reduction': 'sum(sample_weight * class_weight * NLL) / sum(sample_weight * class_weight)',
        'oof_run_name': cfg.OOF_RUN_NAME, 'oof_signature': saved_signature,
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'weights_sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256((cfg.EXPERIMENT_DIR / name).read_bytes()).hexdigest()
            for name in ['config_oof.py', 'oof_data.py', 'score_train_oof.py', 'train_deberta_oof_hard.py']
        },
        'shared_source_sha256': {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ['model.py', 'lexical_features.py', 'data.py', 'evaluation.py']
        },
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )
    print('选模依据：', cfg.SELECT_BY)
    print('每轮训练步数：', len(train_loader))
    print('模型目录：', cfg.MODEL_DIR)
    print('结果目录：', cfg.RESULTS_DIR)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 最终模型 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_epoch(model, train_loader, optimizer, row_criterion, pair_criterion, device)
        print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
        print('各类采样行数：', {label: losses[f'seen_{label}'] for label in cfg.LABELS})
        epoch_dir = cfg.RESULTS_DIR / f'epoch_{epoch}'
        logits = predict_logits(model, dev_loader, device)
        summary = score_logits(logits, dev, epoch_dir, gold_path=cfg.GOLD_PATH)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary['method'] == cfg.SELECT_BY, 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            torch.save(model.state_dict(), cfg.MODEL_DIR / 'best_model.pth')
            info = {'best_epoch': epoch, 'selection': cfg.SELECT_BY, 'weighted_f1': score,
                    'same_checkpoint_metrics': summary.to_dict('records')}
            (cfg.MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，{cfg.SELECT_BY} F1={score:.6f}')
    print(f'\n最佳第 {best_epoch} 轮，独立F1={best_f1:.6f}')
    print('最佳权重：', cfg.MODEL_DIR / 'best_model.pth')
    print('独立/结构成绩：', cfg.RESULTS_DIR / 'best_summary.csv')


if __name__ == '__main__':
    main()
