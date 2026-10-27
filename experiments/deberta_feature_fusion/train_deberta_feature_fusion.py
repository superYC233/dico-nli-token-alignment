import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_feature_fusion import config as cfg
from experiments.deberta_feature_fusion.data import (
    read_data, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_feature_fusion.model import DebertaDiCoModel
from experiments.deberta_feature_fusion.evaluation import (
    get_pair_logits, evaluate_without_changing_rng, compare_with_lexical,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )


def build_model(tokenizer, device):
    # 从同一个NLI预训练主干重新初始化，不加载旧DiCo最佳权重。
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    return DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE,
        alignment_dimension=cfg.ALIGNMENT_DIMENSION,
        alignment_temperature=cfg.ALIGNMENT_TEMPERATURE,
        fusion_hidden=cfg.FUSION_HIDDEN_SIZE, dropout=cfg.DROPOUT,
    ).to(device)


def build_optimizer(model):
    original_parameters = []
    lexical_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('lexical_encoder.'):
            lexical_parameters.append(parameter)
        else:
            original_parameters.append(parameter)
    return torch.optim.AdamW([
        # 主干、对齐、共同分类头保持原分类头的学习率。
        {'params': original_parameters, 'lr': cfg.LEARNING_RATE},
        {'params': lexical_parameters, 'lr': cfg.LEXICAL_LEARNING_RATE},
    ], lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)


def build_criteria(device):
    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float32, device=device),
    )
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float32, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    return row_criterion, pair_criterion


def train_rows(model, loader, optimizer, criterion, device):
    total_loss = 0.0
    for input_ids, attention_mask, features, labels in tqdm(loader, desc='四分类训练'):
        logits = model(input_ids.to(device), attention_mask.to(device), features.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    total_loss = 0.0
    for ids0, mask0, features0, ids1, mask1, features1, targets in tqdm(loader, desc='Real Pair CE 训练'):
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        loss = cfg.PAIR_WEIGHT * pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += pair_loss.item()
    return total_loss / len(loader)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_run_config(model, train_rows_count, dev_rows_count, real_pairs):
    source_paths = [
        Path(__file__), cfg.EXPERIMENT_DIR / 'config.py',
        cfg.EXPERIMENT_DIR / 'model.py', cfg.EXPERIMENT_DIR / 'data.py',
        cfg.EXPERIMENT_DIR / 'evaluation.py',
        PROJECT_ROOT / 'experiments/deberta_lexical/data.py',
        PROJECT_ROOT / 'experiments/deberta_lexical/config.py',
        PROJECT_ROOT / 'experiments/deberta_lexical/evaluation.py',
        PROJECT_ROOT / 'experiments/deberta_lexical/lexical_features.py',
    ]
    return {
        'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED,
        'architecture': 'normalized CLS + token comparison + lexical features, one joint classifier',
        'alignment_dimension': cfg.ALIGNMENT_DIMENSION,
        'alignment_temperature': cfg.ALIGNMENT_TEMPERATURE,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'fusion_hidden_size': cfg.FUSION_HIDDEN_SIZE,
        'fusion_input_size': model.fusion_input_size, 'dropout': cfg.DROPOUT,
        'use_cnn': False, 'use_lora': False,
        'head_parameters': sum(p.numel() for name, p in model.named_parameters()
                               if not name.startswith('deberta_model.')),
        'lexical_parameters': sum(p.numel() for p in model.lexical_encoder.parameters()),
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'feature_source_sha256': file_hash(PROJECT_ROOT / 'experiments/deberta_lexical/lexical_features.py'),
        'source_sha256': {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in source_paths},
        'train_sha256': file_hash(cfg.TRAIN_PATH), 'dev_sha256': file_hash(cfg.DEV_PATH),
        'train_rows': train_rows_count, 'dev_rows': dev_rows_count,
        'real_pairs': real_pairs, 'data_augmentation': False,
        'epochs': cfg.EPOCHS, 'row_batch_size': cfg.ROW_BATCH_SIZE,
        'pair_batch_size': cfg.PAIR_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'max_length': cfg.MAX_LENGTH, 'learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE,
        'weight_decay': cfg.WEIGHT_DECAY, 'row_weights': cfg.ROW_WEIGHTS,
        'pair_weights': cfg.PAIR_WEIGHTS, 'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'dtype': 'float32', 'labels': cfg.FULL_LABELS,
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
        'selection': 'after Pair CE: best Dev pair-aware weighted F1; earliest epoch on ties',
    }


def main(prepare_only=False):
    train, dev = read_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    real_pairs = int(train.groupby('pair_id').size().eq(2).sum())
    print('结构：NLI DeBERTa + 原Token Alignment + Lexical特征 → 一个共同分类头。')
    print('Train / Dev：', len(train), '/', len(dev), '；官方数据，无增强')
    print('Train标签数量：', train['label'].value_counts().to_dict())
    print('Real Pair组数：', real_pairs)
    print('每轮：', math.ceil(len(train) / cfg.ROW_BATCH_SIZE), '批Row CE →',
          math.ceil(real_pairs / cfg.PAIR_BATCH_SIZE), '批0.5×Pair CE')
    if prepare_only:
        print('数据检查完成，没有加载模型或启动训练。')
        return

    model_dir, results_dir = cfg.get_output_dirs()
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError('本次输出目录已存在，请在config.py修改RUN_NAME。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA；本入口用于正式GPU训练。')
    device = torch.device('cuda')
    set_seed(cfg.SEED)
    tokenizer = load_tokenizer()
    model = build_model(tokenizer, device)
    row_loader = DataLoader(
        DiCoDataset(train, tokenizer, train_features, cfg.MAX_LENGTH),
        batch_size=cfg.ROW_BATCH_SIZE, shuffle=True,
    )
    pair_loader = DataLoader(
        PairDataset(train, tokenizer, train_features, cfg.MAX_LENGTH),
        batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True,
    )
    dev_loader = DataLoader(
        DiCoDataset(dev, tokenizer, dev_features, cfg.MAX_LENGTH),
        batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
    )
    row_criterion, pair_criterion = build_criteria(device)
    optimizer = build_optimizer(model)
    run_config = make_run_config(model, len(train), len(dev), real_pairs)
    print('融合输入维度：', model.fusion_input_size)
    print('主干外可训练参数：', run_config['head_parameters'])
    print('主干/对齐/共同分类头学习率：', cfg.LEARNING_RATE)
    print('词汇特征编码器学习率：', cfg.LEXICAL_LEARNING_RATE)
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)

    model_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    save_feature_table(train, train_features, results_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, results_dir / 'dev_features.csv')
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2)
    (model_dir / 'run_config.json').write_text(config_text, encoding='utf-8')
    (results_dir / 'run_config.json').write_text(config_text, encoding='utf-8')

    best_f1 = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        epoch_dir = results_dir / f'epoch_{epoch}'
        model.train()
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        print(f'\n===== 第 {epoch} 轮：Row CE后 =====')
        row_summary = evaluate_without_changing_rng(
            model, dev_loader, dev, device, epoch_dir / 'after_row',
        )

        model.train()
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
        print(f'\n===== 第 {epoch} 轮：Pair CE后 =====')
        pair_summary = evaluate_without_changing_rng(model, dev_loader, dev, device, epoch_dir)
        for stage, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, stage=stage, row_loss=row_loss, pair_loss=pair_loss)
                history.append(record)
        pd.DataFrame(history).to_csv(
            results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
        )
        row_counts = row_summary.set_index('method')
        pair_counts = pair_summary.set_index('method')
        for method in ['independent', 'pair_aware']:
            e_change = int(pair_counts.loc[method, 'E_correct'] - row_counts.loc[method, 'E_correct'])
            n_change = int(pair_counts.loc[method, 'N_correct'] - row_counts.loc[method, 'N_correct'])
            fbe_change = int(pair_counts.loc[method, 'FBE_correct'] - row_counts.loc[method, 'FBE_correct'])
            print(f'{method}，Pair阶段变化：E {e_change:+d}，N {n_change:+d}，FBE {fbe_change:+d}')

        # 只在Pair阶段后的结构F1刷新时保存一份完整模型。
        score = float(pair_counts.loc['pair_aware', 'weighted_f1'])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            best_info = {
                'best_epoch': epoch, 'stage': 'after_pair', 'weighted_f1': score,
                'independent_f1': float(pair_counts.loc['independent', 'weighted_f1']),
                'N_correct': int(pair_counts.loc['pair_aware', 'N_correct']),
                'FBE_correct': int(pair_counts.loc['pair_aware', 'FBE_correct']),
            }
            (model_dir / 'best_metrics.json').write_text(
                json.dumps(best_info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            pair_summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')

    best_prediction = results_dir / f'epoch_{best_epoch}' / 'predictions_with_scores.csv'
    compare_with_lexical(best_prediction, results_dir / 'comparison_with_lexical')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', model_dir / 'best_model.pth')
    print('最佳预测：', best_prediction)
    print('重新预测请运行evaluate_best.py；它会读取最佳权重。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(prepare_only=args.prepare_only)
