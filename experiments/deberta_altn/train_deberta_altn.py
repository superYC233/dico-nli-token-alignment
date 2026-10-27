import argparse
import hashlib
import json
import random
import shutil
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
from experiments.deberta_altn import config as cfg
from experiments.deberta_altn.data import (
    read_data, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_altn.altn_features import FEATURE_NAMES, FEATURE_VERSION, ALIGNMENT_METHOD
from experiments.deberta_altn.model import DebertaDiCoModel
from experiments.deberta_altn.evaluation import (
    get_pair_logits, predict_logits, score_logits, compare_with_original, compare_with_lexical,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    original_parameters = []
    lexical_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(('lexical_output.', 'position_projection.')):
            lexical_parameters.append(parameter)
        else:
            # DeBERTa 主干、CLS 分类头、Token Alignment 都保持原学习率。
            original_parameters.append(parameter)

    parameter_groups = [
        {'params': original_parameters, 'lr': cfg.LEARNING_RATE},
    ]
    if lexical_parameters:
        parameter_groups.append({
            'params': lexical_parameters, 'lr': cfg.LEXICAL_LEARNING_RATE,
        })

    return torch.optim.AdamW(
        parameter_groups, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )


def save_best_weights(model, path):
    # 先写临时文件，成功后替换旧最佳；写盘失败时保留原有最佳模型。
    path = Path(path)
    state = model.state_dict()
    required = sum(value.numel() * value.element_size() for value in state.values())
    if shutil.disk_usage(path.parent).free < required + 64 * 1024 ** 2:
        raise OSError('模型目录所在磁盘空间不足，请先清理空间。')
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_rows(model, loader, optimizer, criterion, device):
    total_loss = 0.0
    for input_ids, attention_mask, features, labels in tqdm(loader, desc='四分类训练'):
        logits = model(input_ids.to(device), attention_mask.to(device), features.to(device))
        loss = criterion(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise ValueError('四分类损失出现 NaN 或无穷。')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    total_loss = 0.0
    for ids0, mask0, features0, ids1, mask1, features1, targets in tqdm(loader, desc='Real Pair CE 训练'):
        # 两个方向分别前向，分别输入对应方向的词汇特征。
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_logits = get_pair_logits(first, second)
        pair_loss = criterion(pair_logits, targets.to(device))
        if not torch.isfinite(pair_loss):
            raise ValueError('Pair 损失出现 NaN 或无穷。')
        loss = cfg.PAIR_WEIGHT * pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += pair_loss.item()
    return total_loss / len(loader)


def evaluate_without_changing_rng(model, loader, dev, device, output_dir):
    # 逐轮评估不应消耗下一轮 shuffle / dropout 的随机序列。
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    try:
        logits = predict_logits(model, loader, device)
        return score_logits(logits, dev, output_dir)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def make_run_config(use_position_features, train_rows_count, dev_rows_count, real_pairs):
    folder = Path(__file__).resolve().parent
    return {
        'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED,
        'use_lexical_features': True, 'use_position_features': use_position_features,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'feature_source_sha256': {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ['lexical_features.py', 'altn_features.py', 'model.py']
        },
        'alignment_method': ALIGNMENT_METHOD,
        'position_feature_size': 8, 'position_projection_initialization': 'zero',
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
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
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
    }


def main(use_position_features=cfg.USE_POSITION_FEATURES):
    model_dir, results_dir = cfg.get_output_dirs(use_position_features)
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError('本次输出目录已存在，请在 config.py 修改 RUN_NAME，避免覆盖旧结果。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA；本入口用于正式 GPU 训练。')
    device = torch.device('cuda')
    train, dev = read_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    print('保留原 20 维词汇分支；新增差异位置特征：', use_position_features)
    print('Train / Dev：', len(train), '/', len(dev))
    print('特征维度：20 + 8 =', len(FEATURE_NAMES))
    if not use_position_features:
        print('当前为 Lexical 对照：模型只读取前 20 维。')

    # 与原版一样，从 NLI 预训练主干开始；不加载已选好的 epoch_7 继续训练。
    set_seed(cfg.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
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
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_position_features=use_position_features,
    ).to(device)

    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float32, device=device),
    )
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float32, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    optimizer = build_optimizer(model)

    print('Real Pair 组数：', len(pair_loader.dataset))
    print('原模型学习率：', optimizer.param_groups[0]['lr'])
    print('原词汇分支参数量：', sum(p.numel() for p in model.lexical_output.parameters()))
    if use_position_features:
        print('新增位置投影参数量：', sum(p.numel() for p in model.position_projection.parameters()))
    print('词汇分支 / 位置投影学习率：', optimizer.param_groups[1]['lr'])
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    model_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    save_feature_table(train, train_features, results_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, results_dir / 'dev_features.csv')
    run_config = make_run_config(use_position_features, len(train), len(dev), len(pair_loader.dataset))
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2)
    (model_dir / 'run_config.json').write_text(config_text, encoding='utf-8')
    (results_dir / 'run_config.json').write_text(config_text, encoding='utf-8')

    best_f1 = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        model.train()
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
        epoch_dir = results_dir / f'epoch_{epoch}'
        summary = evaluate_without_changing_rng(model, dev_loader, dev, device, epoch_dir)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, row_loss=row_loss, pair_loss=pair_loss)
            history.append(record)
        pd.DataFrame(history).to_csv(
            results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
        )
        score = float(summary.loc[summary['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            # 只保存本组最佳权重，包含主干、原有分类头和新增词汇分支。
            save_best_weights(model, model_dir / 'best_model.pth')
            best_info = {'best_epoch': epoch, 'weighted_f1': score}
            (model_dir / 'best_metrics.json').write_text(
                json.dumps(best_info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')

    best_prediction = results_dir / f'epoch_{best_epoch}' / 'predictions_with_scores.csv'
    compare_with_original(best_prediction, results_dir / 'comparison_with_original')
    compare_with_lexical(best_prediction, results_dir / 'comparison_with_lexical')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', model_dir / 'best_model.pth')
    print('最佳预测：', best_prediction)
    print('内存中的模型仍是最后一轮；重新预测请运行 evaluate_best.py 加载最佳权重。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexical-only', action='store_true', help='只关闭新增位置特征，保留原 20 维 Lexical')
    args = parser.parse_args()
    main(use_position_features=cfg.USE_POSITION_FEATURES and not args.lexical_only)
