import argparse
import hashlib
import json
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
from experiments.deberta_abcde import config as cfg
from experiments.deberta_abcde.data_variants import (
    build_train_dataset, validate_groups, RowDataset, PairDataset,
)
from experiments.deberta_abcde.model import DebertaDiCoModel
from experiments.deberta_abcde.evaluation import get_pair_logits, predict_logits, score_logits


# 直接运行本文件只训练这一组；一次跑 A–E 请运行 run_all.py。
MODE = 'C'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(mode):
    original = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    train, n_audit, extra_audit = build_train_dataset(original, mode)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(dev)
    print(f'当前版本：{mode}')
    print(f'原始 Train：{len(original)}；构造后：{len(train)}；Dev：{len(dev)}')
    print('N 筛选：', n_audit['selection'].value_counts().to_dict())
    print('新增行操作：', extra_audit['operation'].value_counts().to_dict())
    return train, dev, n_audit, extra_audit


def train_rows(model, loader, optimizer, criterion, device):
    total = 0.0
    for input_ids, attention_mask, labels in tqdm(loader, desc='四分类训练'):
        logits = model(input_ids.to(device), attention_mask.to(device))
        loss = criterion(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise ValueError('四分类损失出现 NaN 或无穷。')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    total = 0.0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='Real Pair CE 训练'):
        # 沿用原版：两次独立前向，不改成一次拼接大 batch。
        first = model(ids0.to(device), mask0.to(device))
        second = model(ids1.to(device), mask1.to(device))
        pair_logits = get_pair_logits(first, second)
        pair_loss = criterion(pair_logits, targets.to(device))
        if not torch.isfinite(pair_loss):
            raise ValueError('Pair 损失出现 NaN 或无穷。')
        loss = cfg.PAIR_WEIGHT * pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def evaluate_without_changing_rng(model, loader, dev, device, output_dir):
    # 原版是训完 10 轮再预测。现在逐轮预测，但不能让评估消耗的随机数
    # 改变下一轮的 shuffle / dropout，所以评估前后保存、恢复随机状态。
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


def main(mode):
    model_dir, results_dir = cfg.get_output_dirs(mode)
    # 新实验必须使用新目录，防止半途重跑混进上一次的 checkpoint / 曲线。
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError(
            f'本组输出目录已存在：{model_dir} 或 {results_dir}。'
            '请在 config.py 修改 RUN_NAME，或只运行尚未训练的组。'
        )
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用 CUDA。本入口用于正式 GPU 训练。')
    device = torch.device('cuda')
    train, dev, n_audit, extra_audit = prepare_data(mode)

    # 每组重新加载同一个 NLI 预训练主干，不加载旧 epoch_7，也不接着上一组训练。
    set_seed(cfg.SEED)
    cache_dir = get_model_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=cache_dir, local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=cache_dir, local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )

    row_loader = DataLoader(
        RowDataset(train, tokenizer, cfg.MAX_LENGTH),
        batch_size=cfg.ROW_BATCH_SIZE, shuffle=True,
    )
    pair_loader = DataLoader(
        PairDataset(train, tokenizer, cfg.MAX_LENGTH),
        batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True,
    )
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, cfg.MAX_LENGTH),
        batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
    )
    model = DebertaDiCoModel(backbone, tokenizer).to(device)

    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float32, device=device),
    )
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float32, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )

    print('四分类行数：', len(row_loader.dataset))
    print('Real Pair 组数：', len(pair_loader.dataset))
    print('每轮 Row / Pair 更新次数：', len(row_loader), '/', len(pair_loader))
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    model_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    train.to_csv(results_dir / 'train.csv', index=False, encoding='utf-8-sig')
    n_audit.to_csv(results_dir / 'n_source_audit.csv', index=False, encoding='utf-8-sig')
    extra_audit.to_csv(results_dir / 'extra_audit.csv', index=False, encoding='utf-8-sig')
    tokenizer.save_pretrained(model_dir / 'tokenizer')

    run_config = {
        'mode': mode, 'seed': cfg.SEED, 'model_name': cfg.MODEL_NAME,
        'epochs': cfg.EPOCHS, 'train_rows': len(train), 'dev_rows': len(dev),
        'real_pairs': len(pair_loader.dataset), 'max_length': cfg.MAX_LENGTH,
        'row_batch_size': cfg.ROW_BATCH_SIZE, 'pair_batch_size': cfg.PAIR_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'learning_rate': cfg.LEARNING_RATE,
        'weight_decay': cfg.WEIGHT_DECAY, 'row_weights': cfg.ROW_WEIGHTS,
        'pair_weights': cfg.PAIR_WEIGHTS, 'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'dtype': 'float32', 'labels': cfg.FULL_LABELS,
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
    }
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2)
    (model_dir / 'run_config.json').write_text(config_text, encoding='utf-8')
    (results_dir / 'run_config.json').write_text(config_text, encoding='utf-8')

    best_f1 = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== {mode}：第 {epoch}/{cfg.EPOCHS} 轮 =====')
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
        pair_f1 = float(summary.loc[summary['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
        if pair_f1 > best_f1:
            best_f1 = pair_f1
            best_epoch = epoch
            # 只覆盖本组的新文件；不删除其他目录，也不保存每轮大权重。
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            best_info = {'mode': mode, 'seed': cfg.SEED, 'best_epoch': epoch, 'weighted_f1': best_f1}
            (model_dir / 'best_metrics.json').write_text(
                json.dumps(best_info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={best_f1:.6f}')

    print(f'\n{mode} 完成：最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', model_dir / 'best_model.pth')
    print('最佳预测：', results_dir / f'epoch_{best_epoch}' / 'predictions_with_scores.csv')
    print('完成。注意：内存中的模型仍是最后一轮，重新预测请加载 best_model.pth。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list('ABCDE'), default=MODE)
    args = parser.parse_args()
    main(args.mode)
