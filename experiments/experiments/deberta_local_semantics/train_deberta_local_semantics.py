"""右键运行：原0.938配方 + Row阶段局部关系辅助监督。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed, save_weights
from experiments.deberta_local_semantics import config as cfg
from experiments.deberta_local_semantics.prepare_data import prepare, file_hash
from experiments.deberta_local_semantics.data import LocalRowDataset, diagnostic_frame
from experiments.deberta_local_semantics.model import build_model
from experiments.deberta_local_semantics.evaluation import evaluate, evaluate_local, compare_with_reference


def row_objective(model, batch, row_criterion, local_criterion, device, local_weight):
    targets = batch['local_target']
    active = targets.ge(0)
    has_local = local_weight > 0 and bool(active.any())
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    if has_local:
        logits, tokens = model(input_ids, attention_mask, return_tokens=True)
    else:
        logits = model(input_ids, attention_mask)
    row_loss = row_criterion(logits, batch['label'].to(device))
    local_loss = row_loss.new_zeros(())
    local_count, local_correct = 0, 0
    if has_local:
        device_active = active.to(device)
        local_logits = model.local_logits(
            tokens[device_active], batch['span1_mask'][active].to(device),
            batch['span2_mask'][active].to(device),
        )
        local_targets = targets[active].to(device)
        local_loss = local_criterion(local_logits, local_targets)
        local_count = int(active.sum())
        local_correct = int(local_logits.argmax(dim=1).eq(local_targets).sum())
    return row_loss + local_weight * local_loss, row_loss, local_loss, local_count, local_correct


def train_rows(model, loader, optimizer, row_criterion, local_criterion, device, local_weight):
    model.train()
    total_row, total_local = 0.0, 0.0
    local_count, local_correct, local_batches = 0, 0, 0
    for batch in tqdm(loader, desc='Row CE + 局部关系监督'):
        optimizer.zero_grad()
        loss, row_loss, local_loss, count, correct = row_objective(
            model, batch, row_criterion, local_criterion, device, local_weight,
        )
        loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        total_local += local_loss.item() * count
        local_count += count
        local_correct += correct
        local_batches += int(count > 0)
    return {
        'row_loss': total_row / len(loader),
        'local_loss': total_local / local_count if local_count else 0.0,
        'local_train_accuracy': local_correct / local_count if local_count else 0.0,
        'local_supervised_rows': local_count, 'local_batches': local_batches,
    }


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total_pair = 0.0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='原版三组合 Pair CE'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device))
        second = model(ids1.to(device), mask1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (cfg.PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total_pair += pair_loss.item()
    return total_pair / len(loader)


def build_optimizer(model):
    local_parameters = list(model.local_relation_head.parameters())
    local_ids = {id(parameter) for parameter in local_parameters}
    original_parameters = [parameter for parameter in model.parameters() if id(parameter) not in local_ids]
    return torch.optim.AdamW([
        {'params': original_parameters, 'lr': cfg.LEARNING_RATE},
        {'params': local_parameters, 'lr': cfg.LOCAL_LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)


def main(run_name=cfg.RUN_NAME, local_weight=cfg.LOCAL_WEIGHT, prepare_only=False):
    model_dir, results_dir = cfg.get_output_dirs(run_name)
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('RUN_NAME已有训练记录，请换一个名字。')
    train, annotations, diagnostic, data_report = prepare()
    if prepare_only:
        print('数据准备完成；没有加载模型或训练。')
        return

    set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True, use_fast=True,
    )
    rows = LocalRowDataset(train, tokenizer, annotations, cfg.MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    row_loader = DataLoader(rows, batch_size=cfg.ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, cfg.MAX_LENGTH), batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )
    local_diagnostic = LocalRowDataset(diagnostic_frame(diagnostic), tokenizer, diagnostic, cfg.MAX_LENGTH)
    diagnostic_loader = DataLoader(
        local_diagnostic, batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 2),
    )
    model = build_model(tokenizer).to(device)
    optimizer = build_optimizer(model)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    local_criterion = nn.CrossEntropyLoss()
    assert len(rows) == 3290 and len(pairs) == 1380
    print('结构：原CLS + Token Alignment；局部头仅用于Row辅助监督。')
    print('Train / Dev：', len(train), '/', len(dev))
    print('局部监督行数 / 权重：', len(annotations), '/', local_weight)
    print('原模型 / 新局部头学习率：', cfg.LEARNING_RATE, '/', cfg.LOCAL_LEARNING_RATE)
    print('每轮Row / Pair步数：', len(row_loader), '/', len(pair_loader))
    print('可训练参数：', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)

    model_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    settings = {
        'run_name': run_name, 'model_name': cfg.MODEL_NAME,
        'base_revision': getattr(model.deberta_model.config, '_commit_hash', None),
        'seed': cfg.SEED, 'epochs': cfg.EPOCHS, 'max_length': cfg.MAX_LENGTH,
        'learning_rate': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_batch_size': cfg.ROW_BATCH_SIZE, 'pair_batch_size': cfg.PAIR_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'row_weights': cfg.ROW_WEIGHTS,
        'pair_weights': cfg.PAIR_WEIGHTS, 'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'local_weight': local_weight, 'local_hidden_size': cfg.LOCAL_HIDDEN_SIZE,
        'local_learning_rate': cfg.LOCAL_LEARNING_RATE,
        'dtype': 'float32', 'use_lexical_features': False, 'use_lora': False,
        'training_order': 'all Row CE + annotated-span local CE, then original Pair-only CE',
        'local_inference': 'Auxiliary head is not used for official predictions.',
        'selection': 'best after-Pair Dev pair-aware weighted F1; earliest on ties',
        'data': data_report,
        'source_sha256': {str(path.relative_to(cfg.ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), cfg.HERE / 'config.py', cfg.HERE / 'model.py',
            cfg.HERE / 'data.py', cfg.HERE / 'prepare_data.py', cfg.HERE / 'evaluation.py',
            cfg.ANNOTATION_PATH,
        ]},
    }
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
        annotations.to_csv(folder / 'local_train_annotations.csv', index=False, encoding='utf-8-sig')
        diagnostic.to_csv(folder / 'local_diagnostic_annotations.csv', index=False, encoding='utf-8-sig')
    (results_dir / 'training_data.csv').write_bytes(cfg.BASE_TRAIN_PATH.read_bytes())

    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_rows(model, row_loader, optimizer, row_criterion, local_criterion, device, local_weight)
        expected_local = len(annotations) if local_weight > 0 else 0
        assert losses['local_supervised_rows'] == expected_local
        epoch_dir = results_dir / f'epoch_{epoch}'
        print('\n===== Row阶段后 =====')
        row_summary = evaluate(model, dev_loader, dev, device, epoch_dir / 'after_row')
        row_local = evaluate_local(model, diagnostic_loader, diagnostic, device, epoch_dir / 'after_row/local_diagnostic')
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f"Row CE：{losses['row_loss']:.6f}；局部CE：{losses['local_loss']:.6f}；Pair CE：{pair_loss:.6f}")
        print('本轮局部监督曝光：', losses['local_supervised_rows'], '行；不额外复制或抽样。')
        print('\n===== Pair阶段后 =====')
        pair_summary = evaluate(model, dev_loader, dev, device, epoch_dir)
        pair_local = evaluate_local(model, diagnostic_loader, diagnostic, device, epoch_dir / 'local_diagnostic')
        for stage, summary, local_metrics in [
            ('after_row', row_summary, row_local), ('after_pair', pair_summary, pair_local),
        ]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, stage=stage, pair_loss=pair_loss, **losses,
                              local_diagnostic_accuracy=local_metrics['accuracy'],
                              local_diagnostic_ce=local_metrics['mean_ce'])
                history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            pair_summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            best_metrics = {'best_epoch': epoch, 'weighted_f1': score, 'local_weight': local_weight}
            (model_dir / 'best_metrics.json').write_text(json.dumps(best_metrics, indent=2), encoding='utf-8')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')
    compare_with_reference(results_dir / f'epoch_{best_epoch}/predictions_with_scores.csv',
                           results_dir / 'comparison_with_0p938')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('唯一最佳模型：', model_dir / 'best_model.pth')
    print('每轮独立/结构预测和Row/Pair局部诊断：', results_dir / 'epoch_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    parser.add_argument('--local-weight', type=float, default=cfg.LOCAL_WEIGHT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.local_weight, args.prepare_only)
