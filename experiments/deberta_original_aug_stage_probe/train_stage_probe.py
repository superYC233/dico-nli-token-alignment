"""沿用原版 + 增强；只增加 Row CE 后与 Pair CE 后的 Dev 评估。"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# 本模块有 main 保护，导入不会启动训练或加载模型权重。
# 复用原版的模型、数据类、两段训练函数和评分函数，避免顺带改变训练方法。
from experiments.deberta_original_aug import train_original_aug as base
from experiments.deberta_original_aug.scope_augmentation import build_train_variant
from experiments.deberta_original_aug_stage_probe.analyze_stages import (
    compare_stages, summarize_run,
)


RUN_NAME = 'run01_scope'
AUGMENT_MODE = 'scope'  # 当前复查 0.935493 的原版 + 200 条增强。
SEED = 42

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_original_aug_stage_probe') / RUN_NAME
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
REFERENCE_DIR = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'

# 其他参数全部沿用 base：lr=2e-5、10轮、三个 batch 均16、float32。
# Row 权重 [1,1,1,1.5]，Pair 权重 [1,1,1.25]，Pair 系数0.5，平滑0.05。
# 不增加词汇分支、不冻结、不改变损失、不续训旧模型。


def prepare_data():
    if (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已有训练记录，请换新名字。')
    original = pd.read_csv(base.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(base.DEV_PATH, keep_default_na=False)
    train, extra, audit = build_train_variant(original, dev, AUGMENT_MODE, SEED)
    base.validate_groups(train)
    base.validate_groups(dev)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(RESULTS_DIR / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(RESULTS_DIR / 'augmentation_audit.csv', index=False, encoding='utf-8-sig')
    info = {
        'mode': AUGMENT_MODE, 'original_rows': len(original),
        'extra_rows': len(extra), 'train_rows': len(train), 'dev_rows': len(dev),
        'extra_class_counts': extra.label.value_counts().to_dict(),
        'effective_train_sha256': hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest(),
        'annotation_note': 'Same synthetic scope templates; no independent label adjudication.',
    }
    (RESULTS_DIR / 'data_config.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('数据方案：', AUGMENT_MODE)
    print('原 Train / 新增 / 构造后：', len(original), '/', len(extra), '/', len(train))
    print('新增类别数量：', info['extra_class_counts'])
    return train, dev, info


def evaluate_stage(model, loader, dev, device, output_dir):
    # 额外的评估不能消耗后面 shuffle / dropout 使用的随机数。
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    loader_state = loader.generator.get_state() if loader.generator is not None else None
    was_training = model.training
    try:
        return base.evaluate(model, loader, dev, device, output_dir)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if loader_state is not None:
            loader.generator.set_state(loader_state)
        model.train(was_training)


def make_run_config(train, dev, pairs, row_loader, pair_loader, data_info, model):
    return {
        'run_name': RUN_NAME, 'model_name': base.MODEL_NAME, 'seed': SEED,
        'epochs': base.EPOCHS, 'max_length': base.MAX_LENGTH, 'dtype': 'float32',
        'learning_rate': base.LEARNING_RATE, 'weight_decay': base.WEIGHT_DECAY,
        'row_batch_size': base.ROW_BATCH_SIZE, 'pair_batch_size': base.PAIR_BATCH_SIZE,
        'dev_batch_size': base.DEV_BATCH_SIZE, 'row_weights': base.ROW_WEIGHTS,
        'pair_weights': base.PAIR_WEIGHTS, 'pair_weight': base.PAIR_WEIGHT,
        'pair_label_smoothing': base.PAIR_LABEL_SMOOTHING,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'use_lexical_features': False, 'augmentation': data_info,
        'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs': len(pairs),
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'selection_stage': 'after_pair',
        'after_row_is_diagnostic_only': True,
        'train_sha256': hashlib.sha256(base.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(base.DEV_PATH.read_bytes()).hexdigest(),
        'reference_dir': str(REFERENCE_DIR),
        'source_sha256': {
            'probe_script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_training_script': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        },
    }


def main():
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已使用，请换新名字，保留旧模型和结果。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练请使用你的 CUDA 环境。')
    train, dev, data_info = prepare_data()
    base.set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        base.MODEL_NAME, cache_dir=base.get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        base.MODEL_NAME, cache_dir=base.get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    rows = base.RowDataset(train, tokenizer, base.MAX_LENGTH)
    pairs = base.RealPairDataset(rows)
    # 顺序和原版一致；训练 DataLoader 不另换 generator。
    row_loader = DataLoader(rows, batch_size=base.ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=base.PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(
        base.RowDataset(dev, tokenizer, base.MAX_LENGTH),
        batch_size=base.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    model = base.DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    ).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(base.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(base.PAIR_WEIGHTS, device=device),
        label_smoothing=base.PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    run_config = make_run_config(train, dev, pairs, row_loader, pair_loader, data_info, model)
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )

    print('方案：原版 + 范围增强，仅增加两个阶段的评估')
    print('训练 / Dev / 真实 Pair：', len(train), '/', len(dev), '/', len(pairs))
    print('每轮 Row / Pair 步数：', len(row_loader), '/', len(pair_loader))
    print('仍只按 after_pair 的 pair-aware F1 保存一份最佳权重')
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)

    history, phase_effects = [], []
    previous_dir, previous_summary = None, None
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, base.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{base.EPOCHS} 轮 =====')
        epoch_dir = RESULTS_DIR / f'epoch_{epoch}'
        row_dir, pair_dir = epoch_dir / 'after_row', epoch_dir / 'after_pair'

        row_loss = base.train_rows(model, row_loader, optimizer, row_criterion, device)
        print(f'\n===== 第 {epoch} 轮：Row CE 后 =====')
        row_summary = evaluate_stage(model, dev_loader, dev, device, row_dir)
        for record in row_summary.to_dict('records'):
            record.update(epoch=epoch, stage='after_row', row_loss=row_loss, pair_loss=None)
            history.append(record)
        pd.DataFrame(history).to_csv(RESULTS_DIR / 'stage_summary.csv', index=False, encoding='utf-8-sig')

        if previous_dir is not None:
            phase_effects.extend(compare_stages(
                previous_dir / 'predictions_with_scores.csv', row_dir / 'predictions_with_scores.csv',
                previous_summary, row_summary, epoch_dir / 'row_changes', epoch, 'row_ce',
            ))

        pair_loss = base.train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
        print(f'\n===== 第 {epoch} 轮：Pair CE 后 =====')
        pair_summary = evaluate_stage(model, dev_loader, dev, device, pair_dir)
        for record in pair_summary.to_dict('records'):
            record.update(epoch=epoch, stage='after_pair', row_loss=row_loss, pair_loss=pair_loss)
            history.append(record)
        phase_effects.extend(compare_stages(
            row_dir / 'predictions_with_scores.csv', pair_dir / 'predictions_with_scores.csv',
            row_summary, pair_summary, epoch_dir / 'pair_changes', epoch, 'pair_ce',
        ))
        pd.DataFrame(history).to_csv(RESULTS_DIR / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(phase_effects).to_csv(RESULTS_DIR / 'phase_effects.csv', index=False, encoding='utf-8-sig')
        for effect in phase_effects[-2:]:
            print(f'{effect["method"]}：Pair CE 后 F1 变化 {effect["delta_f1_pp"]:+.2f} 点；'
                  f'修好 {effect["fixed"]} 条，破坏 {effect["broken"]} 条；'
                  f'非N→N {effect["nonN_to_N_before"]} → {effect["nonN_to_N_after"]}')

        # after_row 只观测，不参加旧协议的最佳模型竞争。
        score = float(pair_summary.set_index('method').loc['pair_aware', 'weighted_f1'])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            base.save_weights(model, MODEL_DIR / 'best_model.pth')
            pair_summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            info = {'best_epoch': epoch, 'stage': 'after_pair', 'selection_method': 'pair_aware',
                    'weighted_f1': score}
            (MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            print(f'刷新最佳：第 {epoch} 轮 after_pair，pair-aware F1={score:.6f}')
        previous_dir, previous_summary = pair_dir, pair_summary

    summarize_run(RESULTS_DIR, REFERENCE_DIR)
    print(f'\n最佳第 {best_epoch} 轮 after_pair，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', MODEL_DIR / 'best_model.pth')
    print('先看：', RESULTS_DIR / 'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='只准备数据，不加载模型、不训练')
    args = parser.parse_args()
    if args.prepare_only:
        train, dev, _ = prepare_data()
        pair_count = int((train.groupby('pair_id').size() == 2).sum())
        print('每轮 Row / Pair 预计步数：', math.ceil(len(train) / base.ROW_BATCH_SIZE),
              '/', math.ceil(pair_count / base.PAIR_BATCH_SIZE))
        print('数据检查结束，未加载模型、未启动训练。')
    else:
        main()
