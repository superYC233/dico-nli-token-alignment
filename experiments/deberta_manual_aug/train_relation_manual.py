"""两个入口共用的训练代码：新48行对照，混合或先Row后Pair。"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset, collate_groups
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.prepare_relation_data import prepare, PREPARED_DIR, SOURCE_PATH
from experiments.deberta_manual_aug.train_boundary_manual import (
    RealPairDataset, set_seed, evaluate, save_weights, compare_with_reference,
)


# 两个入口均使用下面的超参数；只改变训练安排。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
GROUP_BATCH_SIZE = 8  # 混合版：16组，最多32行；一对的两行不能拆开。
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05
LABELS = ['F', 'B', 'E', 'N']

HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'


def prepare_data(mode, results_dir):
    combined, manual, report = prepare()
    baseline = combined.iloc[:report['baseline_rows']].copy()
    if mode == 'manual':
        train, extra = combined, manual
    elif mode == 'copy':
        train = pd.read_csv(PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
        extra = pd.read_csv(PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    elif mode == 'baseline':
        train, extra = baseline, baseline.iloc[:0].copy()
    else:
        raise ValueError('AUGMENT_MODE只能为manual、copy或baseline。')
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    validate_groups(train)
    validate_groups(dev)
    results_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(results_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(results_dir / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    (results_dir / 'relation_examples_source.csv').write_bytes(SOURCE_PATH.read_bytes())
    for filename in ['manual_label_notes.csv', 'train_word_sources.csv',
                     'copy_control_sources.csv', 'overlap_report.json']:
        (results_dir / filename).write_bytes((PREPARED_DIR / filename).read_bytes())
    info = {
        'mode': mode, 'original_rows': report['original_rows'],
        'baseline_rows': len(baseline), 'old_scope_rows': report['old_scope_rows'],
        'extra_rows': len(extra), 'train_rows': len(train), 'dev_rows': len(dev),
        'extra_class_counts': extra.label.value_counts().to_dict(),
        'effective_train_sha256': hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest(),
        'baseline_snapshot_canonical_sha256': report['baseline_snapshot_canonical_sha256'],
        'relation_design': report['relation_design'],
        'new_manual_vs_dev': report['new_manual_vs_dev'],
        'audit_scope': 'Vocabulary isolation is for NEW MANUAL rows; old scope and copied Train are separately disclosed.',
    }
    (results_dir / 'data_config.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('数据方案：', mode)
    print('原scope基准 / 本次新增 / 构造后：', len(baseline), '/', len(extra), '/', len(train))
    return train, dev, info


def compute_joint_loss(logits, labels, pair_indices, pair_targets, row_criterion, pair_criterion):
    # 所有行均做F/B/E/N四分类。N单例虽不参与Pair CE，仍直接参与同一步的Row CE。
    row_loss = row_criterion(logits, labels)
    if len(pair_targets) > 0:
        first = logits[pair_indices[:, 0]]
        second = logits[pair_indices[:, 1]]
        pair_loss = pair_criterion(get_pair_logits(first, second), pair_targets)
    else:
        pair_loss = logits.new_zeros(())
    return row_loss + PAIR_WEIGHT * pair_loss, row_loss, pair_loss


def train_mixed(model, loader, optimizer, row_criterion, pair_criterion, device):
    model.train()
    total_row, total_pair = 0.0, 0.0
    label_counts = torch.zeros(4, dtype=torch.long)
    pair_count, batches_without_n = 0, 0
    for batch in tqdm(loader, desc='四类混合 Row + Pair'):
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss, row_loss, pair_loss = compute_joint_loss(
            logits, batch['labels'].to(device), batch['pair_indices'].to(device),
            batch['pair_targets'].to(device), row_criterion, pair_criterion,
        )
        loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        total_pair += pair_loss.item()
        label_counts += torch.bincount(batch['labels'], minlength=4)
        pair_count += len(batch['pair_targets'])
        batches_without_n += int(not batch['labels'].eq(3).any())
    return {'row_loss': total_row / len(loader), 'pair_loss': total_pair / len(loader),
            'row_ce_rows': int(label_counts.sum()), 'pair_ce_groups': pair_count,
            'optimizer_steps': len(loader), 'batches_without_N': batches_without_n,
            'forward_rows': int(label_counts.sum()),
            **{f'seen_{label}': int(count) for label, count in zip(LABELS, label_counts)}}


def train_staged(model, row_loader, pair_loader, optimizer, row_criterion, pair_criterion, device):
    model.train()
    total_row, total_pair = 0.0, 0.0
    label_counts = torch.zeros(4, dtype=torch.long)
    # 第一阶段包含全部N单例，四类行整体随机打乱。
    for batch in tqdm(row_loader, desc='四分类 Row CE'):
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        row_loss = row_criterion(logits, batch['label'].to(device))
        row_loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        label_counts += torch.bincount(batch['label'], minlength=4)
    # 第二阶段沿用原版Pair CE：只有FB/BF/EE，不含N，也不含四分类损失。
    pair_count = 0
    for ids0, mask0, ids1, mask1, targets in tqdm(pair_loader, desc='真实 Pair CE'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = pair_criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total_pair += pair_loss.item()
        pair_count += len(targets)
    return {'row_loss': total_row / len(row_loader), 'pair_loss': total_pair / len(pair_loader),
            'row_ce_rows': int(label_counts.sum()), 'pair_ce_groups': pair_count,
            'optimizer_steps': len(row_loader) + len(pair_loader),
            'pair_only_steps_without_N': len(pair_loader),
            'forward_rows': int(label_counts.sum()) + 2 * pair_count,
            **{f'seen_{label}': int(count) for label, count in zip(LABELS, label_counts)}}


def main(run_name, augment_mode='manual', training_mode='staged', prepare_only=False):
    assert training_mode in ['staged', 'mixed']
    model_dir = MODEL_ROOT / run_name
    results_dir = HERE / 'outputs' / run_name
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有模型或训练记录，请换新名字。')
    train, dev, data_info = prepare_data(augment_mode, results_dir)
    pair_count = int(train.groupby('pair_id').size().eq(2).sum())
    group_count = train.pair_id.nunique()
    row_steps = math.ceil(len(train) / ROW_BATCH_SIZE)
    pair_steps = math.ceil(pair_count / PAIR_BATCH_SIZE)
    mixed_steps = math.ceil(group_count / GROUP_BATCH_SIZE)
    print('训练安排：', training_mode)
    print('Row行数 / Pair组数 / 全部组数：', len(train), '/', pair_count, '/', group_count)
    print('两阶段每轮步数：', row_steps, '+', pair_steps, '=', row_steps + pair_steps)
    print('混合每轮步数：', mixed_steps, '；每步最多', GROUP_BATCH_SIZE * 2, '行')
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    if prepare_only:
        print('只准备数据；没有加载模型或训练。')
        return

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    groups = GroupDataset(train, tokenizer, MAX_LENGTH)
    rows = groups.rows
    pairs = RealPairDataset(rows)
    row_loader = DataLoader(
        rows, batch_size=ROW_BATCH_SIZE, shuffle=True,
    )
    pair_loader = DataLoader(
        pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True,
    )
    mixed_loader = DataLoader(
        groups, batch_size=GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(SEED + 30),
    )
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    # 两版同样从NLI初始化，全参数、不冻结、不加词汇分支。
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    order = ('all Row CE batches, then all Pair CE batches' if training_mode == 'staged'
             else 'shuffled pair/singleton groups; one forward; Row CE + 0.5 * Pair CE')
    source_names = ['train_relation_manual.py', f'train_relation_{training_mode}.py',
                    'prepare_relation_data.py', 'relation_examples.csv',
                    'prepare_manual_data.py', 'train_boundary_manual.py']
    config = {
        'run_name': run_name, 'training_mode': training_mode, 'model_name': MODEL_NAME, 'seed': SEED,
        'epochs': EPOCHS, 'max_length': MAX_LENGTH, 'dtype': 'float32',
        'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'group_batch_size': GROUP_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
        'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0, 'use_lexical_features': False,
        'augmentation': data_info, 'real_pairs': len(pairs), 'train_groups': len(groups),
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
        'optimizer_steps_per_epoch': (len(row_loader) + len(pair_loader)
                                      if training_mode == 'staged' else len(mixed_loader)),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_order': order, 'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'row_ce_exposure': 'each train row once per epoch, including all N singletons',
        'pair_ce_exposure': 'each real pair once per epoch; no N term in three-combination Pair CE',
        'comparison_note': 'Mixed vs staged changes batching, update count and forward exposure as well as loss schedule; not a pure shuffle ablation.',
        'shuffle': 'Row and Pair loaders shuffle rows/groups every epoch. Mixed loader shuffles whole groups, preserving the two paired rows.',
        'shuffle_generators': {'row': 'global torch RNG as historical staged recipe',
                               'pair': 'global torch RNG as historical staged recipe',
                               'mixed': SEED + 30, 'dev': SEED + 1},
        'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names},
        'train_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(DEV_PATH.read_bytes()).hexdigest(),
    }
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('同一数据、同一模型、同一损失权重；比较两种训练安排。')
    print('选模：每轮Dev pair-aware F1；同时记录独立预测；只保存一个最佳模型。')
    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        if training_mode == 'mixed':
            losses = train_mixed(model, mixed_loader, optimizer, row_criterion, pair_criterion, device)
        else:
            losses = train_staged(model, row_loader, pair_loader, optimizer, row_criterion, pair_criterion, device)
        assert losses['row_ce_rows'] == len(rows) and losses['pair_ce_groups'] == len(pairs)
        print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
        print('本轮四类Row CE行数：', {label: losses[f'seen_{label}'] for label in LABELS})
        print('优化步数 / 前向总行数：', losses['optimizer_steps'], '/', losses['forward_rows'])
        if training_mode == 'mixed':
            print('本轮不含N的混合批次数：', losses['batches_without_N'])
        summary = evaluate(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}')
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            info = {'best_epoch': epoch, 'selection_method': 'pair_aware', 'weighted_f1': score,
                    'training_mode': training_mode}
            (model_dir / 'best_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')
    best_predictions = results_dir / f'epoch_{best_epoch}/predictions_with_scores.csv'
    compare_with_reference(best_predictions, results_dir / 'comparison_with_0p935')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', model_dir / 'best_model.pth')
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    from experiments.deberta_minimal_contrasts.evaluate_original import evaluate_probes
    evaluate_probes(model, tokenizer, device, output_dir=results_dir / 'preserved_diagnostics')

