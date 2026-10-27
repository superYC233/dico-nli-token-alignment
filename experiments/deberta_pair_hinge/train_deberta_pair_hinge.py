"""右键运行：3290行，混合Row CE + 0.5×16组合hinge；可切换ce3对照。"""
import argparse
import hashlib
import json
import math
import sys
import time
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
from experiments.deberta_pair_hinge import config as cfg
from experiments.deberta_pair_hinge.losses import compute_joint_loss
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset, collate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_manual_aug.train_boundary_manual import set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import read_data, evaluate_phase


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(results_dir):
    # 复用已有读取函数：检查Train快照及Dev哈希与0.938那次一致。
    train, dev, reference = read_data()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'training_data.csv').write_bytes(cfg.TRAIN_PATH.read_bytes())
    (results_dir / 'reference_run_config.json').write_bytes(
        (cfg.REFERENCE_DIR / 'run_config.json').read_bytes(),
    )
    data_info = {
        'train_snapshot': str(cfg.TRAIN_PATH),
        'train_rows': len(train), 'dev_rows': len(dev),
        'class_counts': train.label.value_counts().to_dict(),
        'real_pairs': int(train.groupby('pair_id').size().eq(2).sum()),
        'singleton_rows': int(train.groupby('pair_id').size().eq(1).sum()),
        'train_sha256': file_hash(cfg.TRAIN_PATH),
        'dev_sha256': file_hash(cfg.DEV_PATH),
        'augmentation': reference['augmentation'],
        'new_augmentation': False,
    }
    (results_dir / 'data_config.json').write_text(
        json.dumps(data_info, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    return train, dev, data_info


def build_model(tokenizer, device, model_name=cfg.MODEL_NAME):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    # 0.938原架构：CLS分类 + Token Alignment；全参数更新，不创建Lexical分支。
    return DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    ).to(device)


def build_criteria(device):
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    return row_criterion, pair_criterion


def train_epoch(model, loader, optimizer, row_criterion, pair_criterion, device, pair_loss):
    model.train()
    total_row, total_pair = 0.0, 0.0
    label_counts = torch.zeros(4, dtype=torch.long)
    pair_count, batches_without_n = 0, 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc=f'混合 Row CE + {pair_loss}'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss, row_loss, pair_value = compute_joint_loss(
            logits, batch['labels'].to(device), batch['pair_indices'].to(device),
            batch['pair_targets'].to(device), row_criterion, pair_criterion,
            pair_loss=pair_loss, margin=cfg.HINGE_MARGIN, pair_weight=cfg.PAIR_WEIGHT,
        )
        # 同一次前向、同一个总损失、同一次反向和更新。
        loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        total_pair += pair_value.item()
        label_counts += torch.bincount(batch['labels'], minlength=4)
        pair_count += len(batch['pair_targets'])
        batches_without_n += int(not batch['labels'].eq(3).any())
    row_mean, pair_mean = total_row / len(loader), total_pair / len(loader)
    return {
        'row_loss': row_mean, 'pair_loss': pair_mean,
        'total_loss': row_mean + cfg.PAIR_WEIGHT * pair_mean,
        'pair_objective': pair_loss, 'train_seconds': time.perf_counter() - started,
        'row_ce_rows': int(label_counts.sum()), 'pair_groups': pair_count,
        'optimizer_steps': len(loader), 'batches_without_N': batches_without_n,
        **{f'seen_{label}': int(count) for label, count in zip(cfg.LABELS, label_counts)},
    }


def compare_predictions(prediction_path, reference_path, output_dir):
    old = pd.read_csv(reference_path, keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    old = old[['instance_id', 'prediction', 'pair_prediction']].rename(columns={
        'prediction': 'reference_independent', 'pair_prediction': 'reference_pair_aware',
    })
    merged = current.merge(old, on='instance_id', validate='one_to_one')
    assert len(merged) == len(current) == len(old)
    gold = merged.gold_label.map(dict(zip(cfg.FULL_LABELS, cfg.LABELS)))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before = merged[f'reference_{method}'].eq(gold)
        after = merged[column].eq(gold)
        fixed, broken = merged[~before & after], merged[before & ~after]
        fixed.to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        broken.to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
        records.append({'method': method, 'fixed': len(fixed), 'broken': len(broken),
                        'net_correct': len(fixed) - len(broken)})
    pd.DataFrame(records).to_csv(output_dir / 'changes.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(records).to_string(index=False))


def main(run_name=cfg.RUN_NAME, pair_loss=cfg.PAIR_LOSS, prepare_only=False):
    name = cfg.get_run_name(run_name, pair_loss)
    model_dir = cfg.MODEL_ROOT / name
    results_dir = cfg.HERE / 'outputs' / name
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换新名字，保留旧模型。')
    train, dev, data_info = prepare_data(results_dir)
    group_count = train.pair_id.nunique()
    print('训练数据：官方3042 + scope200 + 原48 =', len(train))
    print('模型：原0.938架构；NLI初始化重新训练；不加载旧最佳权重。')
    print('训练：四类组混合，每步Row CE +', cfg.PAIR_WEIGHT, '*', pair_loss)
    print('真实Pair / N单例：', data_info['real_pairs'], '/', data_info['singleton_rows'])
    print('每步最多行数 / 每轮步数：', cfg.GROUP_BATCH_SIZE * 2, '/', math.ceil(group_count / cfg.GROUP_BATCH_SIZE))
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    if prepare_only:
        print('仅检查及复制训练快照，没有加载模型或训练。')
        return

    set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    model = build_model(tokenizer, device)
    groups = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(
        groups, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED + 30),
    )
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, cfg.MAX_LENGTH), batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )
    row_criterion, pair_criterion = build_criteria(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    model_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    sources = [Path(__file__), cfg.HERE / 'config.py', cfg.HERE / 'losses.py',
               ROOT / 'experiments/deberta_lexical/model.py',
               ROOT / 'experiments/deberta_joint_pair/data.py',
               ROOT / 'experiments/deberta_lexical/evaluation.py',
               ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py']
    run_config = {
        'run_name': name, 'pair_objective': pair_loss, 'model_name': cfg.MODEL_NAME,
        'seed': cfg.SEED, 'epochs': cfg.EPOCHS, 'dtype': 'float32',
        'max_length': cfg.MAX_LENGTH, 'group_batch_size': cfg.GROUP_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'steps_per_epoch': len(loader),
        'learning_rate': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING if pair_loss == 'ce3' else None,
        'hinge_margin': cfg.HINGE_MARGIN if pair_loss == 'hinge16' else None,
        'hinge_negative_set': 'all 15 non-gold combinations, including N' if pair_loss == 'hinge16' else None,
        'pair_reduction': 'gold group weight times loss; divided by sum of gold group weights',
        'loss_reporting': 'mean of per-step losses; batches without pairs contribute zero pair loss',
        'selection': cfg.SELECT_BY, 'training_mode': 'mixed',
        'use_lexical_features': False, 'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data': data_info, 'row_exposure': 'every train row once per epoch, including N',
        'pair_exposure': 'every real pair once per epoch; no synthetic NN',
        'comparison': 'ce3 versus hinge16 uses identical data, model, batches, Row CE and optimizer steps; not a pure candidate-set ablation',
        'decoding': 'original FB/BF/EE on paired rows; independent argmax on singletons',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
    }
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )

    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第{epoch}/{cfg.EPOCHS}轮 =====')
        losses = train_epoch(model, loader, optimizer, row_criterion, pair_criterion, device, pair_loss)
        assert losses['row_ce_rows'] == len(train) and losses['pair_groups'] == groups.pair_count
        print(f"Row CE：{losses['row_loss']:.6f}；{pair_loss}：{losses['pair_loss']:.6f}")
        print('本轮F/B/E/N行数：', {label: losses[f'seen_{label}'] for label in cfg.LABELS})
        print('不含N的混合批次数：', losses['batches_without_N'])
        summary = evaluate_phase(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}', cfg.GOLD_PATH)
        independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        target_met = independent >= 0.87 and paired >= 0.94
        print(f'同一轮：independent={independent:.6f}，pair-aware={paired:.6f}；双目标达成：{target_met}')
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, both_targets_met=target_met, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method.eq(cfg.SELECT_BY), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            info = {
                'best_epoch': epoch, 'selection': cfg.SELECT_BY, 'weighted_f1': score,
                'independent_f1': independent, 'pair_aware_f1': paired,
                'both_targets_met': target_met, 'same_checkpoint_metrics': summary.to_dict('records'),
            }
            (model_dir / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第{epoch}轮，{cfg.SELECT_BY} F1={score:.6f}')

    predictions = results_dir / f'epoch_{best_epoch}/predictions_with_scores.csv'
    best_dir = results_dir / 'best'
    best_dir.mkdir(exist_ok=True)
    (best_dir / predictions.name).write_bytes(predictions.read_bytes())
    print('\n相对0.938版最佳权重的逐条变化：')
    reference_history = pd.read_csv(cfg.REFERENCE_DIR / 'epoch_summary.csv')
    reference_scores = reference_history[reference_history.method.eq('pair_aware')]
    reference_epoch = int(reference_scores.loc[reference_scores.weighted_f1.idxmax(), 'epoch'])
    reference_path = cfg.REFERENCE_DIR / f'epoch_{reference_epoch}/predictions_with_scores.csv'
    compare_predictions(predictions, reference_path, results_dir / 'comparison_with_0p938')
    other = 'ce3' if pair_loss == 'hinge16' else 'hinge16'
    other_path = cfg.HERE / 'outputs' / cfg.get_run_name(run_name, other) / 'best/predictions_with_scores.csv'
    if other_path.exists():
        print('相对另一个混合损失对照的逐条变化：')
        compare_predictions(predictions, other_path, results_dir / f'comparison_with_{other}')
    print(f'\n最佳第{best_epoch}轮；{cfg.SELECT_BY} F1={best_f1:.6f}')
    print('只保留一份最佳权重：', model_dir / 'best_model.pth')
    print('同一最佳权重的两项分数：', results_dir / 'best_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-loss', choices=['ce3', 'hinge16'], default=cfg.PAIR_LOSS)
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.pair_loss, args.prepare_only)
