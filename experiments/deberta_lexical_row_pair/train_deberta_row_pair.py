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
from experiments.deberta_lexical_row_pair import config as cfg
from experiments.deberta_lexical_row_pair.data import (
    read_data, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical_row_pair.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_row_pair.model import DebertaDiCoModel
from experiments.deberta_lexical_row_pair.evaluation import (
    get_pair_logits, predict_logits, score_logits, compare_with_original,
)
from experiments.deberta_lexical_row_pair.analyze_stages import (
    compare_stages, print_stage_table, compare_with_reference,
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
        if name.startswith('lexical_output.'):
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


def compute_pair_stage_loss(first, second, labels0, labels1, targets,
                            row_criterion, pair_criterion):
    # 合并的是已有的两次前向结果，不需要额外运行模型。
    # 四分类 CE 在 F/B/E/N 四列上计算，N 分数过高也会受到惩罚。
    row_logits = torch.cat([first, second], dim=0)
    row_labels = torch.cat([labels0, labels1], dim=0)
    row_loss = row_criterion(row_logits, row_labels)

    pair_logits = get_pair_logits(first, second)
    pair_loss = pair_criterion(pair_logits, targets)
    loss = row_loss + cfg.PAIR_WEIGHT * pair_loss
    return loss, row_loss, pair_loss


def train_pairs(model, loader, optimizer, row_criterion, pair_criterion, device):
    totals = {'pair_stage_row_loss': 0.0, 'pair_loss': 0.0, 'pair_stage_total_loss': 0.0}
    for batch in tqdm(loader, desc='两行四分类 CE + Real Pair CE 训练'):
        ids0, mask0, features0, ids1, mask1, features1, targets, labels0, labels1 = batch
        # 两个方向分别前向，分别输入对应方向的词汇特征。
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        loss, row_loss, pair_loss = compute_pair_stage_loss(
            first, second, labels0.to(device), labels1.to(device), targets.to(device),
            row_criterion, pair_criterion,
        )
        if not torch.isfinite(loss):
            raise ValueError('联合损失出现 NaN 或无穷。')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals['pair_stage_row_loss'] += row_loss.item()
        totals['pair_loss'] += pair_loss.item()
        totals['pair_stage_total_loss'] += loss.item()
    return {name: value / len(loader) for name, value in totals.items()}


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


def make_run_config(use_lexical_features, train_rows_count, dev_rows_count, real_pairs):
    feature_file = Path(__file__).resolve().parent / 'lexical_features.py'
    return {
        'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED,
        'use_lexical_features': use_lexical_features,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'feature_source_sha256': hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'train_rows': train_rows_count, 'dev_rows': dev_rows_count,
        'real_pairs': real_pairs, 'data_augmentation': False,
        'epochs': cfg.EPOCHS, 'row_batch_size': cfg.ROW_BATCH_SIZE,
        'pair_batch_size': cfg.PAIR_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'max_length': cfg.MAX_LENGTH, 'learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE if use_lexical_features else None,
        'weight_decay': cfg.WEIGHT_DECAY, 'row_weights': cfg.ROW_WEIGHTS,
        'pair_weights': cfg.PAIR_WEIGHTS, 'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'dtype': 'float32', 'labels': cfg.FULL_LABELS,
        'training_order': 'all Row CE batches, then all paired Row CE + Real Pair CE batches',
        'pair_stage_objective': 'RowCE(concat(first, second)) + pair_weight * PairCE',
        'pair_stage_row_weight': 1.0,
        'pair_stage_rows': 'the two gold-labeled rows of each real training pair; no singleton N',
        'loss_reporting': 'mean of per-batch criterion values; pair_loss is before multiplying by pair_weight',
        'source_experiment': 'deberta_lexical_stage_probe/run01_rr/lexical_seed_42',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'diagnostic_stages': ['after_row', 'after_pair'],
        'checkpoint_stage': 'after_pair',
        'row_stage_is_diagnostic_only': True,
    }


def main(use_lexical_features=cfg.USE_LEXICAL_FEATURES):
    model_dir, results_dir = cfg.get_output_dirs(use_lexical_features)
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError('本次输出目录已存在，请在 config.py 修改 RUN_NAME，避免覆盖旧结果。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA；本入口用于正式 GPU 训练。')
    device = torch.device('cuda')
    train, dev = read_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    print('词汇特征分支：', use_lexical_features)
    print('Train / Dev：', len(train), '/', len(dev))
    print('词汇特征维度：', len(FEATURE_NAMES))

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
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=use_lexical_features,
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
    print('第二阶段损失：两行四分类 CE +', cfg.PAIR_WEIGHT, '* Pair CE')
    print('原模型学习率：', optimizer.param_groups[0]['lr'])
    if use_lexical_features:
        print('新增分支参数量：', sum(p.numel() for p in model.lexical_output.parameters()))
        print('词汇分支学习率：', optimizer.param_groups[1]['lr'])
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    model_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    save_feature_table(train, train_features, results_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, results_dir / 'dev_features.csv')
    run_config = make_run_config(use_lexical_features, len(train), len(dev), len(pair_loader.dataset))
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2)
    (model_dir / 'run_config.json').write_text(config_text, encoding='utf-8')
    (results_dir / 'run_config.json').write_text(config_text, encoding='utf-8')

    best_f1 = -1.0
    best_epoch = None
    history = []
    stage_history = []
    phase_history = []
    previous_pair_dir = None
    previous_pair_summary = None
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        model.train()
        epoch_dir = results_dir / f'epoch_{epoch}'
        row_dir = epoch_dir / 'after_row'
        pair_dir = epoch_dir / 'after_pair'

        # 第一阶段：沿用原版 Row CE。只增加评估，不用此时的分数选模。
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        print(f'\n===== 第 {epoch} 轮：Row CE 后 =====')
        row_summary = evaluate_without_changing_rng(model, dev_loader, dev, device, row_dir)
        for record in row_summary.to_dict('records'):
            record.update(epoch=epoch, stage='after_row', row_loss=row_loss,
                          pair_stage_row_loss=None, pair_loss=None, pair_stage_total_loss=None)
            stage_history.append(record)
        pd.DataFrame(stage_history).to_csv(
            results_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig',
        )

        # 第 2 轮起，上一轮 after_pair -> 本轮 after_row 就是本轮 Row CE 的变化。
        # 第 1 轮没有训练前评估，因此不虚构第 1 轮 Row CE 的改动统计。
        if previous_pair_dir is not None:
            phase_history.extend(compare_stages(
                previous_pair_dir / 'predictions_with_scores.csv',
                row_dir / 'predictions_with_scores.csv',
                previous_pair_summary, row_summary,
                epoch_dir / 'row_changes', epoch, 'row_ce',
            ))

        # 第二阶段：同一批配对输出，既要求每行答对，也要求组合答对。
        # N 单例仍在第一阶段训练；这里不额外抽样、不增加前向次数。
        model.train()
        pair_losses = train_pairs(
            model, pair_loader, optimizer, row_criterion, pair_criterion, device,
        )
        print(f'第一阶段 Row CE：{row_loss:.6f}')
        print(
            f"第二阶段 Row CE：{pair_losses['pair_stage_row_loss']:.6f}；"
            f"Pair CE：{pair_losses['pair_loss']:.6f}；"
            f"联合损失：{pair_losses['pair_stage_total_loss']:.6f}"
        )
        print(f'\n===== 第 {epoch} 轮：Row CE + Pair CE 后 =====')
        summary = evaluate_without_changing_rng(model, dev_loader, dev, device, pair_dir)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, stage='after_pair', row_loss=row_loss, **pair_losses)
            stage_history.append(record)
        pd.DataFrame(stage_history).to_csv(
            results_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig',
        )
        effects = compare_stages(
            row_dir / 'predictions_with_scores.csv',
            pair_dir / 'predictions_with_scores.csv',
            row_summary, summary, epoch_dir / 'pair_changes', epoch, 'row_plus_pair_ce',
        )
        phase_history.extend(effects)
        pd.DataFrame(phase_history).to_csv(
            results_dir / 'phase_effects.csv', index=False, encoding='utf-8-sig',
        )
        for effect in effects:
            print(
                f"{effect['method']}：联合阶段修好 {effect['fixed']} 条，"
                f"破坏 {effect['broken']} 条；"
                f"E 正确数变化 {effect['E_delta_correct']:+d}，"
                f"N 正确数变化 {effect['N_delta_correct']:+d}"
            )
        previous_pair_dir = pair_dir
        previous_pair_summary = summary
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, row_loss=row_loss, **pair_losses)
            history.append(record)
        pd.DataFrame(history).to_csv(
            results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
        )
        score = float(summary.loc[summary['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            # 只保存本组最佳权重，包含主干、原有分类头和新增词汇分支。
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            best_info = {
                'best_epoch': epoch, 'weighted_f1': score,
                'selection': 'after_pair pair-aware weighted F1',
                'independent_weighted_f1': float(summary.loc[
                    summary['method'] == 'independent', 'weighted_f1',
                ].iloc[0]),
            }
            (model_dir / 'best_metrics.json').write_text(
                json.dumps(best_info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')

    # 仍只在 after_pair 的各轮之间选最佳，不把 after_row 加入候选。
    best_prediction = results_dir / f'epoch_{best_epoch}/after_pair/predictions_with_scores.csv'
    print_stage_table(pd.DataFrame(stage_history))
    if use_lexical_features:
        compare_with_reference(results_dir)
    compare_with_original(best_prediction, results_dir / 'comparison_with_original')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', model_dir / 'best_model.pth')
    print('最佳预测：', best_prediction)
    print('两阶段汇总：', results_dir / 'stage_summary.csv')
    print('各阶段改动：', results_dir / 'phase_effects.csv')
    print('内存中的模型仍是最后一轮；重新预测请运行 evaluate_best.py 加载最佳权重。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='关闭词汇分支，重跑原版对照')
    args = parser.parse_args()
    main(use_lexical_features=cfg.USE_LEXICAL_FEATURES and not args.baseline)
