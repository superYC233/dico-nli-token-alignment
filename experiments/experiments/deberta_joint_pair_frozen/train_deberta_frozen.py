"""右键运行：联合损失不变，只冻结 embedding 和底部 6 层。"""

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
from experiments.deberta_joint_pair_frozen import config as cfg
from experiments.deberta_joint_pair.data import read_data, RowDataset, GroupDataset, collate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, score_logits, compare_with_original


def freeze_backbone(model):
    backbone = model.deberta_model
    encoder = backbone.encoder
    count = cfg.FREEZE_FIRST_N_LAYERS
    if not 0 <= count <= len(encoder.layer):
        raise ValueError('冻结层数必须在 0 和 Transformer 总层数之间。')

    if cfg.FREEZE_EMBEDDINGS:
        backbone.embeddings.requires_grad_(False)
        # DeBERTa 的相对位置表位于 encoder 内，也属于 embedding。
        if encoder.relative_attention:
            encoder.rel_embeddings.requires_grad_(False)
        if 'layer_norm' in encoder.norm_rel_ebd:
            encoder.LayerNorm.requires_grad_(False)

    for layer in encoder.layer[:count]:
        layer.requires_grad_(False)

    # 原 DeBERTa 的卷积位于底部层，和底部层一起冻结。
    if count > 0 and encoder.conv is not None:
        encoder.conv.requires_grad_(False)

    # 只冻结参数，不调用 eval()；保持原有训练模式和 dropout 设置。
    # CLS 分类头、Token Alignment 和顶部 Transformer 层继续更新。
    return model


def build_model(tokenizer):
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    # 复用已核对过的原版结构。False 表示完全不创建词汇分支。
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    )
    return freeze_backbone(model)


def compute_joint_loss(logits, labels, pair_indices, pair_targets, row_criterion, pair_criterion):
    # 所有行都有四分类监督，包含配对的两行，也包含 N 单例。
    row_loss = row_criterion(logits, labels)

    if len(pair_targets) > 0:
        first = logits[pair_indices[:, 0]]
        second = logits[pair_indices[:, 1]]
        pair_logits = get_pair_logits(first, second)
        pair_loss = pair_criterion(pair_logits, pair_targets)
    else:
        # 某个 batch 若全是 N 单例，就正常只训练四分类。
        pair_loss = logits.new_zeros(())

    loss = row_loss + cfg.PAIR_WEIGHT * pair_loss
    return loss, row_loss, pair_loss


def train_epoch(model, loader, optimizer, row_criterion, pair_criterion, device):
    model.train()
    row_total = 0.0
    pair_total = 0.0
    row_count = 0
    pair_count = 0

    for batch in tqdm(loader, desc='联合训练'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        pair_indices = batch['pair_indices'].to(device)
        pair_targets = batch['pair_targets'].to(device)

        optimizer.zero_grad(set_to_none=True)
        # 一次前向，同一份 logits 同时计算两个损失，不再分两个阶段。
        # 第三个参数原本是词汇特征；本版没有词汇分支，传 None 即可。
        logits = model(input_ids, attention_mask, None)
        loss, row_loss, pair_loss = compute_joint_loss(
            logits, labels, pair_indices, pair_targets, row_criterion, pair_criterion,
        )
        loss.backward()
        optimizer.step()

        row_total += row_loss.item()
        pair_total += pair_loss.item()
        row_count += len(labels)
        pair_count += len(pair_targets)

    return {
        # 记录各优化步骤的平均值；没有 pair 的步骤，其 Pair CE 记为 0。
        'row_loss': row_total / len(loader),
        'pair_loss': pair_total / len(loader),
        'train_rows': row_count, 'train_pairs': pair_count,
        'optimizer_steps': len(loader),
    }


def evaluate(model, loader, dev, device, output_dir):
    was_training = model.training
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc='Dev 预测'):
            logits = model(
                batch['input_ids'].to(device), batch['attention_mask'].to(device), None,
            )
            values.append(logits.float().cpu())
    model.train(was_training)
    # 直接复用已有的官方评分和 pair-aware，保持指标及解码公式一致。
    return score_logits(torch.cat(values), dev, output_dir, gold_path=cfg.GOLD_PATH)


def make_dev_loader(dev, tokenizer):
    return DataLoader(
        RowDataset(dev, tokenizer, cfg.MAX_LENGTH),
        batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        # DataLoader 自己使用独立随机数，不干扰训练 shuffle / dropout。
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )


def save_weights(model, path):
    # 只保留一个最佳模型；写完临时文件再替换，避免中断损坏已有最佳。
    temporary = path.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def main():
    set_seed(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, dev = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    train_dataset = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.GROUP_BATCH_SIZE,
        shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    dev_loader = make_dev_loader(dev, tokenizer)
    model = build_model(tokenizer).to(device)

    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, device=device),
    )
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
    )

    # 每次改 RUN_NAME 开新实验，避免覆盖之前的最佳权重。
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    run_config = {
        'model_name': cfg.MODEL_NAME, 'seed': cfg.SEED,
        'use_lexical_features': False, 'epochs': cfg.EPOCHS,
        'learning_rate': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'freeze_embeddings': cfg.FREEZE_EMBEDDINGS,
        'freeze_first_n_layers': cfg.FREEZE_FIRST_N_LAYERS,
        'freeze_bottom_conv': cfg.FREEZE_FIRST_N_LAYERS > 0,
        'frozen_modules_dropout': 'unchanged; model.train() still applies',
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'max_length': cfg.MAX_LENGTH, 'dtype': 'float32',
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT, 'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'group_batch_size': cfg.GROUP_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'train_rows': len(train), 'train_groups': len(train_dataset),
        'train_pairs': train_dataset.pair_count, 'steps_per_epoch': len(train_loader),
        'select_by': cfg.SELECT_BY, 'labels': cfg.FULL_LABELS,
        'training': 'one pass over groups; one forward and one joint optimizer step per batch',
        'row_exposure': 'each original row exactly once per epoch; no augmentation or oversampling',
        'pair_exposure': 'each original reversible group exactly once per epoch',
        'loss': 'full four-class Row CE + 0.5 * three-combination Pair CE',
        'comparison_note': 'same joint recipe as deberta_joint_pair/run01; only selected backbone parameters are frozen',
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'model_dir': str(cfg.MODEL_DIR),
    }
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2)
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(config_text, encoding='utf-8')

    print('模型：原版 DeBERTa + Token Alignment，无词汇分支')
    print('冻结 embedding：', cfg.FREEZE_EMBEDDINGS)
    print('冻结底部层数：', cfg.FREEZE_FIRST_N_LAYERS)
    print('可训练层数：', len(model.deberta_model.encoder.layer) - cfg.FREEZE_FIRST_N_LAYERS)
    print('可训练参数：', run_config['trainable_parameters'], '/', run_config['total_parameters'])
    print('Train / Dev：', len(train), '/', len(dev))
    print('训练组数 / 真实 Pair：', len(train_dataset), '/', train_dataset.pair_count)
    print('每步最多行数：', cfg.GROUP_BATCH_SIZE * 2)
    print('每轮步数：', len(train_loader))
    print('损失：Row CE +', cfg.PAIR_WEIGHT, '* Pair CE')
    print('最佳模型按此指标选择：', cfg.SELECT_BY, 'Dev Weighted F1')
    print('模型目录：', cfg.MODEL_DIR)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_epoch(model, train_loader, optimizer, row_criterion, pair_criterion, device)
        print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
        epoch_dir = cfg.RESULTS_DIR / f'epoch_{epoch}'
        summary = evaluate(model, dev_loader, dev, device, epoch_dir)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(
            cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
        )
        score = float(summary.loc[summary['method'] == cfg.SELECT_BY, 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            save_weights(model, cfg.MODEL_DIR / 'best_model.pth')
            info = {'best_epoch': epoch, 'selection_method': cfg.SELECT_BY, 'weighted_f1': score}
            (cfg.MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，{cfg.SELECT_BY} F1={score:.6f}')

    best_prediction = cfg.RESULTS_DIR / f'epoch_{best_epoch}/predictions_with_scores.csv'
    compare_with_original(best_prediction, cfg.RESULTS_DIR / 'comparison_with_original')
    print(f'\n最佳第 {best_epoch} 轮，{cfg.SELECT_BY} F1={best_f1:.6f}')
    print('最佳权重：', cfg.MODEL_DIR / 'best_model.pth')
    print('最佳轮次两种成绩：', cfg.RESULTS_DIR / 'best_summary.csv')
    print('内存里是最后一轮；evaluate_best.py 会重新加载保存的最佳权重。')


if __name__ == '__main__':
    main()
