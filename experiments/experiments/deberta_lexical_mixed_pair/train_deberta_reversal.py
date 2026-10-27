"""右键运行：原Alignment + Lexical混合版，只把Pair CE换成反向KL。"""
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
from experiments.deberta_lexical_mixed_pair import config_reversal as cfg
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.data import (
    build_feature_tensor, save_feature_table, validate_groups, DiCoDataset,
)
from experiments.deberta_lexical_stage_probe.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, score_logits
# 直接复用刚验证过的四维KL公式，不使用原生分类器的模型或训练循环。
from experiments.deberta_reversal_kl.train_deberta_reversal import compute_loss as compute_reversal_loss


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev


def load_model(tokenizer):
    backbone = AutoModel.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    return DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES,
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


def make_dev_loader(dev, tokenizer, features, max_length, batch_size):
    return DataLoader(
        DiCoDataset(dev, tokenizer, features, max_length),
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )


def train_epoch(model, loader, optimizer, row_criterion, device):
    model.train()
    total_row = 0.0
    total_kl = 0.0
    counts = torch.zeros(4, dtype=torch.long)
    pair_count = 0
    batches_without_n = 0
    for batch in tqdm(loader, desc='原Alignment + Lexical：Row CE + 反向KL'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch['input_ids'].to(device), batch['attention_mask'].to(device),
            batch['features'].to(device),
        )
        loss, row_loss, kl_loss = compute_reversal_loss(
            logits, batch['labels'].to(device), batch['pair_indices'].to(device),
            row_criterion, cfg.CONSISTENCY_WEIGHT,
        )
        loss.backward()
        optimizer.step()
        total_row += row_loss.item()
        total_kl += kl_loss.item()
        counts += torch.bincount(batch['labels'], minlength=4)
        pair_count += len(batch['pair_indices'])
        batches_without_n += int(not (batch['labels'] == 3).any())
    row_mean, kl_mean = total_row / len(loader), total_kl / len(loader)
    return {
        'row_loss': row_mean, 'reversal_kl': kl_mean,
        'weighted_kl': cfg.CONSISTENCY_WEIGHT * kl_mean,
        'total_loss': row_mean + cfg.CONSISTENCY_WEIGHT * kl_mean,
        'train_rows': int(counts.sum()), 'train_pairs': pair_count,
        'optimizer_steps': len(loader), 'batches_without_N': batches_without_n,
        **{f'seen_{label}': int(count) for label, count in zip(cfg.LABELS, counts)},
    }


def main():
    set_seed(cfg.SEED)
    train, dev = read_data()
    train_features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    dataset = GroupDataset(train, tokenizer, train_features, cfg.MAX_LENGTH)
    train_loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    dev_loader = make_dev_loader(dev, tokenizer, dev_features, cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE)
    model = load_model(tokenizer)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    save_feature_table(train, train_features, cfg.RESULTS_DIR / 'train_features.csv')
    save_feature_table(dev, dev_features, cfg.RESULTS_DIR / 'dev_features.csv')

    shared = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe'
    source_paths = [Path(__file__), Path(cfg.__file__), shared / 'model.py', shared / 'data.py',
                    shared / 'lexical_features.py', shared / 'evaluation.py',
                    Path(__file__).with_name('data.py'),
                    PROJECT_ROOT / 'experiments/deberta_reversal_kl/train_deberta_reversal.py']
    run_config = {
        'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
        'backbone_config': model.deberta_model.config.to_dict(),
        'seed': cfg.SEED, 'dtype': 'float32', 'use_nli_pooler': False,
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'epochs': cfg.EPOCHS, 'learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'row_label_smoothing': cfg.ROW_LABEL_SMOOTHING,
        'consistency_weight': cfg.CONSISTENCY_WEIGHT,
        'consistency': 'mean symmetric KL of all four classes; F/B reversed; true pairs only; both sides receive gradients',
        'pair_ce': False, 'scheduler': None, 'gradient_clipping': None,
        'max_length': cfg.MAX_LENGTH, 'group_batch_size': cfg.GROUP_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'labels': cfg.FULL_LABELS,
        'train_rows': len(train), 'dev_rows': len(dev),
        'train_groups': len(dataset), 'train_pairs': dataset.pair_count,
        'steps_per_epoch': len(train_loader), 'selection': cfg.SELECT_BY,
        'data_augmentation': False, 'row_exposure': 'each original row exactly once per epoch',
        'training': 'one mixed forward; original weighted Row CE + consistency_weight * reversal KL',
        'comparison': 'run01_mixed_pair architecture/data/Row CE/LRs retained; Pair CE replaced; selection is independent',
        'removed_with_pair_ce': {'old_pair_weights': [1.0, 1.0, 1.25], 'old_pair_label_smoothing': 0.05},
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')

    device = torch.device('cuda')
    model.to(device)
    optimizer = build_optimizer(model)
    row_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.ROW_WEIGHTS, device=device), label_smoothing=cfg.ROW_LABEL_SMOOTHING,
    )
    print('模型：原DeBERTa + Token Alignment + Lexical，没有pooler或新头')
    print('Train / Dev：', len(train), '/', len(dev))
    print('真实反向组 / 每轮步数：', dataset.pair_count, '/', len(train_loader))
    print('只替换Pair CE：Row CE +', cfg.CONSISTENCY_WEIGHT, '* 反向KL')
    print('Row权重：', cfg.ROW_WEIGHTS, '；Row smoothing：', cfg.ROW_LABEL_SMOOTHING)
    print('主干/原头学习率：', cfg.LEARNING_RATE, '；Lexical学习率：', cfg.LEXICAL_LEARNING_RATE)
    print('无warmup/调度器/新增梯度裁剪；每轮评估，按', cfg.SELECT_BY, '保存最佳')
    print('模型目录：', cfg.MODEL_DIR)
    print('结果目录：', cfg.RESULTS_DIR)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_epoch(model, train_loader, optimizer, row_criterion, device)
        print(f"Row CE：{losses['row_loss']:.6f}；反向KL：{losses['reversal_kl']:.6f}")
        summary = score_logits(
            predict_logits(model, dev_loader, device), dev,
            cfg.RESULTS_DIR / f'epoch_{epoch}', gold_path=cfg.GOLD_PATH,
        )
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary['method'] == cfg.SELECT_BY, 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            torch.save(model.state_dict(), cfg.MODEL_DIR / 'best_model.pth')
            best = {'best_epoch': epoch, 'selection': cfg.SELECT_BY, 'weighted_f1': score,
                    'same_checkpoint_metrics': summary.to_dict('records')}
            for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
                (folder / 'best_metrics.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，independent F1={score:.6f}')
    print(f'\n最佳第 {best_epoch} 轮，independent F1={best_f1:.6f}')
    print('最佳权重：', cfg.MODEL_DIR / 'best_model.pth')
    print('同一权重的独立/结构成绩：', cfg.RESULTS_DIR / 'best_summary.csv')


if __name__ == '__main__':
    main()
