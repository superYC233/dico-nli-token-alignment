"""右键运行：原始DeBERTa-v3-base + 原Lexical + 官方3042行；不加增强。"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from experiments.deberta_base_lexical import config as cfg
# 模型、20维特征、分组和评分直接复用无增强0.930569版。
from experiments.deberta_lexical.data import (
    validate_groups, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, predict_logits, score_logits
from experiments.deberta_lexical.train_deberta_lexical import set_seed


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev


def load_tokenizer(path=None):
    return AutoTokenizer.from_pretrained(path or cfg.get_pretrained_path(), local_files_only=True)


def build_model(tokenizer, device, pretrained_path=None,
                lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES):
    backbone = AutoModel.from_pretrained(
        pretrained_path or cfg.get_pretrained_path(), local_files_only=True,
        dtype=torch.float32, use_safetensors=False,
    )
    return DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=lexical_hidden, use_lexical_features=use_lexical_features,
    ).to(device)


def build_optimizer(model):
    original_parameters, lexical_parameters = [], []
    for name, parameter in model.named_parameters():
        if name.startswith('lexical_output.'):
            lexical_parameters.append(parameter)
        else:
            original_parameters.append(parameter)
    return torch.optim.AdamW([
        {'params': original_parameters, 'lr': cfg.LEARNING_RATE},
        {'params': lexical_parameters, 'lr': cfg.LEXICAL_LEARNING_RATE},
    ], lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)


def build_criteria(device):
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
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


def evaluate(model, loader, dev, device, output_dir):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    try:
        logits = predict_logits(model, loader, device)
        return score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def compare_with_nli(prediction_path, output_dir):
    # 只在训练结束后比较历史预测，历史结果不进入训练或选模。
    history = pd.read_csv(cfg.REFERENCE_DIR / 'epoch_summary.csv')
    selected = history[history.method.eq('pair_aware')]
    epoch = int(selected.loc[selected.weighted_f1.idxmax(), 'epoch'])
    old = pd.read_csv(cfg.REFERENCE_DIR / f'epoch_{epoch}/predictions_with_scores.csv', keep_default_na=False)
    new = pd.read_csv(prediction_path, keep_default_na=False)
    old = old[['instance_id', 'prediction', 'pair_prediction']].rename(columns={
        'prediction': 'nli_independent', 'pair_prediction': 'nli_pair_aware',
    })
    merged = new.merge(old, on='instance_id', validate='one_to_one')
    assert len(merged) == len(old) == len(new)
    full_to_short = dict(zip(['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
                             'EQUIVALENCE', 'NEGATIVE_OTHER'], ['F', 'B', 'E', 'N']))
    gold = merged.gold_label.map(full_to_short)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before = merged[f'nli_{method}'].eq(gold)
        after = merged[column].eq(gold)
        fixed, broken = merged[~before & after], merged[before & ~after]
        fixed.to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        broken.to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
        records.append({'method': method, 'fixed': len(fixed), 'broken': len(broken),
                        'net_correct': len(fixed) - len(broken)})
    pd.DataFrame(records).to_csv(output_dir / 'changes.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(records).to_string(index=False))


def main(run_name=cfg.RUN_NAME, prepare_only=False):
    model_dir, results_dir = cfg.get_output_dirs(run_name)
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请在config.py换新名字。')
    train, dev = read_data()
    train_features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    pretrained_path = cfg.get_pretrained_path()
    print('初始化：', cfg.MODEL_NAME, '；原始预训练版，无NLI任务微调。')
    print('本地模型：', pretrained_path)
    print('词汇特征分支：', cfg.USE_LEXICAL_FEATURES)
    print('Train / Dev：', len(train), '/', len(dev), '；只读官方数据，无增强。')
    print('真实Pair：', int(train.groupby('pair_id').size().eq(2).sum()))
    print('安排：每轮完整Row CE，再完整0.5×三组合Pair CE。')
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    if prepare_only:
        print('数据和特征检查完成，没有加载模型或训练。')
        return

    set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = load_tokenizer()
    backbone = AutoModel.from_pretrained(
        pretrained_path, local_files_only=True, dtype=torch.float32, use_safetensors=False,
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
        lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES,
    ).to(device)
    row_criterion, pair_criterion = build_criteria(device)
    optimizer = build_optimizer(model)
    print('主模型学习率 / 词汇分支学习率：', optimizer.param_groups[0]['lr'], '/', optimizer.param_groups[1]['lr'])
    print('词汇分支参数量：', sum(p.numel() for p in model.lexical_output.parameters()))
    model_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    save_feature_table(train, train_features, results_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, results_dir / 'dev_features.csv')
    config = {
        'model_name': cfg.MODEL_NAME, 'pretrained_path': str(pretrained_path),
        'initialization': 'original Microsoft pretrained checkpoint, not NLI-finetuned',
        'pretrained_format': 'pytorch_model.bin', 'use_safetensors': False,
        'seed': cfg.SEED, 'epochs': cfg.EPOCHS, 'dtype': 'float32',
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'feature_source_sha256': file_hash(PROJECT_ROOT / 'experiments/deberta_lexical/lexical_features.py'),
        'train_sha256': file_hash(cfg.TRAIN_PATH), 'dev_sha256': file_hash(cfg.DEV_PATH),
        'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs': len(pair_loader.dataset),
        'data_augmentation': False, 'max_length': cfg.MAX_LENGTH,
        'row_batch_size': cfg.ROW_BATCH_SIZE, 'pair_batch_size': cfg.PAIR_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'learning_rate': cfg.LEARNING_RATE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT, 'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'comparison_reference': str(cfg.REFERENCE_DIR),
        'source_sha256': {str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in [
            Path(__file__), cfg.EXPERIMENT_DIR / 'config.py',
            PROJECT_ROOT / 'experiments/deberta_lexical/model.py',
            PROJECT_ROOT / 'experiments/deberta_lexical/data.py',
            PROJECT_ROOT / 'experiments/deberta_lexical/evaluation.py',
        ]},
    }
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第{epoch}/{cfg.EPOCHS}轮 =====')
        model.train()
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
        summary = evaluate(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}')
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, row_loss=row_loss, pair_loss=pair_loss)
            history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            temporary = model_dir / 'best_model.tmp'
            torch.save(model.state_dict(), temporary)
            temporary.replace(model_dir / 'best_model.pth')
            info = {'best_epoch': epoch, 'weighted_f1': score,
                    'same_checkpoint_metrics': summary.to_dict('records')}
            (model_dir / 'best_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第{epoch}轮，pair-aware F1={score:.6f}')
    prediction_path = results_dir / f'epoch_{best_epoch}/predictions_with_scores.csv'
    compare_with_nli(prediction_path, results_dir / 'comparison_with_nli_0p9306')
    print(f'\n最佳第{best_epoch}轮，pair-aware F1={best_f1:.6f}')
    print('唯一最佳权重：', model_dir / 'best_model.pth')
    print('同一最佳权重的独立及结构成绩：', results_dir / 'best_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
