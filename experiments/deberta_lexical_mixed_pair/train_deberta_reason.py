"""右键运行：原混合版 + Train依据词监督；Dev只输入原文。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config_reason as cfg
from experiments.deberta_lexical_mixed_pair.reason_data import (
    ReasonGroupDataset, ReasonRowDataset, collate_reason_groups,
    load_reason_train, save_prepared, file_hash,
)
from experiments.deberta_lexical_mixed_pair.reason_model import ReasonDebertaModel, evidence_loss
from experiments.deberta_lexical_mixed_pair.train_deberta_mixed_pair import compute_joint_loss
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, score_logits


def load_tokenizer(settings, tokenizer_path=None):
    return AutoTokenizer.from_pretrained(
        tokenizer_path or settings['model_name'], cache_dir=get_model_cache_dir(),
        revision=settings['base_revision'] if tokenizer_path is None else None,
        local_files_only=True, use_fast=True,
    )


def build_model(tokenizer, settings):
    backbone = AutoModel.from_pretrained(
        settings['model_name'], revision=settings['base_revision'],
        cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    return ReasonDebertaModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=settings['lexical_hidden_size'],
        use_lexical_features=settings['use_lexical_features'],
    )


def build_optimizer(model):
    original, lexical, evidence = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith('lexical_output.'):
            lexical.append(parameter)
        elif name.startswith('evidence_head.'):
            evidence.append(parameter)
        else:
            original.append(parameter)
    return torch.optim.AdamW([
        {'params': original, 'lr': cfg.LEARNING_RATE},
        {'params': lexical, 'lr': cfg.LEXICAL_LEARNING_RATE},
        {'params': evidence, 'lr': cfg.EVIDENCE_LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)


def train_epoch(model, loader, optimizer, row_criterion, pair_criterion, device, reason_weight):
    model.train()
    totals = {'row_loss': 0.0, 'pair_loss': 0.0, 'evidence_loss': 0.0,
              'evidence_batches': 0, 'annotated_rows_seen': 0}
    counts = torch.zeros(4, dtype=torch.long)
    evidence_counts = torch.zeros(3, dtype=torch.long)  # TP、FP、FN
    for batch in tqdm(loader, desc='Row + Pair + 依据词监督'):
        optimizer.zero_grad(set_to_none=True)
        inputs = [batch[key].to(device) for key in ['input_ids', 'attention_mask', 'features']]
        if reason_weight > 0:
            logits, token_logits = model(*inputs, return_evidence=True)
            targets = batch['evidence_targets'].to(device)
            cue_loss = evidence_loss(token_logits, targets)
            valid = targets.ne(-100)
            predicted = token_logits.detach().argmax(dim=-1)
            evidence_counts += torch.tensor([
                int(((predicted == 1) & (targets == 1)).sum()),
                int(((predicted == 1) & (targets == 0)).sum()),
                int(((predicted == 0) & (targets == 1)).sum()),
            ])
            totals['evidence_batches'] += int(valid.any())
            totals['annotated_rows_seen'] += int(valid.any(dim=1).sum())
        else:
            # 同代码零权重对照：完整走原mixed的分类路径。
            logits = model(*inputs)
            cue_loss = logits.new_zeros(())
        joint, row_loss, pair_loss = compute_joint_loss(
            logits, batch['labels'].to(device), batch['pair_indices'].to(device),
            batch['pair_targets'].to(device), row_criterion, pair_criterion, cfg.PAIR_WEIGHT,
        )
        loss = joint + reason_weight * cue_loss
        loss.backward()
        optimizer.step()
        for name, value in [('row_loss', row_loss), ('pair_loss', pair_loss), ('evidence_loss', cue_loss)]:
            totals[name] += value.item()
        counts += torch.bincount(batch['labels'], minlength=4)
    for name in ['row_loss', 'pair_loss', 'evidence_loss']:
        totals[name] /= len(loader)
    tp, fp, fn = evidence_counts.tolist()
    totals['train_evidence_f1'] = 2 * tp / max(1, 2 * tp + fp + fn)
    totals['total_loss'] = totals['row_loss'] + cfg.PAIR_WEIGHT * totals['pair_loss'] + reason_weight * totals['evidence_loss']
    totals.update({f'seen_{label}': int(n) for label, n in zip(cfg.LABELS, counts)})
    return totals


def make_dev_loader(dev, tokenizer, settings):
    features = build_feature_tensor(dev)
    return DataLoader(
        ReasonRowDataset(dev, tokenizer, features, settings['max_length']),
        batch_size=settings['dev_batch_size'], shuffle=False,
        generator=torch.Generator().manual_seed(settings['seed'] + 1),
    )


def evaluate_saved(run_name):
    # 此路径不读取Train、reason CSV或依据标注。
    model_dir = cfg.MODEL_ROOT / run_name
    settings = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = load_tokenizer(settings, model_dir / 'tokenizer')
    model = build_model(tokenizer, settings)
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda')
    model.to(device).eval()
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    loader = make_dev_loader(dev, tokenizer, settings)
    output_dir = cfg.EXPERIMENT_DIR / 'outputs' / run_name / 'evaluation'
    summary = score_logits(predict_logits(model, loader, device), dev, output_dir, gold_path=cfg.GOLD_PATH)
    print('预测只输入原始text1、text2；不读取任何reason。')
    return summary


def main(run_name=None, baseline=False, prepare_only=False, evaluate=False):
    run_name = run_name or (cfg.RUN_NAME + '_baseline' if baseline else cfg.RUN_NAME)
    if evaluate:
        return evaluate_saved(run_name)
    set_seed(cfg.SEED)
    reason_weight = 0.0 if baseline else cfg.EVIDENCE_WEIGHT
    settings = {
        'run_name': run_name, 'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
        'seed': cfg.SEED, 'epochs': cfg.EPOCHS, 'max_length': cfg.MAX_LENGTH,
        'group_batch_size': cfg.GROUP_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES, 'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'learning_rate': cfg.LEARNING_RATE, 'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE,
        'evidence_learning_rate': cfg.EVIDENCE_LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': cfg.ROW_WEIGHTS, 'pair_weights': cfg.PAIR_WEIGHTS,
        'pair_weight': cfg.PAIR_WEIGHT, 'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'evidence_weight': reason_weight, 'selection': cfg.SELECT_BY,
        'training': 'mixed Row CE + 0.5 Pair CE + evidence_weight * token CE',
        'model_inputs': ['text1', 'text2'], 'reason_used_as_input': False,
        'annotation_independently_reviewed': False,
        'data_augmentation': False, 'labels': cfg.FULL_LABELS,
    }
    train = load_reason_train()
    tokenizer = load_tokenizer(settings)
    features = build_feature_tensor(train)
    dataset = ReasonGroupDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    if prepare_only:
        info = save_prepared(train, tokenizer, dataset.evidence_targets, cfg.PREPARED_DIR)
        keys = ['train_rows', 'reason_seed_groups', 'annotated_rows', 'unannotated_rows',
                'evidence_supervised_rows', 'reason_status_counts', 'evidence_label_counts']
        print(json.dumps({key: info[key] for key in keys}, ensure_ascii=False, indent=2))
        print('只准备Train CSV和依据token；没有加载主干权重、读取Dev或训练。')
        print('CSV：', cfg.PREPARED_DIR / 'train_with_reasons.csv')
        return

    model_dir = cfg.MODEL_ROOT / run_name
    output_dir = cfg.EXPERIMENT_DIR / 'outputs' / run_name
    if model_dir.exists() or output_dir.exists():
        raise FileExistsError('此RUN_NAME已有结果，请改RUN_NAME，保留原模型。')
    model_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    prepared = save_prepared(train, tokenizer, dataset.evidence_targets, output_dir)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    dev_loader = make_dev_loader(dev, tokenizer, settings)
    train_loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_reason_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    settings.update(prepared)
    settings['dev_sha256'] = file_hash(cfg.DEV_PATH)
    settings['steps_per_epoch'] = len(train_loader)
    source_paths = [cfg.EXPERIMENT_DIR / name for name in [
        'config_reason.py', 'reason_data.py', 'reason_model.py', 'train_deberta_reason.py',
        'data.py', 'train_deberta_mixed_pair.py',
    ]]
    source_paths += [cfg.PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe' / name
                     for name in ['model.py', 'lexical_features.py', 'data.py', 'evaluation.py']]
    settings['source_sha256'] = {str(path.relative_to(cfg.PROJECT_ROOT)): file_hash(path) for path in source_paths}
    for directory in [model_dir, output_dir]:
        (directory / 'run_config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    device = torch.device('cuda')
    model = build_model(tokenizer, settings).to(device)
    optimizer = build_optimizer(model)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    print('模型：DeBERTa + Alignment + Lexical + 训练期依据词头')
    print('Train / Dev：', len(train), '/', len(dev), '；有理由：', int(train.has_reason.sum()),
          '；参与依据监督：', int(train.has_evidence.sum()))
    print('本轮混合损失：Row CE +', cfg.PAIR_WEIGHT, '* Pair CE +', reason_weight, '* 依据词CE')
    print('选模：', cfg.SELECT_BY, '；预测时不提供reason。')
    print('模型目录：', model_dir)
    history = []
    best_f1 = -1.0
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        losses = train_epoch(model, train_loader, optimizer, row_criterion, pair_criterion, device, reason_weight)
        print(f"Row：{losses['row_loss']:.6f}；Pair：{losses['pair_loss']:.6f}；依据：{losses['evidence_loss']:.6f}")
        summary = score_logits(predict_logits(model, dev_loader, device), dev,
                               output_dir / f'epoch_{epoch}', gold_path=cfg.GOLD_PATH)
        for row in summary.to_dict('records'):
            history.append({'epoch': epoch, **row, **losses})
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method.eq(cfg.SELECT_BY), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1 = score
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            info = {'best_epoch': epoch, 'selection': cfg.SELECT_BY, 'weighted_f1': score,
                    'same_checkpoint_metrics': summary.to_dict('records')}
            for directory in [model_dir, output_dir]:
                (directory / 'best_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，{cfg.SELECT_BY} F1={score:.6f}')
    print('\n最佳权重：', model_dir / 'best_model.pth')
    print('同一轮独立/结构成绩：', output_dir / 'best_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name')
    parser.add_argument('--baseline', action='store_true', help='依据权重为0，同代码mixed对照。')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--evaluate', action='store_true', help='只读取最佳权重预测Dev，不读取Train理由。')
    args = parser.parse_args()
    main(args.run_name, args.baseline, args.prepare_only, args.evaluate)
