"""右键运行：沿用0.9381配方，只关闭Pair CE的标签平滑。"""
import argparse
import hashlib
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
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import (
    REFERENCE_DIR, GOLD_PATH, LABELS, FULL_LABELS,
    read_data, write_train_reference, train_rows, evaluate_phase, stage_changes,
)


RUN_NAME = 'run09_pair_no_smoothing'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
TRAIN_AUDIT_EPOCH = 2
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5

# 本次唯一的训练改动：原来是0.05，现在关闭。
PAIR_LABEL_SMOOTHING = 0.0

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME


def prepare_data():
    # 直接复制0.9381那次的3290行，不重新生成增强数据。
    train, dev, reference = read_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['training_data.csv', 'extra_train_rows.csv', 'relation_examples_source.csv',
                 'manual_label_notes.csv', 'train_word_sources.csv', 'overlap_report.json', 'data_config.json']:
        (RESULTS_DIR / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (RESULTS_DIR / 'reference_run_config.json').write_bytes((REFERENCE_DIR / 'run_config.json').read_bytes())
    write_train_reference(train, RESULTS_DIR / 'train_reference.csv')
    print('Train / Dev：', len(train), '/', len(dev))
    print('数据：官方3042 + 旧scope200 + 新48，与0.9381完全相同。')
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)
    return train, dev, reference


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    # 仍是原来的第二阶段：只有三组合Pair CE，使用同一个AdamW。
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='Pair CE：关闭标签平滑'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def error_breakdown(prediction_path, epoch, phase, split):
    frame = pd.read_csv(prediction_path, keep_default_na=False)
    gold = frame.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        predicted = frame[column]
        records.append({
            'epoch': epoch, 'phase': phase, 'split': split, 'method': method,
            'E_to_FB': int((gold.eq('E') & predicted.isin(['F', 'B'])).sum()),
            'FB_to_E': int((gold.isin(['F', 'B']) & predicted.eq('E')).sum()),
            'F_B_reversal': int(((gold.eq('F') & predicted.eq('B')) |
                                 (gold.eq('B') & predicted.eq('F'))).sum()),
            'nonN_to_N': int((gold.ne('N') & predicted.eq('N')).sum()),
            'N_to_FB': int((gold.eq('N') & predicted.isin(['F', 'B'])).sum()),
            'N_to_E': int((gold.eq('N') & predicted.eq('E')).sum()),
            'N_correct': int((gold.eq('N') & predicted.eq('N')).sum()),
            'errors': int(gold.ne(predicted).sum()),
        })
    return records


def compare_with_reference(prediction_path):
    old = pd.read_csv(REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    new = pd.read_csv(prediction_path, keep_default_na=False)
    assert old.instance_id.tolist() == new.instance_id.tolist()
    gold = new.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before, after = old[column].eq(gold), new[column].eq(gold)
        changed = old[column].ne(new[column])
        records.append({'method': method, 'changed': int(changed.sum()),
                        'fixed': int((~before & after).sum()),
                        'broken': int((before & ~after).sum()),
                        'changed_wrong': int((changed & ~before & ~after).sum())})
        details = new.loc[changed].copy()
        details['old_prediction'] = old.loc[changed, column].values
        details['new_prediction'] = new.loc[changed, column].values
        details.to_csv(RESULTS_DIR / f'changed_vs_0p938_{method}.csv', index=False, encoding='utf-8-sig')
    comparison = pd.DataFrame(records)
    comparison.to_csv(RESULTS_DIR / 'changes_vs_0p938.csv', index=False, encoding='utf-8-sig')
    print('\n相对0.9381最佳模型：')
    print(comparison.to_string(index=False))


def main(prepare_only=False):
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换新名字，保留旧模型。')
    train, dev, reference = prepare_data()
    if prepare_only:
        print('数据准备完成；未加载模型、未启动训练。')
        return

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE,
                            shuffle=False, generator=torch.Generator().manual_seed(SEED + 1))
    train_eval_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
                                   generator=torch.Generator().manual_seed(SEED + 2))
    # 从同一NLI初始化重新训练，不加载0.9381的最佳权重。
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                        label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    config = dict(reference)
    config.update({
        'run_name': RUN_NAME, 'model_name': MODEL_NAME, 'seed': SEED, 'epochs': EPOCHS,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'row_batch_size': ROW_BATCH_SIZE,
        'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING, 'train_rows': len(rows),
        'real_pairs': len(pairs), 'row_steps_per_epoch': len(row_loader),
        'pair_steps_per_epoch': len(pair_loader),
        'optimizer_steps_per_epoch': len(row_loader) + len(pair_loader),
        'reference_run': str(REFERENCE_DIR),
        'only_training_change': 'Pair label smoothing: 0.05 -> 0.0',
        'training_order': 'all Row CE batches, then all Pair CE batches; same optimizer',
        'selection': 'best Dev pair-aware weighted F1 AFTER Pair; earliest epoch on ties',
        'train_audit_epoch': TRAIN_AUDIT_EPOCH, 'average_enabled': False, 'keep_loss_enabled': False,
        'evaluation_preserves_torch_rng': True,
        'source_sha256': {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    })
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('唯一训练改动：Pair标签平滑 0.05 →', PAIR_LABEL_SMOOTHING)
    print('每轮Row / Pair步数：', len(row_loader), '/', len(pair_loader))
    print('每轮先Row CE，再0.5 × Pair CE；按Pair结束后的结构F1只保存一个最佳模型。')

    history, changes, errors = [], [], []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第{epoch}/{EPOCHS}轮：Row =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        for phase in ['after_row', 'after_pair']:
            if phase == 'after_pair':
                pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
                print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
            output_dir = RESULTS_DIR / f'epoch_{epoch}' / phase
            print(f'\n===== 第{epoch}轮 {phase}：Dev =====')
            dev_summary = evaluate_phase(model, dev_loader, dev, device, output_dir / 'dev', GOLD_PATH)
            summaries = [('dev', dev_summary)]
            if epoch == TRAIN_AUDIT_EPOCH:
                print(f'\n===== 第{epoch}轮 {phase}：完整Train诊断 =====')
                train_summary = evaluate_phase(model, train_eval_loader, train, device, output_dir / 'train',
                                               RESULTS_DIR / 'train_reference.csv')
                summaries.append(('train', train_summary))
            for split, summary in summaries:
                for item in summary.to_dict('records'):
                    item.update(epoch=epoch, phase=phase, split=split, row_loss=row_loss,
                                pair_loss=pair_loss if phase == 'after_pair' else None)
                    history.append(item)
                errors.extend(error_breakdown(output_dir / split / 'predictions_with_scores.csv',
                                              epoch, phase, split))
            pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(errors).to_csv(RESULTS_DIR / 'error_breakdown.csv', index=False, encoding='utf-8-sig')

        score = float(dev_summary.loc[dev_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, MODEL_DIR / 'best_model.pth')
            dev_summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            (MODEL_DIR / 'best_metrics.json').write_text(json.dumps(
                {'best_epoch': epoch, 'weighted_f1': score, 'selection_phase': 'after_pair'}, indent=2),
                encoding='utf-8')
            print(f'刷新最佳：第{epoch}轮，pair-aware F1={score:.6f}')
        for split in (['dev', 'train'] if epoch == TRAIN_AUDIT_EPOCH else ['dev']):
            changes.extend(stage_changes(epoch, split, RESULTS_DIR / f'epoch_{epoch}/after_row' / split,
                                         RESULTS_DIR / f'epoch_{epoch}/after_pair' / split))
        pd.DataFrame(changes).to_csv(RESULTS_DIR / 'phase_changes.csv', index=False, encoding='utf-8-sig')

    versions = [('reference_0p938', 2, REFERENCE_DIR / 'epoch_2/summary.csv'),
                ('best_no_smoothing', best_epoch, RESULTS_DIR / 'best_summary.csv'),
                ('fixed_epoch_2', 2, RESULTS_DIR / 'epoch_2/after_pair/dev/summary.csv')]
    comparison = []
    for version, epoch, path in versions:
        for item in pd.read_csv(path).to_dict('records'):
            comparison.append({'version': version, 'epoch': epoch, **item})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False, encoding='utf-8-sig')
    print('\n===== 最终比较 =====')
    print(comparison[['version', 'epoch', 'method', 'weighted_f1', 'N_correct',
                      'FBE_correct']].to_string(index=False))
    compare_with_reference(RESULTS_DIR / f'epoch_{best_epoch}/after_pair/dev/predictions_with_scores.csv')
    print('最佳权重：', MODEL_DIR / 'best_model.pth')
    print('完整结果：', RESULTS_DIR / 'final_comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(prepare_only=args.prepare_only)
