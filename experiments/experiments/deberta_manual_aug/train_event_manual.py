"""右键运行：保留0.935493配方和旧scope200，再加动作/事件200行。"""

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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, score_logits
from experiments.deberta_manual_aug.prepare_event_data import (
    prepare, PREPARED_DIR, SOURCE_PATH, FAMILY_PATH, REFERENCE_DIR,
)


# 只回到0.935的配方，不加载Dev选出的旧权重续训；仍从相同NLI预训练初始化。
# 后续复验/对照只改RUN_NAME和AUGMENT_MODE，旧文件和旧权重保留。
RUN_NAME = 'run03_event_200'
AUGMENT_MODE = 'manual'  # manual=旧3242+事件200；copy=旧3242+复制200；baseline=旧3242。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug') / RUN_NAME
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RealPairDataset(Dataset):
    """只使用训练集中真实存在的两行；N 单例不参与三组合 Pair CE。"""

    def __init__(self, rows):
        self.rows = rows
        self.pairs = []
        targets = {(0, 1): 0, (1, 0): 1, (2, 2): 2}
        for _, group in rows.frame.groupby('pair_id'):
            if len(group) != 2:
                continue
            first, second = group.index.tolist()
            labels = tuple(rows.labels[[first, second]].tolist())
            self.pairs.append((first, second, targets[labels]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second, target = self.pairs[index]
        row0, row1 = self.rows[first], self.rows[second]
        return (row0['input_ids'], row0['attention_mask'],
                row1['input_ids'], row1['attention_mask'],
                torch.tensor(target, dtype=torch.long))


def prepare_data():
    if (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已有训练记录，请换新名字，保留旧结果。')
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    combined, manual, report = prepare()
    baseline = combined.iloc[:report['baseline_rows']].copy()
    if AUGMENT_MODE == 'manual':
        train, extra = combined, manual
    elif AUGMENT_MODE == 'copy':
        train = pd.read_csv(PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
        extra = pd.read_csv(PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    elif AUGMENT_MODE == 'baseline':
        train, extra = baseline, baseline.iloc[:0].copy()
    else:
        raise ValueError('AUGMENT_MODE只能为manual、copy或baseline。')
    validate_groups(train)
    validate_groups(dev)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(RESULTS_DIR / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    # 保存当次源表、标签理由及审计快照；以后修改源表不会抹掉本次记录。
    (RESULTS_DIR / 'event_examples_source.csv').write_bytes(SOURCE_PATH.read_bytes())
    (RESULTS_DIR / 'event_families_source.csv').write_bytes(FAMILY_PATH.read_bytes())
    (RESULTS_DIR / 'manual_label_notes.csv').write_bytes((PREPARED_DIR / 'manual_label_notes.csv').read_bytes())
    (RESULTS_DIR / 'overlap_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    info = {
        'mode': AUGMENT_MODE, 'original_rows': report['original_rows'],
        'baseline_rows': len(baseline), 'old_scope_rows': report['old_scope_rows'],
        'extra_rows': len(extra), 'train_rows': len(train), 'dev_rows': len(dev),
        'extra_class_counts': extra.label.value_counts().to_dict(),
        'effective_train_sha256': hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest(),
        'manual_source_sha256': hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        'event_family_source_sha256': hashlib.sha256(FAMILY_PATH.read_bytes()).hexdigest(),
        'event_design': report['event_design'],
        'baseline_snapshot_canonical_sha256': report['baseline_snapshot_canonical_sha256'],
        'annotation_note': 'Assistant-handcrafted draft; no independent human adjudication; Dev-guided family selection disclosed.',
        'new_manual_vs_dev': report['new_manual_vs_dev'],
        'audit_scope': 'Zero vocabulary/stem overlap applies to NEW MANUAL rows; old scope/Train/copy control are separately audited.',
    }
    (RESULTS_DIR / 'data_config.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('数据方案：', AUGMENT_MODE)
    print('0.935基准 / 本次新增 / 构造后：', len(baseline), '/', len(extra), '/', len(train))
    print('新增类别数量：', info['extra_class_counts'])
    return train, dev, info


def compare_with_reference(prediction_path, output_dir):
    old = pd.read_csv(REFERENCE_DIR / 'epoch_3/predictions_with_scores.csv', keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    old = old[['instance_id', 'pair_prediction']].rename(columns={'pair_prediction': 'reference_prediction'})
    merged = current.merge(old, on='instance_id', validate='one_to_one')
    assert len(merged) == len(current) == len(old)
    short = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
             'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
    gold = merged['gold_label'].map(short)
    old_correct = merged['reference_prediction'] == gold
    new_correct = merged['pair_prediction'] == gold
    fixed = merged[~old_correct & new_correct]
    broken = merged[old_correct & ~new_correct]
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
    fixed.to_csv(output_dir / 'fixed.csv', index=False, encoding='utf-8-sig')
    broken.to_csv(output_dir / 'broken.csv', index=False, encoding='utf-8-sig')
    print(f'相对0.935493：修好 {len(fixed)} 条，新增错误 {len(broken)} 条。')


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    # 第一阶段：包含 N 单例和增强行的全部训练数据，正常做四分类。
    for batch in tqdm(loader, desc='四分类 Row CE'):
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss = criterion(logits, batch['label'].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    # 第二阶段：沿用原版的两个独立前向及 FB / BF / EE 三分类 CE。
    # 这是独立的 Pair 优化阶段，不是同一步的 Row CE + Pair CE 联合损失。
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='真实 Pair CE'):
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        loss = PAIR_WEIGHT * pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def evaluate(model, loader, dev, device, output_dir):
    was_training = model.training
    cuda_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    # 原版在训练完后统一评估；现在逐轮评估，保存/恢复 RNG，避免改变后续训练序列。
    with torch.random.fork_rng(devices=cuda_devices):
        model.eval()
        values = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc='Dev 预测'):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
                values.append(logits.float().cpu())
        summary = score_logits(torch.cat(values), dev, output_dir, gold_path=GOLD_PATH)
    model.train(was_training)
    return summary


def save_weights(model, path):
    # 刷新纪录才保存；临时文件写完再替换上一份最佳模型。
    temporary = path.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def main():
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已使用，请换新名字，避免覆盖旧模型。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练请使用你的 CUDA 环境。')
    train, dev, data_info = prepare_data()
    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    # 与原版同结构，从相同 NLI 预训练初始化重新训练；不读取 epoch_7 接着训练。
    # 全部参数默认可训练，不调用冻结函数，不创建词汇分支。
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    ).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    run_config = {
        'run_name': RUN_NAME, 'model_name': MODEL_NAME, 'seed': SEED,
        'epochs': EPOCHS, 'max_length': MAX_LENGTH, 'dtype': 'float32',
        'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'dev_batch_size': DEV_BATCH_SIZE, 'row_weights': ROW_WEIGHTS,
        'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'use_lexical_features': False, 'augmentation': data_info,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'real_pairs': len(pairs), 'row_steps_per_epoch': len(row_loader),
        'pair_steps_per_epoch': len(pair_loader),
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'historical_reference': 0.9354929652717138,
        'comparison_note': 'Keep original two-stage recipe and exact 3242-row scope snapshot; add handcrafted rows. Use copy mode for matched row counts, class counts and update budget.',
        'source_sha256': {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ['train_event_manual.py', 'prepare_event_data.py', 'prepare_manual_data.py',
                         'event_families.csv', 'event_examples.csv']
        },
        'train_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(DEV_PATH.read_bytes()).hexdigest(),
    }
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )
    print('训练方案：0.935原版 DeBERTa + Token Alignment + 旧scope200 + 动作/事件200')
    print('全参数、无词汇分支、沿用两阶段训练；本次不改损失或选模规则。')
    print('可训练参数：', run_config['trainable_parameters'], '/', run_config['total_parameters'])
    print('Row 行数 / Pair 组数：', len(rows), '/', len(pairs))
    print('每轮 Row / Pair 步数：', len(row_loader), '/', len(pair_loader))
    print('训练顺序：完整 Row CE 一遍，再完整 0.5 × Pair CE 一遍')
    print('选模：Dev pair-aware Weighted F1；同时报告独立预测')
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)

    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}')
        summary = evaluate(model, dev_loader, dev, device, RESULTS_DIR / f'epoch_{epoch}')
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, row_loss=row_loss, pair_loss=pair_loss)
            history.append(record)
        pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method == 'pair_aware', 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, MODEL_DIR / 'best_model.pth')
            info = {'best_epoch': epoch, 'selection_method': 'pair_aware', 'weighted_f1': score}
            (MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'刷新最佳：第 {epoch} 轮，pair-aware F1={score:.6f}')

    best_predictions = RESULTS_DIR / f'epoch_{best_epoch}/predictions_with_scores.csv'
    compare_with_reference(best_predictions, RESULTS_DIR / 'comparison_with_0p935')
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', MODEL_DIR / 'best_model.pth')
    # 最佳模型选好后再测诊断，诊断不参与选模；复用刚才原版诊断的同一评估函数。
    weights = torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval()
    from experiments.deberta_minimal_contrasts.evaluate_original import evaluate_probes
    evaluate_probes(model, tokenizer, device, output_dir=RESULTS_DIR / 'preserved_diagnostics')
    print('训练和最佳模型诊断完成：', RESULTS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='只准备并检查数据，不加载模型、不训练')
    args = parser.parse_args()
    if args.prepare_only:
        train, dev, _ = prepare_data()
        pairs = int((train.groupby('pair_id').size() == 2).sum())
        print('Row / Pair 预计每轮步数：', math.ceil(len(train) / ROW_BATCH_SIZE), '/', math.ceil(pairs / PAIR_BATCH_SIZE))
        print('数据准备完成；未加载模型，未启动训练。')
    else:
        main()
