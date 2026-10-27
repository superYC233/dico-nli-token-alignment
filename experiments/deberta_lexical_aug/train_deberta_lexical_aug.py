"""右键运行：词汇特征版 + 相同的200条范围增强；全参数、两阶段训练。"""

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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical import lexical_features
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.data import (
    validate_groups, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical.evaluation import (
    get_pair_logits, predict_logits, pair_aware, score_logits, compare_with_original,
)
from experiments.deberta_original_aug.scope_augmentation import build_train_variant, DIAGNOSTIC_PATH


# 本文件独立配置，不修改旧词汇版或原版增强实验的配置。
RUN_NAME = 'run01_lexical_scope'
AUGMENT_MODE = 'scope'  # scope=原来的200条增强；copy=等量复制；none=原始Train。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# 与0.930569词汇版相同的特征、分支和学习率。
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 10
MAX_LENGTH = 128
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_aug') / RUN_NAME
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
LEXICAL_REFERENCE = ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42'
AUGMENTED_REFERENCE = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    if (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已有训练记录，请换新名字，保留旧结果。')
    original = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    train, extra, audit = build_train_variant(original, dev, AUGMENT_MODE, SEED)
    validate_groups(train)
    validate_groups(dev)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(RESULTS_DIR / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(RESULTS_DIR / 'augmentation_audit.csv', index=False, encoding='utf-8-sig')
    info = {
        'mode': AUGMENT_MODE, 'original_rows': len(original), 'extra_rows': len(extra),
        'train_rows': len(train), 'dev_rows': len(dev),
        'extra_class_counts': extra.label.value_counts().to_dict(),
        'effective_train_sha256': hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest(),
        'annotation_note': 'Same synthetic scope data as original augmentation; no independent adjudication.',
    }
    (RESULTS_DIR / 'data_config.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('数据方案：', AUGMENT_MODE)
    print('原 Train / 新增 / 构造后：', len(original), '/', len(extra), '/', len(train))
    print('新增类别数量：', info['extra_class_counts'])
    return train, dev, info


def build_optimizer(model):
    original_parameters = []
    lexical_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith('lexical_output.'):
            lexical_parameters.append(parameter)
        else:
            original_parameters.append(parameter)
    return torch.optim.AdamW([
        {'params': original_parameters, 'lr': LEARNING_RATE},
        {'params': lexical_parameters, 'lr': LEXICAL_LEARNING_RATE},
    ], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for input_ids, attention_mask, features, labels in tqdm(loader, desc='四分类 Row CE'):
        logits = model(input_ids.to(device), attention_mask.to(device), features.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for ids0, mask0, features0, ids1, mask1, features1, targets in tqdm(loader, desc='真实 Pair CE'):
        # 两个方向各用各自的特征，仍是旧词汇版的两个独立前向。
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        loss = PAIR_WEIGHT * pair_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def evaluate(model, loader, dev, device, output_dir):
    # 与旧词汇版一样恢复所有随机状态，避免逐轮评估改变后面的训练。
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    try:
        logits = predict_logits(model, loader, device)
        return score_logits(logits, dev, output_dir, gold_path=GOLD_PATH)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def save_weights(model, path):
    temporary = path.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def compare_references(best_epoch, best_summary):
    from experiments.deberta_lexical_stage_probe.analyze_stages import compare_stages
    current = RESULTS_DIR / f'epoch_{best_epoch}/predictions_with_scores.csv'
    records = []
    for name, reference in [('lexical_0p930569', LEXICAL_REFERENCE),
                            ('original_aug_0p935493', AUGMENTED_REFERENCE)]:
        history_path = reference / 'epoch_summary.csv'
        if not history_path.exists():
            print('缺少参照记录，跳过：', reference)
            continue
        history = pd.read_csv(history_path)
        paired = history[history.method == 'pair_aware']
        old_epoch = int(paired.loc[paired.weighted_f1.idxmax(), 'epoch'])
        comparison = compare_stages(
            reference / f'epoch_{old_epoch}/predictions_with_scores.csv', current,
            pd.read_csv(reference / 'best_summary.csv'), best_summary,
            RESULTS_DIR / 'comparisons' / name, best_epoch, name,
        )
        records.extend(comparison)
    if records:
        table = pd.DataFrame(records)
        table.to_csv(RESULTS_DIR / 'reference_comparison.csv', index=False, encoding='utf-8-sig')
        print('\n===== 与旧词汇版、原版增强比较 =====')
        print(table[['training_phase', 'method', 'weighted_f1_before', 'weighted_f1_after',
                     'delta_f1_pp', 'fixed', 'broken']].to_string(index=False))


def evaluate_diagnostics(model, tokenizer, device):
    from experiments.deberta_minimal_contrasts.evaluate_diagnostics import read_review, save_summaries
    review = read_review(DIAGNOSTIC_PATH.parent)
    selected = review[review.eligible & (review.kind != 'anchor') &
                      (review.review_status != 'rejected')].copy().reset_index(drop=True)
    inputs = selected.copy()
    inputs['label'] = inputs.proposed_label.map(dict(zip(LABELS, FULL_LABELS)))
    # 词汇版诊断也必须生成词汇特征，不能把 None 传给这个模型。
    features = build_feature_tensor(inputs)
    loader = DataLoader(DiCoDataset(inputs, tokenizer, features, MAX_LENGTH),
                        batch_size=DEV_BATCH_SIZE, shuffle=False)
    logits = predict_logits(model, loader, device)
    result = selected.copy()
    result['prediction'] = [LABELS[i] for i in logits.argmax(1).tolist()]
    result['pair_prediction'] = [LABELS[i] for i in pair_aware(logits, inputs).tolist()]
    probabilities = logits.softmax(1)
    result['max_probability'] = probabilities.max(1).values.tolist()
    for index, label in enumerate(LABELS):
        result[f'logit_{label}'] = logits[:, index].tolist()
        result[f'prob_{label}'] = probabilities[:, index].tolist()
    print('\n===== 最佳词汇增强模型：独立诊断 =====')
    save_summaries(result.copy(), RESULTS_DIR / 'preserved_diagnostics/independent')
    structured = result.rename(columns={'prediction': 'independent_prediction',
                                        'max_probability': 'independent_max_probability'}).copy()
    structured['prediction'] = structured.pair_prediction
    print('\n===== 最佳词汇增强模型：结构解码诊断（使用自建分组）=====')
    save_summaries(structured, RESULTS_DIR / 'preserved_diagnostics/pair_aware')


def main():
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已使用，请换新名字，避免覆盖旧模型。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练请使用你的 CUDA 环境。')
    train, dev, data_info = prepare_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    row_loader = DataLoader(
        DiCoDataset(train, tokenizer, train_features, MAX_LENGTH),
        batch_size=ROW_BATCH_SIZE, shuffle=True,
    )
    pair_loader = DataLoader(
        PairDataset(train, tokenizer, train_features, MAX_LENGTH),
        batch_size=PAIR_BATCH_SIZE, shuffle=True,
    )
    dev_loader = DataLoader(
        DiCoDataset(dev, tokenizer, dev_features, MAX_LENGTH),
        batch_size=DEV_BATCH_SIZE, shuffle=False,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=LEXICAL_HIDDEN_SIZE, use_lexical_features=True,
    ).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    optimizer = build_optimizer(model)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    save_feature_table(train, train_features, RESULTS_DIR / 'train_features.csv')
    save_feature_table(dev, dev_features, RESULTS_DIR / 'dev_features.csv')
    run_config = {
        'run_name': RUN_NAME, 'model_name': MODEL_NAME, 'seed': SEED,
        'use_lexical_features': True, 'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'feature_source_sha256': hashlib.sha256(Path(lexical_features.__file__).read_bytes()).hexdigest(),
        'lexical_hidden_size': LEXICAL_HIDDEN_SIZE, 'learning_rate': LEARNING_RATE,
        'lexical_learning_rate': LEXICAL_LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS, 'max_length': MAX_LENGTH, 'dtype': 'float32',
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'dev_batch_size': DEV_BATCH_SIZE, 'row_weights': ROW_WEIGHTS,
        'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING, 'labels': FULL_LABELS,
        'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs': len(pair_loader.dataset),
        'data_augmentation': AUGMENT_MODE != 'none', 'augmentation': data_info,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
        'training_order': 'all Row CE batches, then all Real Pair CE batches',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'train_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(DEV_PATH.read_bytes()).hexdigest(),
        'lexical_reference': str(LEXICAL_REFERENCE), 'augmented_reference': str(AUGMENTED_REFERENCE),
        'model_dir': str(MODEL_DIR), 'results_dir': str(RESULTS_DIR),
        'comparison_note': 'Same lexical recipe and extra data; original lexical baseline has fewer updates.',
    }
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8',
        )
    print('模型：全参数 DeBERTa + Token Alignment + 20维词汇分支')
    print('词汇分支参数：', sum(p.numel() for p in model.lexical_output.parameters()))
    print('主模型 / 词汇分支学习率：', optimizer.param_groups[0]['lr'], '/', optimizer.param_groups[1]['lr'])
    print('Train / Dev / 真实Pair：', len(train), '/', len(dev), '/', len(pair_loader.dataset))
    print('每轮 Row / Pair 步数：', len(row_loader), '/', len(pair_loader))
    print('训练：先完整 Row CE，再完整 0.5 × Pair CE；按 pair-aware Dev F1 选最佳')
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
    compare_with_original(best_predictions, RESULTS_DIR / 'comparison_with_original')
    compare_references(best_epoch, pd.read_csv(RESULTS_DIR / 'best_summary.csv'))
    print(f'\n最佳第 {best_epoch} 轮，pair-aware F1={best_f1:.6f}')
    print('最佳权重：', MODEL_DIR / 'best_model.pth')
    state = torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    evaluate_diagnostics(model, tokenizer, device)
    print('训练和最佳模型诊断完成：', RESULTS_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='只准备数据，不加载模型、不训练')
    args = parser.parse_args()
    if args.prepare_only:
        train, dev, _ = prepare_data()
        pairs = int((train.groupby('pair_id').size() == 2).sum())
        print('Row / Pair 预计每轮步数：', math.ceil(len(train) / ROW_BATCH_SIZE), '/', math.ceil(pairs / PAIR_BATCH_SIZE))
        print('数据准备完成；未加载模型，未启动训练。')
    else:
        main()
