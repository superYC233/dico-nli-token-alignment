"""右键运行：DeBERTa + Alignment + Lexical，官方3042 + scope200 + 关系48。"""
import argparse
import hashlib
import json
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
from experiments.deberta_lexical.evaluation import get_pair_logits, predict_logits, score_logits


RUN_NAME = 'run02_lexical_relation_staged'
TRAINING_MODE = 'staged'  # 按旧配方：先完整Row，再完整Pair。
SELECTION_METRIC = 'pair_aware'  # 沿用旧选模方式，同时报告同轮independent。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

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

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
TRAIN_SNAPSHOT = REFERENCE_DIR / 'training_data.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical_aug')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(output_dir):
    # 直接复用0.938145保存的3290行，避免重新生成增强时内容/顺序变化。
    train = pd.read_csv(TRAIN_SNAPSHOT, keep_default_na=False)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    reference = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    canonical_hash = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert canonical_hash == reference['augmentation']['effective_train_sha256']
    assert len(train) == 3290 and len(dev) == 660
    assert train.label.value_counts().to_dict() == {
        'FORWARD_ENTAILMENT': 933, 'BACKWARD_ENTAILMENT': 933,
        'EQUIVALENCE': 894, 'NEGATIVE_OTHER': 530,
    }
    validate_groups(train)
    validate_groups(dev)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'training_data.csv').write_bytes(TRAIN_SNAPSHOT.read_bytes())
    train.iloc[3042:].to_csv(output_dir / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    info = {
        'original_rows': 3042, 'scope_rows': 200, 'relation_rows': 48,
        'train_rows': len(train), 'dev_rows': len(dev),
        'effective_train_sha256': canonical_hash,
        'same_training_snapshot_as_0p938': True,
        'new_augmentation_generated': False,
        'source_path': str(TRAIN_SNAPSHOT), 'source_file_sha256': file_hash(TRAIN_SNAPSHOT),
        'annotation_note': 'Reuse historical assistant-designed scope200 and relation48; no new annotations or local semantic head.',
        'isolation_scope': 'Historical Train/augmentation snapshot is unchanged; no new claim of vocabulary or hidden-Test isolation.',
    }
    (output_dir / 'data_config.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('数据：官方3042 + scope200 + 关系48 =', len(train))
    print('类别数量：', train.label.value_counts().to_dict())
    print('与原0.938的训练快照哈希一致：', canonical_hash)
    return train, dev, info


def build_model(tokenizer, model_name=MODEL_NAME, lexical_hidden=LEXICAL_HIDDEN_SIZE, revision=None):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), local_files_only=True,
        revision=revision, dtype=torch.float32, use_safetensors=True,
    )
    return DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(FEATURE_NAMES),
        lexical_hidden=lexical_hidden, use_lexical_features=True,
    )


def build_optimizer(model):
    original_parameters, lexical_parameters = [], []
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
    for ids, mask, features, labels in tqdm(loader, desc='四分类 Row CE + Lexical'):
        optimizer.zero_grad()
        logits = model(ids.to(device), mask.to(device), features.to(device))
        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for ids0, mask0, features0, ids1, mask1, features1, targets in tqdm(loader, desc='原版 Pair CE + Lexical'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def pair_stage(training_mode, model, loader, optimizer, criterion, device):
    if training_mode == 'row_only':
        # 完全跳过，避免零loss仍让Adam动量或weight decay更新参数。
        return 0.0
    return train_pairs(model, loader, optimizer, criterion, device)


def evaluate(model, loader, dev, device, output_dir):
    was_training = model.training
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        logits = predict_logits(model, loader, device)
        summary = score_logits(logits, dev, output_dir, gold_path=GOLD_PATH)
    model.train(was_training)
    predictions = pd.read_csv(Path(output_dir) / 'predictions_with_scores.csv', keep_default_na=False)
    gold_n = predictions.gold_label.eq('NEGATIVE_OTHER')
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        selected = summary.method.eq(method)
        predicted_n = predictions[column].eq('N')
        summary.loc[selected, 'N_predicted'] = int(predicted_n.sum())
        summary.loc[selected, 'non_N_to_N'] = int((~gold_n & predicted_n).sum())
        summary.loc[selected, 'N_to_non_N'] = int((gold_n & ~predicted_n).sum())
    summary.to_csv(Path(output_dir) / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary[['method', 'N_predicted', 'non_N_to_N', 'N_to_non_N']].to_string(index=False))
    return summary


def selected_score(summary, metric):
    return float(summary.loc[summary.method.eq(metric), 'weighted_f1'].iloc[0])


def save_weights(model, path):
    temporary = path.with_suffix('.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def compare_with_reference(prediction_path, output_dir):
    old = pd.read_csv(REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    joined = current.merge(old[['instance_id', 'prediction', 'pair_prediction']],
                           on='instance_id', suffixes=('', '_reference'), validate='one_to_one')
    short = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
             'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
    gold = joined.gold_label.map(short)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before, after = joined[column + '_reference'].eq(gold), joined[column].eq(gold)
        fixed, broken = joined[~before & after], joined[before & ~after]
        fixed.to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        broken.to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
        records.append({'method': method, 'fixed': len(fixed), 'broken': len(broken)})
        print(f'相对0.938的{method}：修好 {len(fixed)}，破坏 {len(broken)}。')
    pd.DataFrame(records).to_csv(output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')


def main(run_name=RUN_NAME, training_mode=TRAINING_MODE, selection_metric=SELECTION_METRIC, prepare_only=False):
    model_dir, results_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('RUN_NAME已有训练记录，请换一个名字，保留旧模型。')
    train, dev, data_info = prepare_data(results_dir)
    if prepare_only:
        print('数据准备完成；没有加载模型或训练。')
        return
    train_features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    row_loader = DataLoader(DiCoDataset(train, tokenizer, train_features, MAX_LENGTH),
                            batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = None
    if training_mode == 'staged':
        pair_loader = DataLoader(PairDataset(train, tokenizer, train_features, MAX_LENGTH),
                                 batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(DiCoDataset(dev, tokenizer, dev_features, MAX_LENGTH),
                            batch_size=DEV_BATCH_SIZE, shuffle=False,
                            generator=torch.Generator().manual_seed(SEED + 1))
    model = build_model(tokenizer).to(device)
    optimizer = build_optimizer(model)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                       label_smoothing=PAIR_LABEL_SMOOTHING)
    pair_steps = len(pair_loader) if pair_loader is not None else 0
    selection_stage = 'after_pair' if training_mode == 'staged' else 'after_row'
    settings = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'seed': SEED,
        'base_revision': getattr(model.deberta_model.config, '_commit_hash', None),
        'training_mode': training_mode, 'selection_metric': selection_metric,
        'selection_stage': selection_stage, 'epochs': EPOCHS, 'max_length': MAX_LENGTH,
        'dtype': 'float32', 'use_lexical_features': True, 'use_lora': False,
        'local_semantic_supervision': False, 'lexical_hidden_size': LEXICAL_HIDDEN_SIZE,
        'learning_rate': LEARNING_RATE, 'lexical_learning_rate': LEXICAL_LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'row_batch_size': ROW_BATCH_SIZE,
        'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
        'pair_weight': PAIR_WEIGHT if training_mode == 'staged' else 0.0,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs_in_data': 1380,
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': pair_steps,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'augmentation': data_info, 'selection': f'best Dev {selection_metric} at {selection_stage}; earliest on ties',
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), Path(lexical_features.__file__),
            ROOT / 'experiments/deberta_lexical/model.py', ROOT / 'experiments/deberta_lexical/data.py',
            ROOT / 'experiments/deberta_lexical/evaluation.py',
        ]},
        'comparison_scope': 'Default staged + pair_aware selection preserves the historical 0.938 training recipe and adds Lexical. Optional Row-only changes training schedule and must be reported separately.',
    }
    model_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    save_feature_table(train, train_features, results_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, results_dir / 'dev_features.csv')
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print('结构：DeBERTa + Token Alignment + Lexical；全参数训练。')
    print('训练模式 / 选模指标：', training_mode, '/', selection_metric)
    print('每轮Row / Pair步数：', len(row_loader), '/', pair_steps)
    print('原模型 / Lexical学习率：', LEARNING_RATE, '/', LEXICAL_LEARNING_RATE)
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        row_summary = None
        if training_mode == 'staged':
            print('\n===== Row阶段后 =====')
            row_summary = evaluate(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}/after_row')
        pair_loss = pair_stage(training_mode, model, pair_loader, optimizer, pair_criterion, device)
        print(f'Row CE：{row_loss:.6f}；Pair CE：{pair_loss:.6f}；实际Pair更新：{pair_steps}步')
        final_summary = evaluate(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}')
        stages = [(selection_stage, final_summary)]
        if row_summary is not None:
            stages.insert(0, ('after_row', row_summary))
        for stage, summary in stages:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, stage=stage, row_loss=row_loss, pair_loss=pair_loss,
                              pair_steps=pair_steps)
                history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = selected_score(final_summary, selection_metric)
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            final_summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            best_metrics = {
                'best_epoch': epoch, 'selection_metric': selection_metric, 'selection_stage': selection_stage,
                'weighted_f1': score,
                'independent_f1': selected_score(final_summary, 'independent'),
                'pair_aware_f1': selected_score(final_summary, 'pair_aware'),
            }
            (model_dir / 'best_metrics.json').write_text(json.dumps(best_metrics, indent=2), encoding='utf-8')
            print(f'刷新最佳：第 {epoch} 轮，{selection_metric} F1={score:.6f}')
    compare_with_reference(results_dir / f'epoch_{best_epoch}/predictions_with_scores.csv',
                           results_dir / 'comparison_with_0p938')
    print(f'\n最佳第 {best_epoch} 轮，{selection_metric} F1={best_f1:.6f}')
    print('同一最佳权重的独立/结构成绩：')
    print(pd.read_csv(results_dir / 'best_summary.csv')[['method', 'weighted_f1', 'FBE_correct', 'N_correct']].to_string(index=False))
    print('唯一最佳权重：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--training-mode', choices=['row_only', 'staged'], default=TRAINING_MODE)
    parser.add_argument('--selection-metric', choices=['independent', 'pair_aware'], default=SELECTION_METRIC)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.training_mode, args.selection_metric, args.prepare_only)
