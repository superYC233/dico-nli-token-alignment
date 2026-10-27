"""右键运行：原0.938两阶段训练 + 新家族768行；复制对照只改AUGMENT_MODE。"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_family_aug.prepare_family_data import prepare, PREPARED_DIR, file_hash
from experiments.deberta_family_aug.evaluate_family_probes import evaluate_probes
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed, save_weights
from experiments.deberta_manual_aug.train_relation_manual import train_staged, PAIR_WEIGHT
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase


RUN_NAME = 'run01_family768'
# AUGMENT_MODE = 'manual'  # manual=3290+新768；copy=3290+复制768；baseline=原3290。
AUGMENT_MODE = 'copy'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
EPOCHS = 10
MAX_LENGTH = 128
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05

HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_family_aug')
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'


def main(run_name=RUN_NAME, augment_mode=AUGMENT_MODE, prepare_only=False):
    name = run_name + '_' + augment_mode
    results_dir = HERE / 'outputs' / name
    model_dir = MODEL_ROOT / name
    if model_dir.exists() or (results_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换新名字，保留旧模型。')
    combined, extra, _, report = prepare()
    if augment_mode == 'manual':
        train = combined
    elif augment_mode == 'copy':
        train = pd.read_csv(PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
    elif augment_mode == 'baseline':
        train = combined.iloc[:report['baseline_rows']].copy()
    else:
        raise ValueError('AUGMENT_MODE只能为manual、copy或baseline。')
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    pair_count = int(train.groupby('pair_id').size().eq(2).sum())
    print('数据方案：', augment_mode, '；Train / Dev：', len(train), '/', len(dev))
    print('安排：原0.938两阶段，每轮完整Row CE，再完整0.5×Pair CE；本次只改数据。')
    print('每轮Row / Pair步数：', math.ceil(len(train) / ROW_BATCH_SIZE), '/', math.ceil(pair_count / PAIR_BATCH_SIZE))
    print('保留诊断不进入Train、不参与选模。')
    print('模型目录：', model_dir)
    print('结果目录：', results_dir)
    if prepare_only:
        print('数据已准备，没有加载模型或训练。')
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
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    train.to_csv(results_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    (results_dir / 'overlap_report.json').write_bytes((PREPARED_DIR / 'overlap_report.json').read_bytes())
    config = {
        'run_name': name, 'augment_mode': augment_mode, 'model_name': MODEL_NAME,
        'seed': SEED, 'epochs': EPOCHS, 'max_length': MAX_LENGTH, 'dtype': 'float32',
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'dev_batch_size': DEV_BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
        'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'train_rows': len(train), 'real_pairs': len(pairs),
        'optimizer_steps_per_epoch': len(row_loader) + len(pair_loader),
        'training_mode': 'staged', 'use_lexical_features': False,
        'freeze_embeddings': False, 'freeze_first_n_layers': 0,
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties',
        'probe_selection': False, 'diagnostic_rows': report['diagnostic_rows'],
        'augmentation_report_sha256': file_hash(PREPARED_DIR / 'overlap_report.json'),
        'train_sha256': file_hash(results_dir / 'training_data.csv'),
        'dev_sha256': file_hash(DEV_PATH),
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__), HERE / 'family_examples.py', HERE / 'prepare_family_data.py',
            ROOT / 'experiments/deberta_manual_aug/train_relation_manual.py',
            ROOT / 'experiments/deberta_lexical/model.py',
        ]},
    }
    for folder in [model_dir, results_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    best_f1, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第{epoch}/{EPOCHS}轮 =====')
        # 直接复用0.938版本的两个训练循环，不混入hinge/SAM/新约束。
        losses = train_staged(model, row_loader, pair_loader, optimizer, row_criterion, pair_criterion, device)
        assert losses['row_ce_rows'] == len(train) and losses['pair_ce_groups'] == len(pairs)
        print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
        summary = evaluate_phase(model, dev_loader, dev, device, results_dir / f'epoch_{epoch}', GOLD_PATH)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            info = {'best_epoch': epoch, 'selection': 'pair_aware', 'weighted_f1': score,
                    'same_checkpoint_metrics': summary.to_dict('records')}
            (model_dir / 'best_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'刷新最佳：第{epoch}轮，pair-aware F1={score:.6f}')
    print(f'\n最佳第{best_epoch}轮，pair-aware F1={best_f1:.6f}')
    print('唯一最佳权重：', model_dir / 'best_model.pth')
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    evaluate_probes(model, tokenizer, device, results_dir / 'heldout_families', MAX_LENGTH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--augment-mode', choices=['manual', 'copy', 'baseline'], default=AUGMENT_MODE)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.augment_mode, args.prepare_only)
