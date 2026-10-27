"""新增局部对照；复用0.938的模型、损失、两阶段训练和选模方式。"""
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
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_manual_aug import train_relation_manual as original
from experiments.deberta_manual_aug.prepare_local_contrasts import (
    BASELINE_DIR, COLUMNS, HERE, PREPARED_DIR, prepare,
)
from experiments.deberta_manual_aug.evaluate_local_contrasts import predict, summarize


RUN_NAME = 'run12_local128_copy'
AUGMENT_MODE = 'copy'
# AUGMENT_MODE = 'manual'  # manual=3290+新128；copy=3290+复制128；baseline=旧3290。
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME


def prepare_data():
    combined, diagnostic, audit = prepare()
    if AUGMENT_MODE == 'manual':
        train = combined
        extra = pd.read_csv(PREPARED_DIR / 'extra_train_rows.csv', keep_default_na=False)
    elif AUGMENT_MODE == 'copy':
        train = pd.read_csv(PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
        extra = pd.read_csv(PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    elif AUGMENT_MODE == 'baseline':
        train = pd.read_csv(BASELINE_DIR / 'training_data.csv', keep_default_na=False)
        extra = train.iloc[:0].copy()
    else:
        raise ValueError('AUGMENT_MODE只能为manual、copy或baseline。')
    train = train[COLUMNS]
    assert set(train.instance_id).isdisjoint(diagnostic.instance_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(RESULTS_DIR / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    for name in ['examples_source.csv', 'diagnostic_rows.csv', 'copy_control_sources.csv', 'audit.json']:
        (RESULTS_DIR / name).write_bytes((PREPARED_DIR / name).read_bytes())
    print('本次方案：', AUGMENT_MODE)
    print('训练行数 / 新增行数 / 独立诊断行数：', len(train), '/', len(extra), '/', len(diagnostic))
    return train, diagnostic, audit


def training_config(audit):
    # 逐项核对0.938保存的配置，避免不小心把其他实验的参数也带进来。
    config = {name: value for name, value in [
        ('model_name', original.MODEL_NAME), ('seed', original.SEED),
        ('epochs', original.EPOCHS), ('max_length', original.MAX_LENGTH), ('dtype', 'float32'),
        ('learning_rate', original.LEARNING_RATE), ('weight_decay', original.WEIGHT_DECAY),
        ('row_batch_size', original.ROW_BATCH_SIZE), ('pair_batch_size', original.PAIR_BATCH_SIZE),
        ('dev_batch_size', original.DEV_BATCH_SIZE), ('row_weights', original.ROW_WEIGHTS),
        ('pair_weights', original.PAIR_WEIGHTS), ('pair_weight', original.PAIR_WEIGHT),
        ('pair_label_smoothing', original.PAIR_LABEL_SMOOTHING),
        ('freeze_embeddings', False), ('freeze_first_n_layers', 0), ('use_lexical_features', False),
    ]}
    previous = json.loads((BASELINE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    assert all(config[name] == previous[name] for name in config), '训练参数与0.938基准不一致。'
    config.update(run_name=RUN_NAME, augment_mode=AUGMENT_MODE, training_mode='staged',
                  data_audit=audit, selection='best Dev pair-aware weighted F1; independent recorded at the same epoch',
                  diagnostic_in_training=False, diagnostic_used_for_selection=False,
                  initialization='same pretrained NLI checkpoint; not continuation of the Dev-selected 0.938 weights')
    config['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [Path(__file__), HERE / 'prepare_local_contrasts.py',
                                            HERE / 'train_relation_manual.py', HERE / 'train_boundary_manual.py',
                                            ROOT / 'experiments/deberta_lexical/model.py']}
    return config


def main(prepare_only=False):
    assert not MODEL_DIR.exists() and not (RESULTS_DIR / 'run_config.json').exists(), '请换新RUN_NAME，保留已有结果。'
    train, diagnostic, audit = prepare_data()
    config = training_config(audit)
    if prepare_only:
        print('数据和基准参数核对通过；没有加载模型或训练。')
        return

    original.set_seed(original.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        original.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        original.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    groups = original.GroupDataset(train, tokenizer, original.MAX_LENGTH)
    rows, pairs = groups.rows, original.RealPairDataset(groups.rows)
    row_loader = DataLoader(rows, batch_size=original.ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=original.PAIR_BATCH_SIZE, shuffle=True)
    dev = pd.read_csv(original.DEV_PATH, keep_default_na=False)
    dev_loader = DataLoader(original.RowDataset(dev, tokenizer, original.MAX_LENGTH),
                            batch_size=original.DEV_BATCH_SIZE, shuffle=False,
                            generator=torch.Generator().manual_seed(original.SEED + 1))
    model = original.DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(original.ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(original.PAIR_WEIGHTS, device=device),
                                          label_smoothing=original.PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=original.LEARNING_RATE,
                                  weight_decay=original.WEIGHT_DECAY)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    config.update(train_rows=len(rows), real_pairs=len(pairs), row_steps_per_epoch=len(row_loader),
                  pair_steps_per_epoch=len(pair_loader), trainable_parameters=sum(p.numel() for p in model.parameters()))
    for directory in [MODEL_DIR, RESULTS_DIR]:
        (directory / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    best_f1, history = -1.0, []
    for epoch in range(1, original.EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{original.EPOCHS} 轮：Row CE → Pair CE =====')
        losses = original.train_staged(model, row_loader, pair_loader, optimizer,
                                        row_criterion, pair_criterion, device)
        summary = original.evaluate(model, dev_loader, dev, device, RESULTS_DIR / f'epoch_{epoch}')
        independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        pair_score = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        for record in summary.to_dict('records'):
            history.append({**record, 'epoch': epoch, **losses,
                            'same_epoch_both_targets_met': independent >= 0.87 and pair_score >= 0.94})
        pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        if pair_score > best_f1:
            best_f1 = pair_score
            original.save_weights(model, MODEL_DIR / 'best_model.pth')
            best = {'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': pair_score,
                    'selection': 'pair_aware', 'same_epoch_both_targets_met': independent >= 0.87 and pair_score >= 0.94}
            (MODEL_DIR / 'best_metrics.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
            summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新最佳：', best)

    # 训练和按Dev选模全部结束后，才看留出诊断；诊断不参与梯度或挑轮次。
    model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True))
    logits = predict(model, tokenizer, diagnostic, device, original.MAX_LENGTH)
    summarize(logits, diagnostic, RESULTS_DIR / 'heldout_diagnostic')
    print('本次最佳模型：', MODEL_DIR / 'best_model.pth')
    print('同轮Dev指标：', MODEL_DIR / 'best_metrics.json')
    print('留出诊断：', RESULTS_DIR / 'heldout_diagnostic/summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.prepare_only)
