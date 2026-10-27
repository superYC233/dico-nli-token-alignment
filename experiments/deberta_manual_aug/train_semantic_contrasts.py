"""右键运行：原0.938两阶段训练 + 64行语义对照；只保存最佳完整模型。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, stage_changes
from experiments.deberta_bidirectional.data import write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights
from experiments.deberta_bidirectional.train_deberta_bidirectional import file_hash, training_records_exist
from experiments.deberta_manual_aug.prepare_semantic_contrasts import (
    build_train_dataset, frame_hash, SOURCE_PATH, PREPARED_DIR, REFERENCE_DIR,
)


RUN_NAME = 'run20_semantic64_staged_seed42'
AUGMENT_MODE = 'manual'  # manual=3290+新64；copy=3290+复制64；baseline=原3290。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

# 沿用最近的省显存设置。padding长度改变dropout随机轨迹，不能声称逐位复现历史0.938。
DYNAMIC_PADDING = True
GRADIENT_CHECKPOINTING = True
HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'


def build_model(tokenizer, model_name=MODEL_NAME, revision=BASE_REVISION):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    return DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)


def trim_inputs(ids, mask, dynamic_padding=DYNAMIC_PADDING):
    length = int(mask.sum(1).max()) if dynamic_padding else ids.shape[1]
    return ids[:, :length].contiguous(), mask[:, :length].contiguous()


def collate_rows(examples, dynamic_padding=DYNAMIC_PADDING):
    batch = default_collate(examples)
    batch['input_ids'], batch['attention_mask'] = trim_inputs(batch['input_ids'], batch['attention_mask'], dynamic_padding)
    return batch


def collate_pairs(examples):
    ids0, mask0, ids1, mask1, targets = default_collate(examples)
    ids0, mask0 = trim_inputs(ids0, mask0)
    ids1, mask1 = trim_inputs(ids1, mask1)
    return ids0, mask0, ids1, mask1, targets


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    total, weights = 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    for batch in tqdm(loader, desc='四分类 Row CE（全部四类）'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        labels = batch['label'].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += loss.item() * weight
        weights += weight
        counts += torch.bincount(batch['label'], minlength=4)
    return {'row_loss': total / weights, 'row_ce_rows': int(counts.sum()),
            **{f'seen_{label}': int(count) for label, count in zip(['F', 'B', 'E', 'N'], counts)}}


def train_pairs(model, loader, optimizer, criterion, device):
    # 与0.938版相同：三组合Pair CE，不加Row回放、NN或对抗项。
    model.train()
    total, weights, groups = 0.0, 0.0, 0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='真实 Pair CE（FB / BF / EE）'):
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        labels = targets.to(device)
        loss = criterion(get_pair_logits(first, second), labels)
        (PAIR_WEIGHT * loss).backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += loss.item() * weight
        weights += weight
        groups += len(targets)
    return {'pair_loss': total / weights, 'pair_ce_groups': groups}


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, augment_mode=AUGMENT_MODE, prepare_only=False):
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    if not prepare_only and training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有训练记录，请修改RUN_NAME，保留旧结果。')
    train, audit = build_train_dataset(augment_mode)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    validate_groups(train)
    validate_groups(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev_rows = RowDataset(dev, tokenizer, MAX_LENGTH)
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    print('数据模式：', augment_mode, '；Train / Dev：', len(train), '/', len(dev))
    print('训练标签：', train.label.value_counts().to_dict())
    print('Row行 / Pair组 / N单例：', len(rows), '/', len(pairs), '/', int(rows.labels.eq(3).sum()))
    print('每轮：完整Row CE → Dev → 完整Pair CE → Dev。')
    print('Row / Pair更新次数：', row_steps, '/', pair_steps)
    print('最长有效token Train / Dev：', int(rows.encoded['attention_mask'].sum(1).max()),
          '/', int(dev_rows.encoded['attention_mask'].sum(1).max()))
    print('动态padding / 梯度检查点：', DYNAMIC_PADDING, '/', GRADIENT_CHECKPOINTING)
    print('只保存Pair后Dev pair-aware F1最高的模型；N单例仍用模型自身预测。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据与编码检查通过；没有加载模型权重或训练。')
        return

    # 从相同NLI初始化重新训练；不在Dev选出的0.938权重上续训。
    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    if GRADIENT_CHECKPOINTING:
        model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True, collate_fn=collate_rows)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)
    dev_loader = DataLoader(dev_rows, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
                            generator=torch.Generator().manual_seed(SEED + 1))
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
                              generator=torch.Generator().manual_seed(SEED + 2))
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                         label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    train.iloc[audit['baseline_rows']:].to_csv(output_dir / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    (output_dir / 'semantic_contrast_examples.csv').write_bytes(SOURCE_PATH.read_bytes())
    for name in ['manual_label_notes.csv', 'train_word_sources.csv', 'copy_control_sources.csv', 'overlap_report.json']:
        (output_dir / name).write_bytes((PREPARED_DIR / name).read_bytes())
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'augment_mode': augment_mode, 'seed': SEED,
        'model_name': MODEL_NAME, 'base_revision': BASE_REVISION,
        'architecture': 'original CLS four-class logits + Token Alignment residual',
        'full_finetuning': True, 'use_lexical_features': False, 'use_lora': False,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'train_rows': len(train), 'real_pairs': len(pairs), 'train_sha256': frame_hash(train),
        'dev_sha256': file_hash(DEV_PATH), 'reference_run': str(REFERENCE_DIR),
        'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_order': 'all Row CE; evaluate; all original 0.5 * three-way Pair CE; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties',
        'inference': 'original FB/BF/EE decoding; singleton kept as model argmax',
        'augmentation': audit,
        'comparison_note': 'Dynamic padding changes dropout trajectory versus historical0.938. Same-code baseline/copy modes are provided.',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), SOURCE_PATH, HERE / 'prepare_semantic_contrasts.py',
            ROOT / 'experiments/deberta_lexical/model.py', ROOT / 'experiments/deberta_lexical/evaluation.py',
        ]},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    history, stages, changes = [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        row_dir = output_dir / f'epoch_{epoch}/after_row'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        pair_dir = output_dir / f'epoch_{epoch}/after_pair'
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        assert row_loss['row_ce_rows'] == len(rows) and pair_loss['pair_ce_groups'] == len(pairs)
        print('本轮损失：', dict(row_loss, **pair_loss))
        for phase, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, phase=phase, **row_loss, **pair_loss)
                stages.append(record)
                if phase == 'after_pair':
                    history.append(record.copy())
        pd.DataFrame(stages).to_csv(output_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        changes.extend(stage_changes(epoch, 'dev', row_dir, pair_dir))
        pd.DataFrame(changes).to_csv(output_dir / 'stage_changes.csv', index=False, encoding='utf-8-sig')
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected = {'epoch': epoch, 'phase': 'after_pair', 'pair_aware_f1': score,
                        'independent_f1': float(pair_summary.loc[pair_summary.method.eq('independent'), 'weighted_f1'].iloc[0])}
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新最佳模型：', selected)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_weights(model, model_dir / 'best_model.pth')
    evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    selected_predictions = pd.read_csv(output_dir / f'epoch_{best_epoch}/after_pair/predictions_with_scores.csv')
    reloaded = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert selected_predictions.instance_id.tolist() == reloaded.instance_id.tolist()
    assert selected_predictions[['prediction', 'pair_prediction']].equals(reloaded[['prediction', 'pair_prediction']])
    print('\n最佳轮次：', best_epoch, '；pair-aware F1：', best_score)
    print('最佳完整模型：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--mode', choices=['manual', 'copy', 'baseline'], default=AUGMENT_MODE)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.mode, args.prepare_only)
