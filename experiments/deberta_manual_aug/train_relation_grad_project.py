"""右键先做无更新诊断；MODE切换为mixed/project后，由你启动训练。"""
import argparse
import hashlib
import json
import sys
from itertools import islice
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
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset, collate_groups
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, write_train_reference
from experiments.deberta_manual_aug.row_priority_gradients import (
    separate_gradients, measure_gradients, projection_coefficient, assign_gradients,
)


# diagnose：只读0.938权重、计算Train梯度，不创建优化器、不更新参数。
# mixed：普通混合对照；project：同一混合训练，只对冲突Pair梯度做投影。
MODE = 'project'
RUN_NAME = 'run13_gradient'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

DIAGNOSTIC_BATCHES = 24
DIAGNOSTIC_DROPOUT = True
DIAGNOSTIC_WEIGHT = Path(
    'D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged/best_model.pth'
)

# 两组都按同一规则选一个checkpoint；目标只用于选模，不参与训练损失。
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94
LABELS = ['F', 'B', 'E', 'N']
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / 'outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_train():
    # 固定使用0.938的3290行，不重新构造数据；诊断模式不读取Dev样本。
    reference = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    train = pd.read_csv(REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    digest = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert digest == reference['augmentation']['effective_train_sha256']
    assert len(train) == 3290
    validate_groups(train)
    return train, reference, digest


def compute_losses(logits, batch, row_criterion, pair_criterion):
    labels = batch['labels'].to(logits.device)
    row_loss = row_criterion(logits, labels)
    pair_loss = None
    if len(batch['pair_targets']) > 0:
        indices = batch['pair_indices'].to(logits.device)
        pair_scores = get_pair_logits(logits[indices[:, 0]], logits[indices[:, 1]])
        pair_loss = pair_criterion(pair_scores, batch['pair_targets'].to(logits.device))
    return row_loss, pair_loss


def write_gradient_reports(batch_records, layer_records, output_dir):
    batches = pd.DataFrame(batch_records)
    layers = pd.DataFrame(layer_records)
    batches.to_csv(output_dir / 'gradient_batches.csv', index=False, encoding='utf-8-sig')
    layers.to_csv(output_dir / 'gradient_layers.csv', index=False, encoding='utf-8-sig')
    records = []
    combined = pd.concat([batches, layers], ignore_index=True)
    for (epoch, group), frame in combined.groupby(['epoch', 'group'], sort=False):
        valid = frame[frame.has_both_gradients]
        records.append({
            'epoch': epoch, 'group': group, 'batches': len(frame),
            'batches_with_both_gradients': len(valid),
            'conflict_rate': float(valid.conflict.mean()) if len(valid) else None,
            'mean_cosine': float(valid.cosine.mean()) if len(valid) else None,
            'mean_row_grad_norm': float(frame.row_grad_norm.mean()),
            'mean_weighted_pair_grad_norm': float(frame.weighted_pair_grad_norm.mean()),
            'mean_pair_to_row_norm_ratio': float(valid.pair_to_row_norm_ratio.mean()) if len(valid) else None,
        })
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / 'gradient_summary.csv', index=False, encoding='utf-8-sig')
    print('\n本轮梯度统计：')
    print(summary[summary.epoch.eq(batches.epoch.max())].to_string(index=False))


def run_batches(model, loader, train, row_criterion, pair_criterion, device,
                mode, epoch, batch_records, layer_records, diagnostic_batches, optimizer=None):
    diagnostic = mode == 'diagnose'
    model.train(DIAGNOSTIC_DROPOUT if diagnostic else True)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    totals = {'row_loss': 0.0, 'pair_loss': 0.0, 'row_ce_rows': 0, 'pair_ce_groups': 0}
    sequence = islice(loader, diagnostic_batches) if diagnostic else loader
    steps = min(diagnostic_batches, len(loader)) if diagnostic else len(loader)
    for step, batch in enumerate(tqdm(sequence, total=steps, desc=f'{mode} Row + Pair'), start=1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        row_loss, pair_loss = compute_losses(logits, batch, row_criterion, pair_criterion)
        row_gradients, pair_gradients = separate_gradients(
            row_loss, pair_loss, parameters, PAIR_WEIGHT,
        )
        measured = measure_gradients(names, row_gradients, pair_gradients)
        global_record = measured[0]
        candidate = projection_coefficient(global_record)
        coefficient = candidate if mode == 'project' else 0.0
        global_record.update({
            'epoch': epoch, 'step': step, 'mode': mode,
            'row_loss': row_loss.item(), 'pair_loss': pair_loss.item() if pair_loss is not None else 0.0,
            'projection_coefficient_if_enabled': candidate,
            'projection_applied': mode == 'project' and global_record['conflict'],
            'applied_coefficient': coefficient,
            'row_dot_pair_if_projected': global_record['row_pair_dot']
                - candidate * global_record['row_grad_norm'] ** 2,
            # 这是与合成梯度的内积，不是AdamW实际参数步的内积。
            'row_dot_assigned_gradient': None if diagnostic else
                global_record['row_grad_norm'] ** 2 + global_record['row_pair_dot']
                - coefficient * global_record['row_grad_norm'] ** 2,
            'source_instance_ids': '|'.join(train.iloc[batch['source_rows'].tolist()].instance_id),
        })
        counts = torch.bincount(batch['labels'], minlength=4).tolist()
        global_record.update({f'{label}_rows': count for label, count in zip(LABELS, counts)})
        batch_records.append(global_record)
        for record in measured[1:]:
            record.update(epoch=epoch, step=step, mode=mode)
            layer_records.append(record)

        if not diagnostic:
            assign_gradients(parameters, row_gradients, pair_gradients, coefficient)
            optimizer.step()
        totals['row_loss'] += global_record['row_loss']
        totals['pair_loss'] += global_record['pair_loss']
        totals['row_ce_rows'] += len(batch['labels'])
        totals['pair_ce_groups'] += len(batch['pair_targets'])
        # 释放两份梯度及旧图，避免下一批前向仍持有上一批显存。
        del row_gradients, pair_gradients, logits, row_loss, pair_loss
    totals['row_loss'] /= steps
    totals['pair_loss'] /= steps
    totals['optimizer_steps'] = 0 if diagnostic else steps
    return totals


def selection_record(summary, epoch):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    return {
        'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': paired,
        'selection_score': min(independent / TARGET_INDEPENDENT, paired / TARGET_PAIR),
    }


def main(mode=MODE, run_name=RUN_NAME, prepare_only=False, diagnostic_batches=DIAGNOSTIC_BATCHES):
    if mode not in ['diagnose', 'mixed', 'project']:
        raise ValueError('MODE只能是diagnose、mixed或project。')
    if diagnostic_batches < 1:
        raise ValueError('DIAGNOSTIC_BATCHES至少为1。')
    output_dir = HERE / 'outputs' / f'{run_name}_{mode}'
    model_dir = MODEL_ROOT / f'{run_name}_{mode}'
    train, reference, train_hash = read_train()
    groups = train.groupby('pair_id').size()
    print('模式：', mode)
    print('训练数据：', len(train), '行；Pair组：', int(groups.eq(2).sum()), '；N单例：', int(groups.eq(1).sum()))
    print('结果目录：', output_dir)
    if prepare_only:
        print('只检查数据和设置，没有加载模型、没有写模型、没有训练。')
        return
    if (output_dir / 'run_config.json').exists() or (mode != 'diagnose' and model_dir.exists()):
        raise FileExistsError('此RUN_NAME和MODE已有记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    if mode == 'diagnose':
        weights = torch.load(DIAGNOSTIC_WEIGHT, map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights
        print('只读诊断权重：', DIAGNOSTIC_WEIGHT)
    else:
        print('从相同NLI初始化重新训练，不加载0.938权重续训。')
    model.to(device)
    dataset = GroupDataset(train, tokenizer, MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(SEED + 30),
    )
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ['training_data.csv', 'data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (output_dir / 'reference_run_config.json').write_bytes((REFERENCE_DIR / 'run_config.json').read_bytes())
    source_paths = [Path(__file__), HERE / 'row_priority_gradients.py',
                    ROOT / 'experiments/deberta_lexical/model.py',
                    ROOT / 'experiments/deberta_joint_pair/data.py']
    config = {
        'run_name': run_name, 'mode': mode, 'seed': SEED, 'model_name': MODEL_NAME,
        'train_rows': len(train), 'train_sha256': train_hash,
        'training_data_file_sha256': file_hash(REFERENCE_DIR / 'training_data.csv'),
        'reference_config_sha256': file_hash(REFERENCE_DIR / 'run_config.json'),
        'epochs': EPOCHS, 'group_batch_size': GROUP_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING, 'dtype': 'float32',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'freeze_first_n_layers': 0, 'use_lexical_features': False,
        'steps_per_training_epoch': len(loader),
        'diagnostic_batches': diagnostic_batches, 'diagnostic_dropout': DIAGNOSTIC_DROPOUT,
        'diagnostic_weight': str(DIAGNOSTIC_WEIGHT) if mode == 'diagnose' else None,
        'diagnostic_weight_sha256': file_hash(DIAGNOSTIC_WEIGHT) if mode == 'diagnose' else None,
        'selection': 'max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties',
        'target_independent': TARGET_INDEPENDENT, 'target_pair': TARGET_PAIR,
        'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
        'gradient_rule': 'global asymmetric projection: gp <- gp - min(gr dot gp / ||gr||^2, 0)*gr',
        'optimizer_note': 'Projection is on loss gradients, not the AdamW-preconditioned parameter step.',
        'ablation_note': 'mixed/project both use two autograd.grad calls, one forward, identical batch order and optimizer settings.',
        'diagnostic_scope': 'Train only at a fixed checkpoint; local evidence, not a reconstruction of the original trajectory.',
    }
    (output_dir / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    batch_records, layer_records = [], []

    if mode == 'diagnose':
        versions = [p._version for p in model.parameters()]
        run_batches(model, loader, train, row_criterion, pair_criterion, device,
                    mode, 0, batch_records, layer_records, diagnostic_batches)
        assert versions == [p._version for p in model.parameters()]
        assert all(p.grad is None for p in model.parameters())
        write_gradient_reports(batch_records, layer_records, output_dir)
        (output_dir / 'diagnostic_complete.json').write_text(json.dumps({
            'parameter_versions_unchanged': True, 'optimizer_created': False,
            'optimizer_steps': 0, 'batches': len(batch_records), 'dev_read': False,
        }, indent=2), encoding='utf-8')
        print('\n诊断完成：没有创建优化器、没有更新参数，没有评估或训练Dev。')
        return

    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    assert file_hash(DEV_PATH) == reference['dev_sha256']
    validate_groups(dev)
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    train_loader = DataLoader(
        dataset.rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 2),
    )
    write_train_reference(train, output_dir / 'train_reference.csv')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model_dir.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    model.deberta_model.config.save_pretrained(model_dir / 'backbone_config')
    (model_dir / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('模型目录：', model_dir)
    print('每轮步数：', len(loader), '；每个训练行每轮出现一次。')
    print('只保存一个最佳模型，两项F1均取同一轮。')
    history, selections = [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        losses = run_batches(model, loader, train, row_criterion, pair_criterion, device,
                             mode, epoch, batch_records, layer_records, diagnostic_batches, optimizer)
        assert losses['row_ce_rows'] == len(train) and losses['pair_ce_groups'] == dataset.pair_count
        write_gradient_reports(batch_records, layer_records, output_dir)
        print('本轮损失及曝光：', losses)
        summary = evaluate_phase(model, dev_loader, dev, device, output_dir / f'epoch_{epoch}', GOLD_PATH)
        for record in summary.to_dict('records'):
            record.update(epoch=epoch, **losses)
            history.append(record)
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        selected = selection_record(summary, epoch)
        selections.append(selected)
        pd.DataFrame(selections).to_csv(output_dir / 'selection_history.csv', index=False, encoding='utf-8-sig')
        if selected['selection_score'] > best_score:
            best_score, best_epoch = selected['selection_score'], epoch
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新同轮最佳：', selected)

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    print('\n最佳轮次：', best_epoch)
    print('最佳权重：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['diagnose', 'mixed', 'project'], default=MODE)
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--diagnostic-batches', type=int, default=DIAGNOSTIC_BATCHES)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.mode, args.run_name, args.prepare_only, args.diagnostic_batches)
