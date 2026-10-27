"""右键运行：原0.938两阶段配方，只在Row阶段增加SiFT对抗正则。"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import evaluation_functions
from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, stage_changes
from experiments.deberta_bidirectional.data import LABELS, write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights
from experiments.deberta_bidirectional.train_deberta_bidirectional import read_data, file_hash, training_records_exist
from experiments.deberta_bidirectional.sift_regularizer import SiFT, capture_rng


RUN_NAME = 'run06_original_staged_sift_seed42'
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
DYNAMIC_PADDING = True
GRADIENT_CHECKPOINTING = True
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

# 第一轮只试这一组，不同时改原来的学习率、类别权重或数据。
SIFT_WEIGHT = 1.0
SIFT_INIT_PERTURBATION = 1e-2
SIFT_STEP_SIZE = 1e-4
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_bidirectional')


def sift_settings(weight):
    return {'enabled': weight > 0, 'weight': weight,
            'init_perturbation': SIFT_INIT_PERTURBATION, 'step_size': SIFT_STEP_SIZE,
            'noise_seed': SEED + 100, 'normalization_eps': 1e-7}


def build_model(tokenizer, model_name=MODEL_NAME, revision=BASE_REVISION, sift_options=None):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    options = sift_settings(SIFT_WEIGHT) if sift_options is None else sift_options
    model.sift = SiFT(model, **{key: options[key] for key in [
        'init_perturbation', 'step_size', 'noise_seed', 'normalization_eps',
    ]}) if options['enabled'] else None
    return model


def trim_inputs(ids, mask):
    length = int(mask.sum(1).max()) if DYNAMIC_PADDING else ids.shape[1]
    return ids[:, :length].contiguous(), mask[:, :length].contiguous()


def collate_rows(examples):
    batch = default_collate(examples)
    batch['input_ids'], batch['attention_mask'] = trim_inputs(batch['input_ids'], batch['attention_mask'])
    return batch


def collate_pairs(examples):
    ids0, mask0, ids1, mask1, targets = default_collate(examples)
    ids0, mask0 = trim_inputs(ids0, mask0)
    ids1, mask1 = trim_inputs(ids1, mask1)
    return ids0, mask0, ids1, mask1, targets


def train_rows(model, loader, optimizer, criterion, device, sift_weight=SIFT_WEIGHT):
    model.train()
    started = time.perf_counter()
    total, total_weight, adv_total = 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    diagnostics = []
    for step, batch in enumerate(tqdm(loader, desc='原Row CE + SiFT'), 1):
        optimizer.zero_grad(set_to_none=True)
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        state = capture_rng(device) if sift_weight > 0 else None
        logits = model(ids, mask, None)
        row_loss = criterion(logits, labels)
        if sift_weight > 0:
            adv_loss = model.sift.loss(logits, lambda: model(ids, mask, None), mask, state)
        else:
            adv_loss = logits.new_zeros(())
        loss = row_loss + sift_weight * adv_loss
        loss.backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += row_loss.item() * weight
        total_weight += weight
        adv_total += adv_loss.item() * len(labels)
        counts += torch.bincount(batch['label'], minlength=4)
        record = {'step': step, 'rows': len(labels), 'tokens': ids.shape[1],
                  'row_loss': row_loss.item(), 'sift_loss': adv_loss.item(),
                  'weighted_sift_loss': sift_weight * adv_loss.item(), 'loss': loss.item()}
        if sift_weight > 0:
            record.update(model.sift.last_stats)
        diagnostics.append(record)
        del logits, row_loss, adv_loss, loss
    result = {'row_loss': total / total_weight, 'sift_loss': adv_total / int(counts.sum()),
              'sift_weight': sift_weight, 'row_ce_rows': int(counts.sum()),
              'row_optimizer_steps': len(loader), 'row_train_seconds': time.perf_counter() - started}
    result.update({f'seen_{label}': int(count) for label, count in zip(LABELS, counts)})
    return result, diagnostics


def train_pairs(model, loader, optimizer, criterion, device):
    # 原三组合Pair阶段：不加SiFT扰动、Row回放或NN候选。
    model.train()
    started = time.perf_counter()
    total, groups = 0.0, 0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='原版Pair CE（无对抗扰动）'):
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * loss).backward()
        optimizer.step()
        total += loss.item()
        groups += len(targets)
        if model.sift is not None:
            model.sift.clear()
    return {'pair_loss': total / len(loader), 'pair_ce_groups': groups,
            'pair_optimizer_steps': len(loader), 'pair_train_seconds': time.perf_counter() - started}


def source_hashes():
    paths = [Path(__file__).resolve(), HERE / 'sift_regularizer.py', HERE / 'SIFT_LICENSE.txt',
             HERE / 'train_deberta_bidirectional.py', HERE / 'alignment_joint_model.py', HERE / 'data.py',
             ROOT / 'experiments/deberta_joint_pair/data.py', ROOT / 'experiments/deberta_lexical/model.py',
             ROOT / 'experiments/deberta_lexical/evaluation.py',
             ROOT / 'experiments/deberta_manual_aug/train_boundary_manual.py',
             ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py']
    paths.extend(sorted(Path(evaluation_functions.__file__).resolve().parent.glob('*.py')))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'], config['sift'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, prepare_only=False, sift_weight=SIFT_WEIGHT):
    hashes = source_hashes()
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev_rows = RowDataset(dev, tokenizer, MAX_LENGTH)
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    options = sift_settings(sift_weight)
    print('原0.938模型和两阶段训练：Row CE + SiFT → 评估 → 原Pair CE → 评估。')
    print('Train / Dev：', len(train), '/', len(dev), '；真实Pair组：', len(pairs))
    print('Row / Pair更新：', row_steps, '/', pair_steps, '；合计：', row_steps + pair_steps)
    print('SiFT配置：', options)
    print('每轮正式编码行：6050；SiFT额外搜索/扰动前向只用于Row阶段。')
    print('动态padding / 梯度检查点：', DYNAMIC_PADDING, '/', GRADIENT_CHECKPOINTING)
    print('Train / Dev最长有效token：', int(rows.encoded['attention_mask'].sum(1).max()),
          '/', int(dev_rows.encoded['attention_mask'].sum(1).max()))
    print('seed / 学习率 / 轮次：', SEED, '/', LEARNING_RATE, '/', EPOCHS)
    print('选模：Pair后Dev pair-aware F1最高；单例仍由模型四分类。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据和路径检查完成，没有加载模型或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer, sift_options=options).to(device)
    if GRADIENT_CHECKPOINTING:
        model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('全部可训练参数：', parameter_count)
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
    write_train_reference(train, output_dir / 'train_reference.csv')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment four-class residual',
        'full_finetuning': True, 'use_lora': False, 'use_joint_head': False,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'train_rows': len(train), 'real_pairs': len(pairs), 'train_sha256': train_hash,
        'dev_sha256': file_hash(DEV_PATH), 'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'optimizer_steps_per_epoch': row_steps + pair_steps, 'trainable_parameters': parameter_count,
        'sift': options,
        'sift_source': 'https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/sift/sift.py',
        'sift_adaptations': 'mask padding; pair dropout RNG across three forwards; autograd.grad for delta; subtract detached entropy constants from official stop-target bidirectional CE',
        'normalization': 'when enabled, parameter-free pre-embedding-LayerNorm normalization persists in clean/Pair/eval/reload',
        'training_order': 'all Row CE + SiFT; evaluate; all original three-combination Pair CE; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware F1; earliest on ties',
        'inference': 'no perturbation; original FB/BF/EE decoding; singleton kept as model argmax',
        'checkpoint_format': 'original_alignment_full_v1', 'reference_run': str(REFERENCE_DIR),
        'comparison_note': 'dynamic padding changes dropout trajectory; SiFT normalization also persists at inference; not a byte-identical 0.938 rerun',
        'source_sha256': hashes,
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    history, stages, changes, adv_history = [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses, diagnostics = train_rows(model, row_loader, optimizer, row_criterion, device, sift_weight)
        print('Row阶段：', row_losses)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        print('\n===== Row+SiFT结束，Pair尚未开始 =====')
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        assert row_losses['row_ce_rows'] == len(train) and pair_losses['pair_ce_groups'] == len(pairs)
        losses = dict(row_losses, **pair_losses, optimizer_steps=row_steps + pair_steps)
        print('Pair阶段：', pair_losses)
        print('\n===== 原Pair CE结束 =====')
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        for phase, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, phase=phase, **losses)
                stages.append(record)
                if phase == 'after_pair':
                    history.append(record.copy())
        pd.DataFrame(stages).to_csv(output_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        changes.extend(stage_changes(epoch, 'dev', row_dir, pair_dir))
        pd.DataFrame(changes).to_csv(output_dir / 'stage_changes.csv', index=False, encoding='utf-8-sig')
        for record in diagnostics:
            record['epoch'] = epoch
        adv_history.extend(diagnostics)
        pd.DataFrame(adv_history).to_csv(output_dir / 'sift_batches.csv', index=False, encoding='utf-8-sig')
        independent = float(pair_summary.loc[pair_summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected = {'epoch': epoch, 'phase': 'after_pair', 'independent_f1': independent,
                        'pair_aware_f1': score, 'selection_score': score}
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新Pair后最佳模型：', selected)

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_weights(model, model_dir / 'best_model.pth')
    evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    old = pd.read_csv(output_dir / f'epoch_{best_epoch}/after_pair/predictions_with_scores.csv')
    new = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert old.instance_id.tolist() == new.instance_id.tolist()
    differences = {column: int(old[column].ne(new[column]).sum()) for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values())
    (output_dir / 'reload_check.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    print('\n最佳轮次（Pair后）：', best_epoch)
    print('最佳完整模型：', model_dir / 'best_model.pth')
    print(pd.read_csv(output_dir / 'best_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--sift-weight', type=float, default=SIFT_WEIGHT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only, args.sift_weight)
