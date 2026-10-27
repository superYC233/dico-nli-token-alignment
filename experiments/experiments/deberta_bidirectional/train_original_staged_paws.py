"""右键运行：原0.938模型与Row→Pair安排，仅Row阶段加入PAWS。"""
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
from torch.utils.data import DataLoader
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
from experiments.deberta_bidirectional.train_deberta_alignment_joint import collate_original_units
from experiments.deberta_bidirectional.prepare_paws_data import prepare, PREPARED_DIR
from experiments.deberta_bidirectional.paws_auxiliary import PawsDataset, equivalence_loss, interleaved_batches


RUN_NAME = 'run04_original_staged_paws_seed42_EPOCH15'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
EPOCHS = 15
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

PAWS_WEIGHT = 0.1
PAWS_EVERY = 4
PAWS_UNIT_BATCH_SIZE = 8
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_bidirectional')


def build_model(tokenizer, model_name=MODEL_NAME, revision=BASE_REVISION):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    # 真正复用原0.938的四类模型，没有双向联合头、LoRA或新增分类层。
    return DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)


def paws_loss_original(model, batch, device):
    # 原模型逐行分类，因此PAWS的AB/BA分别算二元CE再平均。
    # 两方向原始四类logits不相加，也不使用联合头的反转映射。
    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
    units = batch['row_units'].to(device)
    labels = batch['labels'].to(device)
    first = equivalence_loss(logits[2 * units], labels)
    second = equivalence_loss(logits[2 * units + 1], labels)
    return (first + second) / 2


def train_rows(model, loader, paws_loader, optimizer, criterion, device,
               paws_weight=PAWS_WEIGHT, paws_every=PAWS_EVERY):
    model.train()
    started = time.perf_counter()
    row_sum, row_weights, auxiliary_sum = 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    paws_counts = torch.zeros(2, dtype=torch.long)
    exposure = []
    sequence = interleaved_batches(loader, paws_loader, paws_every, paws_weight > 0)
    for step, batch, auxiliary in tqdm(sequence, total=len(loader), desc='原版Row CE + PAWS'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        labels = batch['label'].to(device)
        loss = criterion(logits, labels)
        value = loss.item()
        weight = criterion.weight[labels].sum().item()
        loss.backward()
        row_sum += value * weight
        row_weights += weight
        counts += torch.bincount(batch['label'], minlength=4)
        del logits, labels, loss
        if auxiliary is not None:
            auxiliary_loss = paws_loss_original(model, auxiliary, device)
            (paws_weight * auxiliary_loss).backward()
            auxiliary_sum += auxiliary_loss.item() * len(auxiliary['labels'])
            paws_counts += torch.bincount(auxiliary['labels'], minlength=2)
            exposure.append({'row_step': step, 'paws_source_ids': '|'.join(
                str(value) for value in paws_loader.dataset.frame.iloc[
                    auxiliary['source_rows'].tolist()
                ].source_id.tolist()
            ), 'paws_equivalent': int(auxiliary['labels'].eq(1).sum()),
                'paws_not_equivalent': int(auxiliary['labels'].eq(0).sum())})
            del auxiliary_loss
        optimizer.step()
    paws_rows = int(paws_counts.sum())
    result = {'row_loss': row_sum / row_weights, 'row_ce_rows': int(counts.sum()),
              'row_optimizer_steps': len(loader), 'row_train_seconds': time.perf_counter() - started,
              'paws_loss': auxiliary_sum / paws_rows if paws_rows else 0.0,
              'paws_rows': paws_rows, 'paws_batches': len(exposure),
              'paws_equivalent_rows': int(paws_counts[1]), 'paws_not_equivalent_rows': int(paws_counts[0]),
              'paws_weight': paws_weight, 'learning_rate': optimizer.param_groups[0]['lr']}
    result.update({f'seen_{label}': count for label, count in zip(LABELS, counts.tolist())})
    return result, exposure


def train_pairs(model, loader, optimizer, criterion, device, pair_weight=PAIR_WEIGHT):
    # 原Pair CE原样保留：先正向/反向各前向，再FB/BF/EE三组合CE。
    model.train()
    started = time.perf_counter()
    total, groups = 0.0, 0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='原版Pair CE（无PAWS）'):
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        loss = criterion(get_pair_logits(first, second), targets.to(device))
        (pair_weight * loss).backward()
        optimizer.step()
        total += loss.item()
        groups += len(targets)
    return {'pair_loss': total / len(loader), 'pair_ce_groups': groups,
            'pair_optimizer_steps': len(loader), 'pair_train_seconds': time.perf_counter() - started}


def source_hashes():
    paths = [HERE / name for name in ['train_original_staged_paws.py', 'prepare_paws_data.py',
             'paws_auxiliary.py', 'data.py', 'train_deberta_bidirectional.py',
             'train_deberta_alignment_joint.py', 'alignment_joint_model.py']]
    paths.extend([ROOT / 'experiments/deberta_lexical/model.py',
                  ROOT / 'experiments/deberta_lexical/evaluation.py',
                  ROOT / 'experiments/deberta_joint_pair/data.py',
                  ROOT / 'experiments/deberta_manual_aug/train_boundary_manual.py',
                  ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py'])
    paths.extend(sorted(Path(evaluation_functions.__file__).resolve().parent.glob('*.py')))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, prepare_only=False):
    hashes = source_hashes()
    train, dev, train_hash = read_data()
    paws, manifest = prepare(PREPARED_DIR)
    assert manifest['request']['model_name'] == MODEL_NAME
    assert manifest['request']['base_revision'] == BASE_REVISION
    assert manifest['request']['max_length'] == MAX_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    paws_dataset = PawsDataset(paws, tokenizer, MAX_LENGTH)
    dev_dataset = RowDataset(dev, tokenizer, MAX_LENGTH)
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    auxiliary_batches = row_steps // PAWS_EVERY if PAWS_WEIGHT > 0 else 0
    print('模型：原0.938的CLS四类分支 + Token Alignment四类残差；不是联合头。')
    print('训练顺序：Row CE + 少量PAWS → 评估 → 原Pair CE → 再评估。')
    print('Train / Dev：', len(train), '/', len(dev), '；真实Pair组：', len(pairs))
    print('每轮Row / Pair更新：', row_steps, '/', pair_steps, '；合计：', row_steps + pair_steps)
    print('PAWS池：', len(paws), '；每轮Row阶段约看：', auxiliary_batches * PAWS_UNIT_BATCH_SIZE)
    print('PAWS权重 / 间隔：', PAWS_WEIGHT, '/', PAWS_EVERY, '；Pair阶段不加PAWS。')
    print('seed / 学习率 / 轮次：', SEED, '/', LEARNING_RATE, '/', EPOCHS)
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据、分词和源码路径检查完成，没有加载模型或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    print('全部可训练参数：', sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    # 与原0.938一样，Row/Pair shuffle使用全局torch RNG；PAWS sampler独立。
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    paws_loader = DataLoader(
        paws_dataset, batch_size=PAWS_UNIT_BATCH_SIZE, shuffle=True, collate_fn=collate_original_units,
        generator=torch.Generator().manual_seed(SEED + 100),
    )
    dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False,
                            generator=torch.Generator().manual_seed(SEED + 1))
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
                              generator=torch.Generator().manual_seed(SEED + 2))
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    for source, destination in [('paws_train.csv', 'paws_train.csv'),
                                ('preparation.json', 'paws_preparation.json'),
                                ('overlap_report.json', 'paws_overlap_report.json')]:
        (output_dir / destination).write_bytes((PREPARED_DIR / source).read_bytes())
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment four-class residual',
        'full_finetuning': True, 'use_lora': False, 'use_joint_head': False,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': False,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'train_rows': len(train), 'real_pairs': len(pairs), 'train_sha256': train_hash,
        'dev_sha256': file_hash(DEV_PATH), 'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'optimizer_steps_per_epoch': row_steps + pair_steps,
        'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'paws_weight': PAWS_WEIGHT, 'paws_every': PAWS_EVERY, 'paws_unit_batch_size': PAWS_UNIT_BATCH_SIZE,
        'paws_pool_rows': len(paws), 'paws_selected_sha256': manifest['selected_sha256'],
        'paws_manifest_sha256': file_hash(PREPARED_DIR / 'preparation.json'),
        'paws_batches_per_epoch': auxiliary_batches,
        'paws_supervision': 'average binary E-vs-nonE CE over both PAWS input orders; nonE is not N',
        'training_order': 'all Row CE updates with PAWS every fourth update; evaluate; all original Pair CE updates; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware F1; earliest on ties; after-row never used to select checkpoint',
        'checkpoint_format': 'original_alignment_full_v1', 'reference_run': str(REFERENCE_DIR),
        'source_sha256': hashes,
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    history, stages, changes, exposure_history, selections = [], [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses, exposure = train_rows(model, row_loader, paws_loader, optimizer, row_criterion, device)
        assert row_losses['row_ce_rows'] == len(train) and row_losses['paws_batches'] == auxiliary_batches
        print('Row阶段：', row_losses)
        row_dir = output_dir / f'epoch_{epoch}/after_row'
        pair_dir = output_dir / f'epoch_{epoch}/after_pair'
        print('\n===== Row结束，Pair尚未开始 =====')
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
        assert pair_losses['pair_ce_groups'] == len(pairs)
        losses = dict(row_losses, **pair_losses)
        losses['optimizer_steps'] = row_steps + pair_steps
        print('Pair阶段：', pair_losses)
        print('\n===== Pair结束 =====')
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
        for record in exposure:
            record['epoch'] = epoch
        exposure_history.extend(exposure)
        pd.DataFrame(exposure_history).to_csv(output_dir / 'paws_exposure.csv', index=False, encoding='utf-8-sig')
        independent = float(pair_summary.loc[pair_summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        selected = {'epoch': epoch, 'phase': 'after_pair', 'independent_f1': independent,
                    'pair_aware_f1': score, 'selection_score': score}
        selections.append(selected)
        pd.DataFrame(selections).to_csv(output_dir / 'selection_history.csv', index=False, encoding='utf-8-sig')
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新Pair结束后的最佳模型：', selected)

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_weights(model, model_dir / 'best_model.pth')
    evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    old = pd.read_csv(output_dir / f'epoch_{best_epoch}/after_pair/predictions_with_scores.csv')
    new = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert old.instance_id.tolist() == new.instance_id.tolist()
    differences = {column: int(old[column].ne(new[column]).sum())
                   for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values())
    (output_dir / 'reload_check.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    print('\n最佳轮次（Pair后）：', best_epoch)
    print('最佳完整原模型：', model_dir / 'best_model.pth')
    print(pd.read_csv(output_dir / 'best_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
