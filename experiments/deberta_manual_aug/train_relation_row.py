"""共用训练代码：两种架构都只做四分类Row CE，不计算Pair CE。"""
import hashlib
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
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset, collate_groups
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_manual_aug.train_boundary_manual import set_seed
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, write_train_reference
from experiments.deberta_manual_aug.train_relation_grad_project import read_train, file_hash
from experiments.deberta_manual_aug.relation_lora_model import (
    parameter_report, save_trainable_weights, load_trainable_weights,
)
from experiments.deberta_manual_aug.relation_row_model import build_model


# 两个入口共用这些参数；对照已完成的run19，不叠加新学习率调度。
SEED = 7
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ['query_proj', 'value_proj']
EPOCHS = 20
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94

CLS_RUN = 'run22_cls_row_seed7'
ALIGNMENT_RUN = 'run23_alignment_row_seed7'
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / 'outputs/run06_relation48_staged'
MIXED_DIR = HERE / 'outputs/run19_lora_r16_seed7_dropout01'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']


def lora_settings():
    return {
        'r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT,
        'target_modules': LORA_TARGET_MODULES, 'bias': 'none',
        'init_lora_weights': True, 'use_rslora': False,
    }


def shared_initialization_hash(model):
    # 记录两组共有的LoRA和CLS初值；冻结底座另由base_revision固定。
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if 'lora_A.' in name or 'lora_B.' in name or name.startswith('output.'):
            digest.update(name.encode('utf-8'))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    started = time.perf_counter()
    totals = {'row_loss': 0.0, 'row_ce_rows': 0, 'pair_ce_groups': 0}
    counts = torch.zeros(4, dtype=torch.long)
    plan = []
    for step, batch in enumerate(tqdm(loader, desc=f'{model.head_type}：只有四类Row CE'), start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        # Pair的索引和标签不参与损失；分组只为保持与mixed相同的批次。
        loss = criterion(logits, batch['labels'].to(device))
        loss.backward()
        optimizer.step()
        totals['row_loss'] += loss.item()
        totals['row_ce_rows'] += len(batch['labels'])
        counts += torch.bincount(batch['labels'], minlength=4)
        plan.append({'epoch': epoch, 'step': step, 'source_rows': '|'.join(
            str(index) for index in batch['source_rows'].tolist()
        )})
        del logits, loss
    totals['row_loss'] /= len(loader)
    totals['optimizer_steps'] = len(loader)
    totals['learning_rate'] = optimizer.param_groups[0]['lr']
    totals['train_seconds'] = time.perf_counter() - started
    totals.update({f'seen_{label}': count for label, count in zip(LABELS, counts.tolist())})
    return totals, plan


def selection_record(summary, epoch):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    return {'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': paired,
            'selection_score': min(independent / TARGET_INDEPENDENT, paired / TARGET_PAIR)}


def compare_experiments(output_dir):
    records = []
    for name, folder in [('mixed：CLS+Alignment+Pair CE', MIXED_DIR),
                         ('CLS+Row CE', HERE / 'outputs' / CLS_RUN),
                         ('CLS+Alignment+Row CE', HERE / 'outputs' / ALIGNMENT_RUN)]:
        if not (folder / 'best_summary.csv').exists():
            continue
        config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        best = json.loads((folder / 'best_metrics.json').read_text(encoding='utf-8'))
        for row in pd.read_csv(folder / 'best_summary.csv').to_dict('records'):
            records.append({'experiment': name, 'run_name': config['run_name'],
                            'seed': config['seed'], 'epoch': best['epoch'],
                            'method': row['method'], 'weighted_f1': row['weighted_f1'],
                            'N_correct': row['N_correct'], 'F_correct': row['F_correct'],
                            'B_correct': row['B_correct'], 'E_correct': row['E_correct'],
                            'nonN_to_N': row.get('nonN_to_N')})
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir / 'architecture_comparison.csv', index=False, encoding='utf-8-sig')
    print('\n已完成实验对比：每个实验的两项成绩来自同一选中轮次。')
    print(comparison.to_string(index=False))


def main(run_name, head_type, prepare_only=False):
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    train, reference, train_hash = read_train()
    print('架构：', 'CLS' if head_type == 'cls' else 'CLS + Token Alignment')
    print('训练损失：只有四类Row CE；Pair CE = 0。')
    print('Train：', len(train), '；seed：', SEED, '；r/alpha/dropout：', LORA_R, LORA_ALPHA, LORA_DROPOUT)
    print('固定学习率 / 轮次：', LEARNING_RATE, '/', EPOCHS)
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('只检查同一3290行数据和配置，没有加载模型或训练。')
        return
    if model_dir.exists() or (output_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有模型或记录，请改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    mixed_config = json.loads((MIXED_DIR / 'run_config.json').read_text(encoding='utf-8'))
    revision = mixed_config['base_revision']
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=revision, local_files_only=True,
    )
    settings = lora_settings()
    model = build_model(MODEL_NAME, tokenizer, head_type, settings, revision=revision)
    initialization_hash = shared_initialization_hash(model)
    model.to(device)
    report = pd.DataFrame(parameter_report(model))
    print('\n参数统计：')
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())

    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    assert file_hash(DEV_PATH) == reference['dev_sha256']
    validate_groups(dev)
    dataset = GroupDataset(train, tokenizer, MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(SEED + 30),
    )
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    train_loader = DataLoader(
        dataset.rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(SEED + 2),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=False)
    for name in ['training_data.csv', 'data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    write_train_reference(train, output_dir / 'train_reference.csv')
    report.to_csv(output_dir / 'parameters.csv', index=False, encoding='utf-8-sig')
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    model.deberta_model.config.save_pretrained(model_dir / 'backbone_config')
    source_paths = [Path(__file__), HERE / 'relation_row_model.py',
                    HERE / f'train_relation_{head_type}_row.py', HERE / 'relation_lora_model.py',
                    ROOT / 'experiments/deberta_lexical/model.py',
                    ROOT / 'experiments/deberta_joint_pair/data.py']
    config = {
        'run_name': run_name, 'head_type': head_type, 'use_alignment': head_type == 'alignment',
        'use_lora': True, 'lora': settings, 'model_name': MODEL_NAME,
        'base_revision': revision, 'shared_initialization_sha256': initialization_hash,
        'seed': SEED, 'train_rows': len(train), 'train_sha256': train_hash,
        'dev_sha256': reference['dev_sha256'], 'epochs': EPOCHS,
        'group_batch_size': GROUP_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'learning_rate': LEARNING_RATE, 'scheduler': None,
        'weight_decay': WEIGHT_DECAY, 'row_weights': ROW_WEIGHTS,
        'pair_weight': 0.0, 'pair_ce_groups_per_epoch': 0,
        'dtype': 'float32', 'steps_per_training_epoch': len(loader),
        'trainable_parameters': int(report.loc[report.trainable, 'parameters'].sum()),
        'total_parameters': int(report.parameters.sum()),
        'training_mode': 'four-class Row CE only; one forward and one backward',
        'batching': 'same shuffled pair/singleton groups as run19; pair metadata never enters loss',
        'selection': 'max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties',
        'target_independent': TARGET_INDEPENDENT, 'target_pair': TARGET_PAIR,
        'checkpoint_format': 'row_architecture_trainable_only_v1',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
        'comparison_reference': str(MIXED_DIR),
        'comparison_note': 'CLS vs alignment changes only the added alignment branch. Alignment Row-only vs run19 removes Pair CE; one combined backward changes floating-point accumulation from the historical two-backward implementation.',
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('每轮步数：', len(loader), '；数据中可逆组：', dataset.pair_count, '；参与Pair CE：0。')
    print('每轮仍报告两种预测，按相同双目标规则只保存一份最佳权重。')

    history, selections, batch_plan = [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        losses, plan = train_epoch(model, loader, optimizer, criterion, device, epoch)
        assert losses['row_ce_rows'] == len(train) and losses['pair_ce_groups'] == 0
        print('本轮损失及曝光：', losses)
        batch_plan.extend(plan)
        pd.DataFrame(batch_plan).to_csv(output_dir / 'batch_plan.csv', index=False, encoding='utf-8-sig')
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
            save_trainable_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新同轮最佳：', selected)

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_trainable_weights(model, model_dir / 'best_model.pth')
    evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    saved = pd.read_csv(output_dir / f'epoch_{best_epoch}/predictions_with_scores.csv')
    reloaded = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert saved.instance_id.tolist() == reloaded.instance_id.tolist()
    differences = {column: int(saved[column].ne(reloaded[column]).sum())
                   for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values()), '最佳权重重载后预测不一致。'
    (output_dir / 'reload_check.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    compare_experiments(output_dir)
    print('\n最佳轮次：', best_epoch)
    print('最佳LoRA及任务头：', model_dir / 'best_model.pth')
