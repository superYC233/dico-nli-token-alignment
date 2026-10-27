"""右键运行：普通mixed训练，只把DeBERTa编码器改为LoRA适配。"""
import argparse
import json
import sys
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
from experiments.deberta_manual_aug.train_relation_grad_project import read_train, file_hash, compute_losses
from experiments.deberta_manual_aug.relation_lora_model import (
    build_model, parameter_report, save_trainable_weights, load_trainable_weights,
)

# 对照run19：学习率从1e-4逐步降到1e-5，同时将相同损失合并为一次反向。
RUN_NAME = 'run21_lora_r16_seed7_dropout01_linear_fast'

USE_LORA = True
SEED = 7
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

LORA_TARGET_MODULES = ['query_proj', 'value_proj']

# 沿用run19的数据、损失、batch和20轮，不是全程固定1e-5。
EPOCHS = 20
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-4
END_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / 'outputs/run06_relation48_staged'
COMPARISON_DIR = HERE / 'outputs/run13_gradient_mixed'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')
LABELS = ['F', 'B', 'E', 'N']


def lora_settings():
    return {
        'r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT,
        'target_modules': LORA_TARGET_MODULES, 'bias': 'none',
        'init_lora_weights': True, 'use_rslora': False,
    }


def train_epoch(model, loader, optimizer, scheduler, row_criterion, pair_criterion, device, epoch):
    model.train()
    totals = {'row_loss': 0.0, 'pair_loss': 0.0, 'row_ce_rows': 0, 'pair_ce_groups': 0}
    totals['learning_rate_start'] = optimizer.param_groups[0]['lr']
    counts = torch.zeros(4, dtype=torch.long)
    plan = []
    for step, batch in enumerate(tqdm(loader, desc='LoRA普通mixed Row + Pair'), start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        row_loss, pair_loss = compute_losses(logits, batch, row_criterion, pair_criterion)
        # 梯度相加等价于损失相加后一次反向，Pair系数只乘一次。
        loss = row_loss
        if pair_loss is not None:
            loss = loss + PAIR_WEIGHT * pair_loss
        loss.backward()
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        totals['row_loss'] += row_loss.item()
        totals['pair_loss'] += pair_loss.item() if pair_loss is not None else 0.0
        totals['row_ce_rows'] += len(batch['labels'])
        totals['pair_ce_groups'] += len(batch['pair_targets'])
        counts += torch.bincount(batch['labels'], minlength=4)
        plan.append({'epoch': epoch, 'step': step, 'source_rows': '|'.join(
            str(index) for index in batch['source_rows'].tolist()
        )})
        del logits, row_loss, pair_loss, loss
    totals['row_loss'] /= len(loader)
    totals['pair_loss'] /= len(loader)
    totals['optimizer_steps'] = len(loader)
    totals['learning_rate_last_step'] = learning_rate
    totals['learning_rate_next_step'] = optimizer.param_groups[0]['lr']
    totals.update({f'seen_{label}': count for label, count in zip(LABELS, counts.tolist())})
    return totals, plan


def selection_record(summary, epoch):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    return {'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': paired,
            'selection_score': min(independent / TARGET_INDEPENDENT, paired / TARGET_PAIR)}


def compare_reference(summary, output_dir):
    old = pd.read_csv(COMPARISON_DIR / 'best_summary.csv')
    records = []
    for method in ['independent', 'pair_aware']:
        before = old[old.method.eq(method)].iloc[0]
        after = summary[summary.method.eq(method)].iloc[0]
        records.append({'method': method, 'full_finetune_f1': before.weighted_f1,
                        'current_f1': after.weighted_f1,
                        'delta_percentage_points': 100 * (after.weighted_f1 - before.weighted_f1),
                        'full_N_correct': before.N_correct, 'current_N_correct': after.N_correct,
                        'full_FBE_correct': before.FBE_correct, 'current_FBE_correct': after.FBE_correct})
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir / 'comparison_with_mixed.csv', index=False, encoding='utf-8-sig')
    print('\n与普通mixed选中模型的同轮指标比较：')
    print(comparison.to_string(index=False))


def main(run_name=RUN_NAME, prepare_only=False):
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    train, reference, train_hash = read_train()
    print('Train：', len(train), '；LoRA：', USE_LORA)
    print('r / alpha / dropout：', LORA_R, '/', LORA_ALPHA, '/', LORA_DROPOUT)
    print('学习率：', LEARNING_RATE, '→', END_LEARNING_RATE, '；每步线性衰减；每批一次反向。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('只检查原3290行数据与设置，没有加载模型或训练。')
        return
    if model_dir.exists() or (output_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有模型或记录，请改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    # 与普通mixed相同的NLI初始化和任务头；LoRA的B初始为0，初始输出不变。
    settings = lora_settings()
    model = build_model(MODEL_NAME, tokenizer, USE_LORA, settings).to(device)
    report = pd.DataFrame(parameter_report(model))
    print('\n参数统计：')
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())
    if USE_LORA:
        assert not report.loc[report.group.eq('original_encoder'), 'trainable'].any()
        assert report.loc[report.group.isin(['lora', 'task_heads']), 'trainable'].all()

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
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(PAIR_WEIGHTS, device=device), label_smoothing=PAIR_LABEL_SMOOTHING,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=END_LEARNING_RATE / LEARNING_RATE,
        total_iters=EPOCHS * len(loader),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=False)
    for name in ['training_data.csv', 'data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (output_dir / 'reference_run_config.json').write_bytes((COMPARISON_DIR / 'run_config.json').read_bytes())
    write_train_reference(train, output_dir / 'train_reference.csv')
    report.to_csv(output_dir / 'parameters.csv', index=False, encoding='utf-8-sig')
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    model.deberta_model.config.save_pretrained(model_dir / 'backbone_config')
    source_paths = [Path(__file__), HERE / 'relation_lora_model.py',
                    HERE / 'train_relation_grad_project.py',
                    ROOT / 'experiments/deberta_lexical/model.py', ROOT / 'experiments/deberta_joint_pair/data.py']
    config = {
        'run_name': run_name, 'use_lora': USE_LORA, 'lora': settings,
        'model_name': MODEL_NAME, 'base_revision': model.deberta_model.config._commit_hash,
        'seed': SEED, 'train_rows': len(train), 'train_sha256': train_hash,
        'dev_sha256': reference['dev_sha256'],
        'epochs': EPOCHS, 'group_batch_size': GROUP_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'end_learning_rate': END_LEARNING_RATE, 'scheduler': 'linear_per_optimizer_step',
        'scheduler_total_steps': EPOCHS * len(loader),
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING, 'dtype': 'float32',
        'steps_per_training_epoch': len(loader), 'trainable_parameters': int(report.loc[report.trainable, 'parameters'].sum()),
        'total_parameters': int(report.parameters.sum()),
        'training_mode': 'ordinary mixed; one forward; one backward on Row CE + weighted Pair CE',
        'gradient_projection': False, 'use_lexical_features': False,
        'selection': 'max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties',
        'target_independent': TARGET_INDEPENDENT, 'target_pair': TARGET_PAIR,
        'checkpoint_format': 'trainable_parameters_only_v1; LoRA A/B plus all task-head parameters',
        'reference_results': str(COMPARISON_DIR),
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
        'comparison_note': 'Compare with run19_lora_r16_seed7_dropout01: same seed/data/rank/dropout/loss/batch/epochs; linearly decay learning rate from 1e-4 to 1e-5. One combined backward preserves the mathematical objective but may change floating-point accumulation. reference_results remains the original full-finetune mixed control.',
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('每轮步数：', len(loader), '；只保存一份最佳LoRA及任务头。')
    print('选模：同一轮两项F1共同评估，不按Dev loss选择。')

    history, selections, batch_plan = [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        losses, plan = train_epoch(model, loader, optimizer, scheduler, row_criterion, pair_criterion, device, epoch)
        assert losses['row_ce_rows'] == len(train) and losses['pair_ce_groups'] == dataset.pair_count
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
    del optimizer, scheduler
    load_trainable_weights(model, model_dir / 'best_model.pth')
    best = evaluate_phase(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
    evaluate_phase(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    saved = pd.read_csv(output_dir / f'epoch_{best_epoch}/predictions_with_scores.csv')
    reloaded = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert saved.instance_id.tolist() == reloaded.instance_id.tolist()
    differences = {column: int(saved[column].ne(reloaded[column]).sum())
                   for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values()), '最佳权重重载后预测不一致。'
    (output_dir / 'reload_check.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    compare_reference(best, output_dir)
    print('\n最佳轮次：', best_epoch)
    print('最佳LoRA及任务头：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
