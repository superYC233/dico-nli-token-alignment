"""右键运行：原0.938数据/模型，Row和Pair阶段分别使用SAM。"""
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

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, stage_changes
from experiments.deberta_bidirectional.data import write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights
from experiments.deberta_bidirectional.train_deberta_bidirectional import read_data, file_hash, training_records_exist
from experiments.deberta_manual_aug.sam_perturbation import SAMParameters, capture_rng, replay_rng


RUN_NAME = 'run27_row_pair_sam'
SAM_RHO = 0.05  # 普通SAM的全模型L2扰动半径；设0则走普通Row CE。
PAIR_SAM_RHO = 0.05  # 新增：Pair阶段也用SAM；设0恢复run26的Pair训练路径。
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
DYNAMIC_PADDING = True
GRADIENT_CHECKPOINTING = True
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / 'outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_manual_aug')


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
    batch['input_ids'], batch['attention_mask'] = trim_inputs(
        batch['input_ids'], batch['attention_mask'], dynamic_padding,
    )
    return batch


def collate_pairs(examples, dynamic_padding=DYNAMIC_PADDING):
    first_ids, first_mask, second_ids, second_mask, targets = default_collate(examples)
    first_ids, first_mask = trim_inputs(first_ids, first_mask, dynamic_padding)
    second_ids, second_mask = trim_inputs(second_ids, second_mask, dynamic_padding)
    return first_ids, first_mask, second_ids, second_mask, targets


def train_rows_sam(model, loader, optimizer, criterion, device, rho=SAM_RHO):
    model.train()
    started = time.perf_counter()
    perturbation = SAMParameters([p for p in model.parameters() if p.requires_grad], rho)
    clean_total, sam_total, total_weight = 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    records = []
    for step, batch in enumerate(tqdm(loader, desc='Row CE + SAM' if rho > 0 else '普通Row CE'), 1):
        optimizer.zero_grad(set_to_none=True)
        ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        states = capture_rng(device) if rho > 0 else None
        logits = model(ids, mask, None)
        clean_loss = criterion(logits, labels)
        clean_loss.backward()
        clean_value = clean_loss.item()
        del logits, clean_loss
        stats = {'clean_grad_norm': None, 'sam_radius': 0.0}
        if rho > 0:
            stats = perturbation.perturb()
            optimizer.zero_grad(set_to_none=True)
            try:
                # 第一遍只用于寻找扰动；只用第二遍的梯度更新，不能把两遍累加。
                with replay_rng(states, device):
                    logits = model(ids, mask, None)
                    sam_loss = criterion(logits, labels)
                    sam_loss.backward()
                    sam_value = sam_loss.item()
                del logits, sam_loss
            finally:
                # AdamW必须在恢复原参数后执行；Pair和预测均不能带临时扰动。
                perturbation.restore()
        else:
            sam_value = clean_value
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        clean_total += clean_value * weight
        sam_total += sam_value * weight
        total_weight += weight
        counts += torch.bincount(batch['label'], minlength=4)
        records.append({'step': step, 'rows': len(labels), 'tokens': ids.shape[1],
                        'row_loss': clean_value, 'sam_row_loss': sam_value,
                        'sam_loss_increase': sam_value - clean_value, 'sam_rho': rho, **stats})
    row_count = int(counts.sum())
    return {
        'row_loss': clean_total / total_weight, 'sam_row_loss': sam_total / total_weight,
        'sam_rho': rho, 'row_ce_rows': row_count,
        'row_forward_rows': row_count * (2 if rho > 0 else 1),
        'row_optimizer_steps': len(loader), 'row_train_seconds': time.perf_counter() - started,
        **{f'seen_{label}': int(count) for label, count in zip('FBEN', counts)},
    }, records


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def train_pairs(model, loader, optimizer, criterion, device):
    # 和原版相同：两个方向分别前向，只有0.5*三组合Pair CE。
    model.train()
    total = 0.0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='原版Pair CE（不使用SAM）'):
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def train_pairs_sam(model, loader, optimizer, criterion, device, rho=PAIR_SAM_RHO):
    model.train()
    started = time.perf_counter()
    perturbation = SAMParameters([p for p in model.parameters() if p.requires_grad], rho)
    total, sam_total, pair_count = 0.0, 0.0, 0
    records = []
    description = 'Pair CE + SAM' if rho > 0 else '原版Pair CE（不使用SAM）'
    for step, (ids0, mask0, ids1, mask1, targets) in enumerate(tqdm(loader, desc=description), 1):
        optimizer.zero_grad(set_to_none=True)
        ids0, mask0 = ids0.to(device), mask0.to(device)
        ids1, mask1 = ids1.to(device), mask1.to(device)
        targets = targets.to(device)
        # 在两个方向前向之前一起记录；重放时也要重放完整的两个方向。
        states = capture_rng(device) if rho > 0 else None
        first = model(ids0, mask0, None)
        second = model(ids1, mask1, None)
        pair_loss = criterion(get_pair_logits(first, second), targets)
        (PAIR_WEIGHT * pair_loss).backward()
        clean_value = pair_loss.item()
        del first, second, pair_loss
        stats = {'clean_grad_norm': None, 'sam_radius': 0.0}
        if rho > 0:
            stats = perturbation.perturb()
            optimizer.zero_grad(set_to_none=True)
            try:
                with replay_rng(states, device):
                    first = model(ids0, mask0, None)
                    second = model(ids1, mask1, None)
                    sam_loss = criterion(get_pair_logits(first, second), targets)
                    # 第一遍梯度已经清空；第二遍仍只乘一次0.5。
                    (PAIR_WEIGHT * sam_loss).backward()
                    sam_value = sam_loss.item()
                del first, second, sam_loss
            finally:
                perturbation.restore()
        else:
            sam_value = clean_value
        # 恢复原参数后，AdamW才更新一次。
        optimizer.step()
        total += clean_value
        sam_total += sam_value
        pair_count += len(targets)
        records.append({'step': step, 'pair_groups': len(targets),
                        'first_tokens': ids0.shape[1], 'second_tokens': ids1.shape[1],
                        'pair_loss': clean_value, 'sam_pair_loss': sam_value,
                        'sam_loss_increase': sam_value - clean_value,
                        'weighted_pair_loss': PAIR_WEIGHT * clean_value,
                        'weighted_sam_pair_loss': PAIR_WEIGHT * sam_value,
                        'pair_weight': PAIR_WEIGHT, 'pair_sam_rho': rho, **stats})
    return {
        'pair_loss': total / len(loader), 'sam_pair_loss': sam_total / len(loader),
        'pair_sam_rho': rho, 'pair_ce_groups': pair_count,
        'pair_forward_rows': 2 * pair_count * (2 if rho > 0 else 1),
        'pair_optimizer_steps': len(loader), 'pair_train_seconds': time.perf_counter() - started,
    }, records


def main(run_name=RUN_NAME, rho=SAM_RHO, prepare_only=False, pair_rho=PAIR_SAM_RHO):
    if rho < 0 or pair_rho < 0:
        raise ValueError('SAM_RHO和PAIR_SAM_RHO必须大于等于0。')
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev_rows = RowDataset(dev, tokenizer, MAX_LENGTH)
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    print('基准：原0.938的3290行、CLS+Token Alignment；从同一NLI初始化重训。')
    print('Train / Dev / 真实Pair：', len(rows), '/', len(dev_rows), '/', len(pairs))
    print('每轮：Row阶段 → 评估 → Pair-only阶段 → 评估，两阶段分别使用配置的SAM。')
    print('Row SAM rho / Pair SAM rho：', rho, '/', pair_rho)
    print('Row / Pair更新：', row_steps, '/', pair_steps)
    print('Pair系数 / 标签平滑：', PAIR_WEIGHT, '/', PAIR_LABEL_SMOOTHING)
    print('Row SAM：', rho > 0, '；Pair SAM：', pair_rho > 0, '；每批仍只更新一次AdamW。')
    print('动态padding / 梯度检查点：', DYNAMIC_PADDING, '/', GRADIENT_CHECKPOINTING)
    print('选模：Pair后Dev pair-aware F1最高；只保留一个最佳完整模型。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据和编码检查完成，没有加载模型或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有训练记录，请修改RUN_NAME，保留旧结果。')
    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    if GRADIENT_CHECKPOINTING:
        model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True, collate_fn=collate_rows)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True, collate_fn=collate_pairs)
    dev_loader = DataLoader(dev_rows, batch_size=DEV_BATCH_SIZE, collate_fn=collate_rows,
                            generator=torch.Generator().manual_seed(SEED + 1))
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, collate_fn=collate_rows,
                              generator=torch.Generator().manual_seed(SEED + 2))
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                         label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    for filename in ['data_config.json', 'overlap_report.json']:
        (output_dir / filename).write_bytes((REFERENCE_DIR / filename).read_bytes())
    sources = [Path(__file__).resolve(), HERE / 'sam_perturbation.py',
               ROOT / 'experiments/deberta_lexical/model.py', ROOT / 'experiments/deberta_lexical/evaluation.py',
               ROOT / 'experiments/deberta_joint_pair/data.py',
               ROOT / 'experiments/deberta_manual_aug/train_boundary_manual.py',
               ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py',
               ROOT / 'experiments/deberta_bidirectional/train_deberta_bidirectional.py']
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment residual',
        'full_finetuning': True, 'use_lora': False, 'use_lexical_features': False,
        'train_rows': len(rows), 'real_pairs': len(pairs), 'train_sha256': train_hash,
        'dev_sha256': file_hash(DEV_PATH), 'reference_run': str(REFERENCE_DIR),
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'sam_rho': rho, 'pair_sam_rho': pair_rho, 'sam_adaptive': False,
        'sam_scope': 'all trainable parameters; enabled stages: ' + ', '.join(
            stage for stage, radius in [('Row', rho), ('Pair', pair_rho)] if radius > 0),
        'sam_rule': 'e=stage_rho*grad(stage_objective)/(global_L2_norm+1e-12); restore w; AdamW uses perturbed gradient only',
        'pair_sam_objective': '0.5 * Pair CE; weight applied once per backward, not twice',
        'sam_dropout': 'reuse complete clean-forward RNG; Pair replay includes both directional forwards; restore post-clean RNG',
        'sam_paper': 'https://arxiv.org/abs/2010.01412',
        'sam_code_reference': 'https://github.com/davda54/sam/blob/main/sam.py',
        'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'optimizer_steps_per_epoch': row_steps + pair_steps,
        'forward_rows_per_epoch': (2 if rho > 0 else 1) * len(rows) + (2 if pair_rho > 0 else 1) * 2 * len(pairs),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_order': 'all Row batches with Row SAM setting; evaluate; all three-combination Pair-only batches with Pair SAM setting; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware F1; earliest on ties',
        'inference': 'original four-class argmax and FB/BF/EE decoding; singleton kept as model argmax',
        'comparison_note': 'Compared with run26, only Pair SAM is enabled by default; Row function/model/data/loss settings are unchanged. Pair rho0 reproduces the previous Pair path. Dynamic padding remains enabled and differs from historical0.938 fixed128.',
        'checkpoint_note': 'best model only; no optimizer resume state or temporary SAM parameters',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    history, stages, changes, sam_records, pair_sam_records = [], [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses, diagnostics = train_rows_sam(model, row_loader, optimizer, row_criterion, device, rho)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses, pair_diagnostics = train_pairs_sam(
            model, pair_loader, optimizer, pair_criterion, device, pair_rho,
        )
        losses = dict(row_losses, **pair_losses)
        assert losses['row_ce_rows'] == len(rows) and losses['pair_ce_groups'] == len(pairs)
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        print('本轮损失：', losses)
        for phase, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, phase=phase, **losses)
                stages.append(record)
                if phase == 'after_pair':
                    history.append(record.copy())
        for record in diagnostics:
            record['epoch'] = epoch
        sam_records.extend(diagnostics)
        pd.DataFrame(sam_records).to_csv(output_dir / 'sam_batches.csv', index=False, encoding='utf-8-sig')
        for record in pair_diagnostics:
            record['epoch'] = epoch
        pair_sam_records.extend(pair_diagnostics)
        pd.DataFrame(pair_sam_records).to_csv(output_dir / 'pair_sam_batches.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(stages).to_csv(output_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        changes.extend(stage_changes(epoch, 'dev', row_dir, pair_dir))
        pd.DataFrame(changes).to_csv(output_dir / 'stage_changes.csv', index=False, encoding='utf-8-sig')
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected = {'epoch': epoch, 'phase': 'after_pair', 'pair_aware_f1': score,
                        'independent_f1': float(pair_summary.loc[pair_summary.method.eq('independent'), 'weighted_f1'].iloc[0]),
                        'row_independent_f1': float(row_summary.loc[row_summary.method.eq('independent'), 'weighted_f1'].iloc[0])}
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
    old = pd.read_csv(output_dir / f'epoch_{best_epoch}/after_pair/predictions_with_scores.csv')
    new = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert old.instance_id.tolist() == new.instance_id.tolist()
    assert old[['prediction', 'pair_prediction']].equals(new[['prediction', 'pair_prediction']])
    print('\n最佳轮次：', best_epoch, '；pair-aware F1：', best_score)
    print('最佳完整模型：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--sam-rho', type=float, default=SAM_RHO)
    parser.add_argument('--pair-sam-rho', type=float, default=PAIR_SAM_RHO)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.sam_rho, args.prepare_only, args.pair_sam_rho)
