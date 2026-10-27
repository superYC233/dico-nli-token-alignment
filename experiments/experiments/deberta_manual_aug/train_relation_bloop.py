"""右键运行：原0.938数据/模型，保留Row阶段，第二阶段用BLOOP保护Row梯度。"""
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
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase, stage_changes
from experiments.deberta_bidirectional.data import write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights
from experiments.deberta_bidirectional.train_deberta_bidirectional import read_data, file_hash, training_records_exist
from experiments.deberta_manual_aug.row_priority_gradients import separate_gradients
from experiments.deberta_manual_aug.bloop_gradients import BloopGradients


RUN_NAME = 'run25_row_pair_ema'
MODE = 'bloop'  # mixed为同代码普通相加对照；默认只运行bloop。
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
EMA_DECAY = 0.9  # 每步保留90%旧梯度、混入10%当前Row梯度；不是模型权重平均。
PROJECTION_EPS = 1e-7
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


def trim_batch(batch, dynamic_padding=DYNAMIC_PADDING):
    length = int(batch['attention_mask'].sum(1).max()) if dynamic_padding else batch['input_ids'].shape[1]
    batch['input_ids'] = batch['input_ids'][:, :length].contiguous()
    batch['attention_mask'] = batch['attention_mask'][:, :length].contiguous()
    return batch


def collate_rows(examples, dynamic_padding=DYNAMIC_PADDING):
    return trim_batch(default_collate(examples), dynamic_padding)


def pair_batches(rows, pair_loader, n_generator):
    # 第二阶段每行各一次：原1380组配对行 + 原530条N，没有新增数据。
    negatives = torch.where(rows.labels.eq(3))[0]
    negatives = negatives[torch.randperm(len(negatives), generator=n_generator)]
    total_pairs = len(pair_loader.dataset)
    seen_pairs, n_start = 0, 0
    for first, second, targets in pair_loader:
        count = len(targets)
        seen_pairs += count
        n_end = len(negatives) * seen_pairs // total_pairs
        n_rows = negatives[n_start:n_end]
        n_start = n_end
        source = torch.cat([first, second, n_rows])
        yield trim_batch({
            'input_ids': rows.encoded['input_ids'][source],
            'attention_mask': rows.encoded['attention_mask'][source],
            'labels': rows.labels[source], 'pair_targets': targets,
            'source_rows': source, 'pair_count': count, 'n_count': len(n_rows),
        })


def compute_losses(logits, labels, pair_targets, row_criterion, pair_criterion):
    count = len(pair_targets)
    row_loss = row_criterion(logits, labels)
    pair_loss = pair_criterion(get_pair_logits(logits[:count], logits[count:2 * count]), pair_targets)
    return row_loss, pair_loss


def train_rows(model, loader, optimizer, criterion, device, gradients):
    model.train()
    total, weights, row_count = 0.0, 0.0, 0
    for batch in tqdm(loader, desc='第一阶段：原四分类 Row CE'):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['label'].to(device)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss = criterion(logits, labels)
        loss.backward()
        # 辅助目标尚未启用，仍仅按Row更新；同时让EMA跟踪当前主梯度。
        gradients.update_ema([parameter.grad for parameter in gradients.parameters])
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += loss.item() * weight
        weights += weight
        row_count += len(labels)
    return {'row_loss': total / weights, 'row_ce_rows': row_count}


def train_pair_bloop(model, rows, pair_loader, n_generator, optimizer,
                     row_criterion, pair_criterion, device, gradients, mode=MODE):
    model.train()
    row_sum, row_weights, pair_sum = 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    records = []
    batches = pair_batches(rows, pair_loader, n_generator)
    for step, batch in enumerate(tqdm(batches, total=len(pair_loader), desc=f'第二阶段：{mode} Row + Pair'), 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        labels = batch['labels'].to(device)
        row_loss, pair_loss = compute_losses(logits, labels, batch['pair_targets'].to(device),
                                             row_criterion, pair_criterion)
        # Pair在这里已乘0.5；BLOOP合成时不能再乘一次。
        row_gradients, pair_gradients = separate_gradients(
            row_loss, pair_loss, gradients.parameters, PAIR_WEIGHT,
        )
        stats = gradients.assign(row_gradients, pair_gradients, mode)
        optimizer.step()
        weight = row_criterion.weight[labels].sum().item()
        row_sum += row_loss.item() * weight
        row_weights += weight
        pair_sum += pair_loss.item()
        counts += torch.bincount(batch['labels'], minlength=4)
        stats.update(step=step, mode=mode, row_loss=row_loss.item(), pair_loss=pair_loss.item(),
                     weighted_pair_loss=PAIR_WEIGHT * pair_loss.item(), pair_groups=batch['pair_count'],
                     n_rows=batch['n_count'], rows=len(labels),
                     source_rows='|'.join(map(str, batch['source_rows'].tolist())))
        records.append(stats)
        # 不让上一批的图和两套梯度留到下一批前向。
        del row_gradients, pair_gradients, logits, row_loss, pair_loss
    result = {
        'pair_stage_row_loss': row_sum / row_weights, 'pair_loss': pair_sum / len(pair_loader),
        'pair_stage_rows': int(counts.sum()), 'pair_stage_n_rows': int(counts[3]),
        'pair_ce_groups': sum(record['pair_groups'] for record in records),
        'ema_updates_total': gradients.updates,
        **{f'pair_seen_{label}': int(count) for label, count in zip('FBEN', counts)},
    }
    return result, records


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, mode=MODE, prepare_only=False):
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev_rows = RowDataset(dev, tokenizer, MAX_LENGTH)
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    effective_name = f'{run_name}_{mode}'
    model_dir, output_dir = MODEL_ROOT / effective_name, HERE / 'outputs' / effective_name
    print('基准：0.938的3290行、原CLS+Token Alignment；从同一NLI初始化重训。')
    print('模式：', mode, '；Train / Dev / 真实Pair：', len(rows), '/', len(dev_rows), '/', len(pairs))
    print('第一阶段完整Row CE；第二阶段完整Row为主、Pair为辅。')
    print('梯度合成：', 'BLOOP按Row梯度EMA投影Pair' if mode == 'bloop' else '普通Row + 0.5*Pair相加对照')
    print('Row / 第二阶段更新：', row_steps, '/', pair_steps, '；梯度EMA decay：', EMA_DECAY)
    print('第二阶段每行各一次，最多39行；没有新64行、NN或N/非N辅助损失。')
    print('动态padding / 梯度检查点：', DYNAMIC_PADDING, '/', GRADIENT_CHECKPOINTING)
    print('选模：第二阶段后Dev pair-aware F1最高，同时报告同一模型的independent。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据和编码检查完成；没有加载模型权重或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME和MODE已有训练记录，请换RUN_NAME，保留旧结果。')
    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    if GRADIENT_CHECKPOINTING:
        model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True, collate_fn=collate_rows)
    pair_loader = DataLoader(pairs.pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    n_generator = torch.Generator().manual_seed(SEED + 100)
    dev_loader = DataLoader(dev_rows, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
                            generator=torch.Generator().manual_seed(SEED + 1))
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
                              generator=torch.Generator().manual_seed(SEED + 2))
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                         label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    gradients = BloopGradients([p for p in model.parameters() if p.requires_grad], EMA_DECAY, PROJECTION_EPS)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    sources = [Path(__file__).resolve(), HERE / 'bloop_gradients.py', HERE / 'row_priority_gradients.py',
               ROOT / 'experiments/deberta_lexical/model.py', ROOT / 'experiments/deberta_lexical/evaluation.py',
               ROOT / 'experiments/deberta_bidirectional/train_deberta_bidirectional.py']
    config = {
        'run_name': effective_name, 'requested_run_name': run_name, 'mode': mode,
        'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment residual',
        'full_finetuning': True, 'use_lexical_features': False, 'use_lora': False,
        'train_rows': len(train), 'real_pairs': len(pairs), 'n_rows': int(rows.labels.eq(3).sum()),
        'train_sha256': train_hash, 'dev_sha256': file_hash(DEV_PATH), 'reference_run': str(REFERENCE_DIR),
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'ema_decay': EMA_DECAY, 'ema_new_gradient_fraction': 1.0 - EMA_DECAY, 'projection_eps': PROJECTION_EPS,
        'ema_initialization': 'first Row gradient; no random initialization',
        'ema_schedule': 'update every Row and second-stage step; persist across epochs',
        'bloop_rescale': False, 'bloop_use_ratio_only': False,
        'bloop_paper': 'https://arxiv.org/abs/2402.02998',
        'bloop_code_reference': 'https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/optim.py',
        'gradient_rule': ('gr + (gp - dot(m,gp)/(dot(m,m)+eps)*m); gp already includes pair_weight'
                          if mode == 'bloop' else 'gr + gp; gp already includes pair_weight'),
        'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'encoded_rows_per_epoch': 2 * len(rows),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_order': 'full Row CE; evaluate; same-batch full Row and Pair with chosen gradient combination; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties',
        'inference': 'unchanged four-class argmax and FB/BF/EE decoding; singleton remains model argmax',
        'comparison_note': 'Historical0.938 uses Pair-only second stage and6050 encoded rows; this uses6580. MODE mixed is a matched batch/EMA/computation control; not identical to run13 group-mixed schedule.',
        'optimizer_note': 'Projection is on gradients before AdamW; no pointwise or finite-step no-forgetting guarantee.',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    history, stages, changes, gradient_records = [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses = train_rows(model, row_loader, optimizer, row_criterion, device, gradients)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses, records = train_pair_bloop(model, rows, pair_loader, n_generator, optimizer,
                                               row_criterion, pair_criterion, device, gradients, mode)
        assert row_losses['row_ce_rows'] == pair_losses['pair_stage_rows'] == len(rows)
        assert pair_losses['pair_stage_n_rows'] == int(rows.labels.eq(3).sum())
        assert pair_losses['pair_ce_groups'] == len(pairs)
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        losses = dict(row_losses, **pair_losses)
        print('本轮损失与曝光：', losses)
        for phase, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, phase=phase, **losses)
                stages.append(record)
                if phase == 'after_pair':
                    history.append(record.copy())
        for record in records:
            record['epoch'] = epoch
        gradient_records.extend(records)
        pd.DataFrame(gradient_records).to_csv(output_dir / 'gradient_batches.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(stages).to_csv(output_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        changes.extend(stage_changes(epoch, 'dev', row_dir, pair_dir))
        pd.DataFrame(changes).to_csv(output_dir / 'stage_changes.csv', index=False, encoding='utf-8-sig')
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected = {'epoch': epoch, 'phase': 'after_pair', 'pair_aware_f1': score,
                        'independent_f1': float(pair_summary.loc[pair_summary.method.eq('independent'), 'weighted_f1'].iloc[0]),
                        'row_independent_f1': float(row_summary.loc[row_summary.method.eq('independent'), 'weighted_f1'].iloc[0]),
                        'nonN_to_N': int(pair_summary.loc[pair_summary.method.eq('independent'), 'nonN_to_N'].iloc[0])}
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新最佳模型：', selected)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, gradients
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
    parser.add_argument('--mode', choices=['bloop', 'mixed'], default=MODE)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.mode, args.prepare_only)
