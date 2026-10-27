"""右键运行：原3290行，Row CE之后训练Pair CE + N/非N二分类损失。"""
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


RUN_NAME = 'run24_pair_gate_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16  # 16组两行Pair，再分配约6或7条已有N；最多39行。
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

# 新增训练目标。权重沿用Row里N/非N的权重，没有额外提高N权重。
GATE_WEIGHT = 1.0
GATE_CLASS_WEIGHTS = [1.0, 1.5]  # 非N、N。
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


def pair_gate_batches(rows, pair_loader, n_generator):
    # 相同的3290行：配对行第二阶段各一次，已有530条N也各一次。没有新增或反转N。
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


def n_gate_logits(logits):
    # softmax后恰好等于[P(F)+P(B)+P(E), P(N)]，不是max(F,B,E)。
    logits = logits.float()
    return torch.stack([torch.logsumexp(logits[:, :3], dim=-1), logits[:, 3]], dim=-1)


def compute_losses(logits, labels, pair_targets, pair_criterion, gate_criterion,
                   gate_weight=GATE_WEIGHT, pair_weight=PAIR_WEIGHT):
    count = len(pair_targets)
    pair_loss = pair_criterion(get_pair_logits(logits[:count], logits[count:2 * count]), pair_targets)
    gate_loss = gate_criterion(n_gate_logits(logits), labels.eq(3).long())
    return pair_weight * pair_loss + gate_weight * gate_loss, pair_loss, gate_loss


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    total, weight_sum, row_count = 0.0, 0.0, 0
    for batch in tqdm(loader, desc='第一阶段：原四分类 Row CE'):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['label'].to(device)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += loss.item() * weight
        weight_sum += weight
        row_count += len(labels)
    return {'row_loss': total / weight_sum, 'row_ce_rows': row_count}


def train_pair_gate(model, rows, pair_loader, n_generator, optimizer,
                    pair_criterion, gate_criterion, device, gate_weight=GATE_WEIGHT):
    model.train()
    pair_sum, gate_sum, gate_weights, objective_sum = 0.0, 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    false_n, correct_n, exposures = 0, 0, []
    batches = pair_gate_batches(rows, pair_loader, n_generator)
    for step, batch in enumerate(tqdm(batches, total=len(pair_loader), desc='第二阶段：Pair CE + N/非N'), 1):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['labels'].to(device)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss, pair_loss, gate_loss = compute_losses(
            logits, labels, batch['pair_targets'].to(device), pair_criterion, gate_criterion, gate_weight,
        )
        loss.backward()
        optimizer.step()
        gate_scale = gate_criterion.weight[labels.eq(3).long()].sum().item()
        pair_sum += pair_loss.item()
        gate_sum += gate_loss.item() * gate_scale
        gate_weights += gate_scale
        objective_sum += loss.item()
        counts += torch.bincount(batch['labels'], minlength=4)
        predicted = logits.detach().argmax(1)
        false_n += int((labels.ne(3) & predicted.eq(3)).sum())
        correct_n += int((labels.eq(3) & predicted.eq(3)).sum())
        exposures.append({'step': step, 'pairs': batch['pair_count'], 'n_rows': batch['n_count'],
                          'rows': len(labels), 'source_rows': '|'.join(map(str, batch['source_rows'].tolist())),
                          'pair_loss': pair_loss.item(), 'gate_loss': gate_loss.item(), 'loss': loss.item()})
    record = {'pair_loss': pair_sum / len(pair_loader), 'gate_loss': gate_sum / gate_weights,
              'pair_gate_objective': objective_sum / len(pair_loader), 'gate_weight': gate_weight,
              'gate_rows': int(counts.sum()), 'gate_n_rows': int(counts[3]),
              'pair_ce_groups': sum(x['pairs'] for x in exposures),
              'train_nonN_to_N_during_pair': false_n, 'train_N_correct_during_pair': correct_n,
              **{f'gate_seen_{label}': int(count) for label, count in zip('FBEN', counts)}}
    return record, exposures


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, prepare_only=False, gate_weight=GATE_WEIGHT):
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows, dev_rows = RowDataset(train, tokenizer, MAX_LENGTH), RowDataset(dev, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    n_count = int(rows.labels.eq(3).sum())
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    print('原0.938架构和3290行数据；第一阶段Row CE；第二阶段0.5×Pair CE + N/非N损失。')
    print('Train / Dev / 真实Pair / 已有N：', len(rows), '/', len(dev_rows), '/', len(pairs), '/', n_count)
    print('每条训练行每轮编码两次：Row一次，Pair+二分类一次；不额外多采样N，不构造NN。')
    print('Row / Pair更新：', row_steps, '/', pair_steps, '；Gate系数：', gate_weight)
    print('选模仍按Pair后Dev pair-aware F1，同时记录independent和非N→N。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据与编码检查完成，没有加载模型权重或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有训练记录，请换新RUN_NAME，保留旧结果。')
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
    gate_criterion = nn.CrossEntropyLoss(weight=torch.tensor(GATE_CLASS_WEIGHTS, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    sources = [Path(__file__).resolve(), ROOT / 'experiments/deberta_lexical/model.py',
               ROOT / 'experiments/deberta_lexical/evaluation.py',
               ROOT / 'experiments/deberta_bidirectional/train_deberta_bidirectional.py',
               ROOT / 'experiments/deberta_manual_aug/train_boundary_manual.py',
               ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py']
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment residual',
        'full_finetuning': True, 'use_lexical_features': False, 'use_lora': False,
        'train_rows': len(train), 'real_pairs': len(pairs), 'n_rows': n_count,
        'train_sha256': train_hash, 'dev_sha256': file_hash(DEV_PATH), 'reference_run': str(REFERENCE_DIR),
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'gate_weight': gate_weight, 'gate_class_weights': GATE_CLASS_WEIGHTS,
        'gate_logit_order': ['logsumexp(F,B,E)', 'N'], 'gate_target': 'Train label == NEGATIVE_OTHER',
        'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'encoded_rows_per_epoch': 2 * len(train), 'second_stage_rows': len(train),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_order': 'full Row CE; evaluate; original three-way Pair CE + binary N gate; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties',
        'inference': 'unchanged four-class argmax; original FB/BF/EE decoding; singleton kept as model argmax',
        'comparison_note': 'Compared with full Row replay, remove non-N single-row conditional FBE CE. Compared with historical0.938, add gate and N exposure; dynamic padding changes RNG trajectory.',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    history, stages, changes, exposure_history = [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses = train_rows(model, row_loader, optimizer, row_criterion, device)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses, exposures = train_pair_gate(model, rows, pair_loader, n_generator, optimizer,
                                                pair_criterion, gate_criterion, device, gate_weight)
        assert row_losses['row_ce_rows'] == pair_losses['gate_rows'] == len(train)
        assert pair_losses['gate_n_rows'] == n_count and pair_losses['pair_ce_groups'] == len(pairs)
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        losses = dict(row_losses, **pair_losses)
        print('本轮损失与曝光：', losses)
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
        for item in exposures:
            item['epoch'] = epoch
        exposure_history.extend(exposures)
        pd.DataFrame(exposure_history).to_csv(output_dir / 'pair_gate_batches.csv', index=False, encoding='utf-8-sig')
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
    parser.add_argument('--gate-weight', type=float, default=GATE_WEIGHT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only, args.gate_weight)
