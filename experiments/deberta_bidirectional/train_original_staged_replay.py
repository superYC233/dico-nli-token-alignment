"""右键运行：原Row阶段，再做同批四分类CE + Pair CE + 全部N回放。"""
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


RUN_NAME = 'run05_original_staged_replay_seed42_mem'
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

# 选同一轮的两项成绩，不把不同轮次的峰值拼起来。
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94
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
    return DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)


def trim_padding(batch):
    # 原数据右侧补齐到128；每批只把有效token送入模型，不截断正文。
    if DYNAMIC_PADDING:
        length = int(batch['attention_mask'].sum(dim=1).max())
        batch['input_ids'] = batch['input_ids'][:, :length].contiguous()
        batch['attention_mask'] = batch['attention_mask'][:, :length].contiguous()
    return batch


def collate_rows(examples):
    return trim_padding(default_collate(examples))


def replay_batches(rows, pair_loader, n_generator):
    # 只打乱Train里的N单例，不构造反向N，也不使用PAWS。
    negatives = torch.where(rows.labels.eq(3))[0]
    negatives = negatives[torch.randperm(len(negatives), generator=n_generator)]
    total_pairs = len(pair_loader.dataset)
    seen_pairs, n_start = 0, 0
    for first, second, targets in pair_loader:
        pair_count = len(targets)
        seen_pairs += pair_count
        # 按已处理的Pair比例分配N；最后一批正好用完，不重复也不遗漏。
        n_end = len(negatives) * seen_pairs // total_pairs
        n_rows = negatives[n_start:n_end]
        n_start = n_end
        source_rows = torch.cat([first, second, n_rows])
        yield trim_padding({
            'input_ids': rows.encoded['input_ids'][source_rows],
            'attention_mask': rows.encoded['attention_mask'][source_rows],
            'labels': rows.labels[source_rows],
            'pair_targets': targets,
            'source_rows': source_rows,
            'pair_count': pair_count,
            'n_count': len(n_rows),
        })


def compute_replay_loss(logits, labels, pair_targets, row_criterion, pair_criterion):
    pair_count = len(pair_targets)
    # 同一份输出：所有行独立分类；只有前面的真实配对行算三组合CE。
    row_loss = row_criterion(logits, labels)
    first = logits[:pair_count]
    second = logits[pair_count:2 * pair_count]
    pair_loss = pair_criterion(get_pair_logits(first, second), pair_targets)
    return row_loss + PAIR_WEIGHT * pair_loss, row_loss, pair_loss


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    started = time.perf_counter()
    total, total_weight = 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    for batch in tqdm(loader, desc='原版Row CE'):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['label'].to(device)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total += loss.item() * weight
        total_weight += weight
        counts += torch.bincount(batch['label'], minlength=4)
    record = {'row_loss': total / total_weight, 'row_ce_rows': int(counts.sum()),
              'row_optimizer_steps': len(loader), 'row_train_seconds': time.perf_counter() - started}
    record.update({f'row_seen_{label}': int(count) for label, count in zip(LABELS, counts)})
    return record


def train_pair_replay(model, rows, pair_loader, n_generator, optimizer,
                      row_criterion, pair_criterion, device):
    model.train()
    started = time.perf_counter()
    row_sum, row_weights, pair_sum, objective_sum = 0.0, 0.0, 0.0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    exposure = []
    sequence = replay_batches(rows, pair_loader, n_generator)
    for step, batch in enumerate(tqdm(sequence, total=len(pair_loader), desc='同批四分类 + Pair CE + N回放'), 1):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['labels'].to(device)
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss, row_loss, pair_loss = compute_replay_loss(
            logits, labels, batch['pair_targets'].to(device), row_criterion, pair_criterion,
        )
        loss.backward()
        optimizer.step()
        weight = row_criterion.weight[labels].sum().item()
        row_sum += row_loss.item() * weight
        row_weights += weight
        pair_sum += pair_loss.item()
        objective_sum += loss.item()
        counts += torch.bincount(batch['labels'], minlength=4)
        source = batch['source_rows'].tolist()
        pairs = batch['pair_count']
        ids = rows.frame.iloc[source].instance_id.tolist()
        exposure.append({
            'step': step, 'pair_groups': pairs, 'n_rows': batch['n_count'], 'rows': len(source),
            'source_rows': '|'.join(map(str, source)),
            'first_instance_ids': '|'.join(ids[:pairs]),
            'second_instance_ids': '|'.join(ids[pairs:2 * pairs]),
            'n_instance_ids': '|'.join(ids[2 * pairs:]),
            'row_loss': row_loss.item(), 'pair_loss': pair_loss.item(), 'loss': loss.item(),
        })
    record = {
        'pair_stage_row_loss': row_sum / row_weights, 'pair_loss': pair_sum / len(pair_loader),
        'pair_stage_objective': objective_sum / len(pair_loader),
        'pair_stage_ce_rows': int(counts.sum()), 'pair_ce_groups': sum(x['pair_groups'] for x in exposure),
        'replayed_n_rows': int(counts[3]), 'pair_optimizer_steps': len(pair_loader),
        'pair_train_seconds': time.perf_counter() - started,
    }
    record.update({f'pair_seen_{label}': int(count) for label, count in zip(LABELS, counts)})
    return record, exposure


def selection_record(epoch, summary):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    return {'epoch': epoch, 'phase': 'after_pair', 'independent_f1': independent,
            'pair_aware_f1': paired,
            'selection_score': min(independent / TARGET_INDEPENDENT, paired / TARGET_PAIR),
            'targets_met': independent >= TARGET_INDEPENDENT and paired >= TARGET_PAIR}


def source_hashes():
    paths = [Path(__file__).resolve(), HERE / 'train_deberta_bidirectional.py',
             HERE / 'data.py', HERE / 'alignment_joint_model.py',
             ROOT / 'experiments/deberta_joint_pair/data.py',
             ROOT / 'experiments/deberta_lexical/model.py',
             ROOT / 'experiments/deberta_lexical/evaluation.py',
             ROOT / 'experiments/deberta_manual_aug/train_boundary_manual.py',
             ROOT / 'experiments/deberta_stage_average/train_deberta_stage_average.py']
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pairs = RealPairDataset(rows)
    dev_dataset = RowDataset(dev, tokenizer, MAX_LENGTH)
    n_count = int(rows.labels.eq(3).sum())
    assert len(pairs) * 2 + n_count == len(rows)
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    row_steps = (len(rows) + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE
    pair_steps = (len(pairs) + PAIR_BATCH_SIZE - 1) // PAIR_BATCH_SIZE
    print('模型：原0.938的CLS四分类 + Token Alignment四分类残差，全参数训练。')
    print('训练顺序：原Row CE → 评估 → 同批四分类CE + 0.5×Pair CE + N回放 → 评估。')
    print('Train / Dev：', len(train), '/', len(dev), '；真实Pair组：', len(pairs))
    print('第二阶段：', len(pairs) * 2, '条配对行 +', n_count, '条N，每行恰好一次。')
    print('每轮Row / Pair更新：', row_steps, '/', pair_steps, '；合计：', row_steps + pair_steps)
    print('每轮四分类行曝光：', len(rows) * 2, '；无PAWS，无反向N，无NN组合。')
    print('动态padding / 编码器梯度检查点：', DYNAMIC_PADDING, '/', GRADIENT_CHECKPOINTING)
    print('Train / Dev最长有效token：', int(rows.encoded['attention_mask'].sum(1).max()),
          '/', int(dev_dataset.encoded['attention_mask'].sum(1).max()))
    print('seed / 学习率 / 轮次：', SEED, '/', LEARNING_RATE, '/', EPOCHS)
    print('选模：同轮min(independent/0.87, pair-aware/0.94)，只保存一个最佳模型。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据、分词和源码检查完成，没有加载模型或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    if GRADIENT_CHECKPOINTING:
        model.deberta_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('全部可训练参数：', parameter_count)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True, collate_fn=collate_rows)
    # 这里只加载原两行的行号，分词结果和金标直接从同一个RowDataset取。
    pair_loader = DataLoader(pairs.pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    n_generator = torch.Generator().manual_seed(SEED + 100)
    dev_loader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
                            generator=torch.Generator().manual_seed(SEED + 1))
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False, collate_fn=collate_rows,
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
    (output_dir / 'reference_run_config.json').write_bytes((REFERENCE_DIR / 'run_config.json').read_bytes())
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS four-class logits + Token Alignment four-class residual',
        'full_finetuning': True, 'use_lora': False, 'use_joint_head': False, 'use_paws': False,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'max_length': MAX_LENGTH, 'dtype': 'float32', 'dynamic_padding': DYNAMIC_PADDING,
        'gradient_checkpointing': GRADIENT_CHECKPOINTING,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
        'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
        'pair_stage_row_weight': 1.0, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'train_rows': len(train), 'real_pairs': len(pairs), 'n_rows': n_count,
        'train_sha256': train_hash, 'dev_sha256': file_hash(DEV_PATH),
        'row_steps_per_epoch': row_steps, 'pair_steps_per_epoch': pair_steps,
        'optimizer_steps_per_epoch': row_steps + pair_steps,
        'row_ce_exposure_per_epoch': len(rows) * 2, 'encoded_rows_per_epoch': len(rows) * 2,
        'trainable_parameters': parameter_count,
        'n_replay': 'shuffle Train N independently; allocate by cumulative pair count; each N once per epoch',
        'training_order': 'all Row CE; evaluate; same-pair Row CE plus N replay and 0.5 Pair CE; evaluate',
        'selection': 'best AFTER-PAIR min(independent/0.87, pair_aware/0.94); earliest on ties',
        'target_independent': TARGET_INDEPENDENT, 'target_pair': TARGET_PAIR,
        'comparison_note': 'original uses 6050 encoded rows; replay uses 6580 and adds full-strength same-pair Row CE; not a pure shuffle change',
        'checkpoint_format': 'original_alignment_full_v1', 'reference_run': str(REFERENCE_DIR),
        'source_sha256': hashes,
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    history, stages, changes, exposure_history, selections = [], [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_losses = train_rows(model, row_loader, optimizer, row_criterion, device)
        print('Row阶段：', row_losses)
        row_dir = output_dir / f'epoch_{epoch}/after_row'
        pair_dir = output_dir / f'epoch_{epoch}/after_pair'
        print('\n===== Row结束，第二阶段尚未开始 =====')
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses, exposure = train_pair_replay(
            model, rows, pair_loader, n_generator, optimizer, row_criterion, pair_criterion, device,
        )
        assert row_losses['row_ce_rows'] == pair_losses['pair_stage_ce_rows'] == len(train)
        assert pair_losses['pair_ce_groups'] == len(pairs) and pair_losses['replayed_n_rows'] == n_count
        losses = dict(row_losses, **pair_losses, optimizer_steps=row_steps + pair_steps)
        print('第二阶段：', pair_losses)
        print('\n===== 同批四分类 + Pair CE + N回放结束 =====')
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
        pd.DataFrame(exposure_history).to_csv(output_dir / 'pair_replay_exposure.csv', index=False, encoding='utf-8-sig')
        selected = selection_record(epoch, pair_summary)
        selections.append(selected)
        pd.DataFrame(selections).to_csv(output_dir / 'selection_history.csv', index=False, encoding='utf-8-sig')
        print('同轮选模指标：', selected)
        if selected['selection_score'] > best_score:
            best_score, best_epoch = selected['selection_score'], epoch
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新同轮双指标最佳模型。')

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
    print('\n最佳轮次（同轮双指标）：', best_epoch)
    print('最佳完整原模型：', model_dir / 'best_model.pth')
    print(pd.read_csv(output_dir / 'best_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
