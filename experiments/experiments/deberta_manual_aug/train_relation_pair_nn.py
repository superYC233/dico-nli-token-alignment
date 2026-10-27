"""右键运行：先Row CE，再FB/BF/EE/NN四组合Pair CE。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from evaluation_functions import score_files
from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_manual_aug.train_boundary_manual import set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import (
    REFERENCE_DIR, GOLD_PATH, LABELS, FULL_LABELS, read_data,
    write_train_reference, train_rows, score_predictions,
)


RUN_NAME = 'run10_pair_nn'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
EPOCHS = 10
TRAIN_AUDIT_EPOCH = 2
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25, 1.0]  # FB、BF、EE、NN；NN先用普通权重。
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.0  # 与run09关闭平滑的对照一致。

# 沿用0.9381的选模标准；四组合解码同时报告，用来检查N是否参与竞争。
# 若另一次实验要按四组合选模，可改为'pair_four'，并换RUN_NAME。
SELECT_BY = 'pair_aware'
COMBINATIONS = [(0, 1), (1, 0), (2, 2), (3, 3)]
METHODS = [('independent', 'prediction'), ('pair_aware', 'pair_prediction'),
           ('pair_four', 'four_prediction')]
HERE = Path(__file__).resolve().parent
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME


def text_key(text):
    return ' '.join(str(text).casefold().split())


def reverse_row(row):
    reversed_row = row.copy()
    reversed_row['text1'], reversed_row['text2'] = row['text2'], row['text1']
    reversed_row['text1_lang'], reversed_row['text2_lang'] = row['text2_lang'], row['text1_lang']
    reversed_row['instance_id'] = row['instance_id'] + '__pair_nn_reverse'
    return reversed_row


def build_pair_data(train):
    """只看Train：保留真实F/B/E组；给不冲突的N单例构造反向行。"""
    known_labels = {}
    for row in train.to_dict('records'):
        key = (text_key(row['text1']), text_key(row['text2']))
        known_labels.setdefault(key, set()).add(row['label'])
    reverse_rows, pairs, audit = [], [], []
    target_ids = {(0, 1): 0, (1, 0): 1, (2, 2): 2}
    label_ids = dict(zip(FULL_LABELS, range(4)))
    for pair_id, group in train.groupby('pair_id'):
        indices = group.index.tolist()
        if len(group) == 2:
            first, second = indices
            target = target_ids[tuple(group.label.map(label_ids))]
            pairs.append((first, second, target))
            continue
        row = group.iloc[0].to_dict()
        assert row['label'] == 'NEGATIVE_OTHER'
        key = (text_key(row['text1']), text_key(row['text2']))
        labels = known_labels.get(key, set()) | known_labels.get(key[::-1], set())
        accepted = labels == {'NEGATIVE_OTHER'}
        audit.append({'instance_id': row['instance_id'], 'pair_id': pair_id,
                      'text1': row['text1'], 'text2': row['text2'], 'accepted': accepted,
                      'known_train_labels_both_directions': '|'.join(sorted(labels)),
                      'reason': 'accepted' if accepted else 'Train has a conflicting non-N label'})
        if accepted:
            reverse_index = len(train) + len(reverse_rows)
            reverse_rows.append(reverse_row(row))
            pairs.append((indices[0], reverse_index, 3))
    expanded = pd.concat([train, pd.DataFrame(reverse_rows, columns=train.columns)], ignore_index=True)
    return expanded, pairs, pd.DataFrame(audit)


class FourPairDataset(Dataset):
    def __init__(self, rows, pairs):
        self.rows = rows
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first, second, target = self.pairs[index]
        row0, row1 = self.rows[first], self.rows[second]
        return (row0['input_ids'], row0['attention_mask'],
                row1['input_ids'], row1['attention_mask'],
                torch.tensor(target, dtype=torch.long))


def get_four_logits(first, second):
    return torch.stack([
        first[:, 0] + second[:, 1],  # FB
        first[:, 1] + second[:, 0],  # BF
        first[:, 2] + second[:, 2],  # EE
        first[:, 3] + second[:, 3],  # NN，必须进入同一个四组合softmax。
    ], dim=1)


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='四组合Pair CE：含NN'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = criterion(get_four_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def build_eval_loaders(rows, tokenizer):
    """不读金标：有反向行就复用；单例交换文本后额外前向。"""
    frame = rows.frame
    reverse_indices = torch.empty(len(frame), dtype=torch.long)
    extra_rows = []
    for _, group in frame.groupby('pair_id'):
        indices = group.index.tolist()
        if len(indices) == 2:
            first, second = indices
            assert frame.loc[first, 'text1'] == frame.loc[second, 'text2']
            assert frame.loc[first, 'text2'] == frame.loc[second, 'text1']
            reverse_indices[first], reverse_indices[second] = second, first
        else:
            index = indices[0]
            reverse_indices[index] = len(frame) + len(extra_rows)
            extra = reverse_row(frame.loc[index].to_dict())
            # RowDataset要求label列；这里只填固定占位，不使用单例金标作反向监督。
            extra['label'] = FULL_LABELS[0]
            extra_rows.append(extra)
    extra = RowDataset(pd.DataFrame(extra_rows, columns=frame.columns), tokenizer, MAX_LENGTH)
    # 原始行仍固定batch16，不与额外反向行拼批，保持原三组合评估的批次组成。
    loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
                        generator=torch.Generator().manual_seed(SEED + 1))
    reverse_loader = DataLoader(extra, batch_size=DEV_BATCH_SIZE, shuffle=False,
                                generator=torch.Generator().manual_seed(SEED + 2))
    return loader, reverse_loader, reverse_indices


def pair_four_decode(logits, reverse_logits, frame):
    predictions = logits.argmax(dim=1).clone()
    combinations = torch.tensor(COMBINATIONS, device=logits.device)
    for _, group in frame.groupby('pair_id'):
        first = group.index[0]
        scores = get_four_logits(logits[first:first + 1], reverse_logits[first:first + 1])
        labels = combinations[scores.argmax(dim=1).item()]
        predictions[first] = labels[0]
        if len(group) == 2:
            predictions[group.index[1]] = labels[1]
    # 每个组都比较四个候选；没有利用单例/配对身份来排除N。
    return predictions


def score_all(logits, reverse_logits, frame, output_dir, gold_path):
    # 先复用原来的独立预测、三组合解码和官方评分。
    summary = score_predictions(logits, frame, output_dir, gold_path)
    saved = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    four = pair_four_decode(logits, reverse_logits, frame)
    saved['four_prediction'] = [LABELS[index] for index in four.tolist()]
    for index, label in enumerate(LABELS):
        saved[f'reverse_logit_{label}'] = reverse_logits[:, index].tolist()
    saved.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')

    score_dir = output_dir / 'pair_four'
    score_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = score_dir / 'official_predictions.csv'
    pd.DataFrame({'instance_id': frame.instance_id,
                  'label': [FULL_LABELS[index] for index in four.tolist()]}).to_csv(prediction_path, index=False)
    # 只给原始行评分：生成的反向单例不算作新的Dev/Train评估样本。
    report = score_files(gold_path, prediction_path, output_dir=score_dir)
    n = report.classification.per_label['NEGATIVE_OTHER']
    record = {'method': 'pair_four', 'weighted_f1': report.weighted_f1,
              'accuracy': report.classification.accuracy, 'soft_cons': report.soft_cons,
              'hard_cons': report.hard_cons, 'N_precision': n.precision, 'N_recall': n.recall,
              'N_f1': n.f1, 'N_correct': n.true_positive}
    for short, full in zip(LABELS[:3], FULL_LABELS[:3]):
        metrics = report.classification.per_label[full]
        record[f'{short}_correct'], record[f'{short}_f1'] = metrics.true_positive, metrics.f1
    record['FBE_correct'] = record['F_correct'] + record['B_correct'] + record['E_correct']
    summary = pd.concat([summary, pd.DataFrame([record])], ignore_index=True)
    gold = saved.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    decoding_changes = []
    for method, column in METHODS:
        predicted = saved[column]
        index = summary.index[summary.method.eq(method)][0]
        counts = {
            'nonN_to_N': int((gold.ne('N') & predicted.eq('N')).sum()),
            'E_to_FB': int((gold.eq('E') & predicted.isin(['F', 'B'])).sum()),
            'FB_to_E': int((gold.isin(['F', 'B']) & predicted.eq('E')).sum()),
            'F_B_reversal': int(((gold.eq('F') & predicted.eq('B')) |
                                 (gold.eq('B') & predicted.eq('F'))).sum()),
            'N_to_FB': int((gold.eq('N') & predicted.isin(['F', 'B'])).sum()),
            'N_to_E': int((gold.eq('N') & predicted.eq('E')).sum()),
        }
        for label in LABELS:
            counts[f'{label}_predicted'] = int(predicted.eq(label).sum())
        for name, value in counts.items():
            summary.loc[index, name] = value
        confusion = pd.crosstab(gold.rename('gold'), predicted.rename('prediction')).reindex(
            index=LABELS, columns=LABELS, fill_value=0)
        confusion.to_csv(output_dir / method / 'confusion_matrix.csv', encoding='utf-8-sig')
        before, after = saved.prediction.eq(gold), predicted.eq(gold)
        changed = saved.prediction.ne(predicted)
        decoding_changes.append({'method': method, 'changed': int(changed.sum()),
                                 'fixed': int((~before & after).sum()),
                                 'broken': int((before & ~after).sum()),
                                 'changed_wrong': int((changed & ~before & ~after).sum())})
    pd.DataFrame(decoding_changes).to_csv(output_dir / 'all_decoding_changes.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print('新增四组合解码：')
    print(summary[summary.method.eq('pair_four')].to_string(index=False))
    return summary


def evaluate_phase(model, eval_data, frame, device, output_dir, gold_path):
    loader, reverse_loader, reverse_indices = eval_data
    was_training = model.training
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            values, reverse_values = [], []
            with torch.inference_mode():
                for batch in tqdm(loader, desc='原始行预测'):
                    values.append(model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None).cpu())
                for batch in tqdm(reverse_loader, desc='单例反向预测'):
                    reverse_values.append(model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None).cpu())
            logits = torch.cat(values).float()
            expanded = torch.cat([logits, torch.cat(reverse_values).float()])
            return score_all(logits, expanded[reverse_indices], frame, output_dir, gold_path)
    finally:
        model.train(was_training)


def stage_changes(epoch, split):
    phase_dir = RESULTS_DIR / f'epoch_{epoch}'
    row = pd.read_csv(phase_dir / 'after_row' / split / 'predictions_with_scores.csv')
    pair = pd.read_csv(phase_dir / 'after_pair' / split / 'predictions_with_scores.csv')
    gold = row.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    changes = []
    for method, column in METHODS:
        before, after = row[column].eq(gold), pair[column].eq(gold)
        record = {'epoch': epoch, 'split': split, 'method': method,
                  'changed': int(row[column].ne(pair[column]).sum()),
                  'fixed': int((~before & after).sum()), 'broken': int((before & ~after).sum()),
                  'row_nonN_to_N': int((gold.ne('N') & row[column].eq('N')).sum()),
                  'pair_nonN_to_N': int((gold.ne('N') & pair[column].eq('N')).sum())}
        for label in LABELS:
            record[f'{label}_correct_delta'] = int((gold.eq(label) & after).sum() - (gold.eq(label) & before).sum())
        changes.append(record)
    print(pd.DataFrame(changes).to_string(index=False))
    return changes


def prepare_data():
    train, dev, reference = read_data()
    expanded, pairs, audit = build_pair_data(train)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['training_data.csv', 'extra_train_rows.csv', 'relation_examples_source.csv',
                 'manual_label_notes.csv', 'train_word_sources.csv', 'overlap_report.json', 'data_config.json']:
        (RESULTS_DIR / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (RESULTS_DIR / 'reference_run_config.json').write_bytes((REFERENCE_DIR / 'run_config.json').read_bytes())
    expanded.to_csv(RESULTS_DIR / 'pair_input_rows.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(RESULTS_DIR / 'nn_reverse_audit.csv', index=False, encoding='utf-8-sig')
    groups = pd.DataFrame([{'first_index': first, 'second_index': second,
                           'first_instance_id': expanded.loc[first, 'instance_id'],
                           'second_instance_id': expanded.loc[second, 'instance_id'],
                           'target': ['FB', 'BF', 'EE', 'NN'][target]} for first, second, target in pairs])
    groups.to_csv(RESULTS_DIR / 'pair_groups.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, RESULTS_DIR / 'train_reference.csv')
    print('Row训练行数 / Dev行数：', len(train), '/', len(dev))
    print('Pair组合数量：', groups.target.value_counts().to_dict())
    print('N反向接受 / 冲突跳过：', int(audit.accepted.sum()), '/', int((~audit.accepted).sum()))
    print('冲突N只跳过Pair增强，仍保留在原始Row训练中。')
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)
    return train, dev, expanded, pairs, reference


def main(prepare_only=False):
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换新名字，保留旧模型。')
    train, dev, expanded, pair_indices, reference = prepare_data()
    if prepare_only:
        print('数据准备完成；没有加载模型或启动训练。')
        return
    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    backbone = AutoModel.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
                                         dtype=torch.float32, use_safetensors=True)
    rows = RowDataset(train, tokenizer, MAX_LENGTH)
    pair_rows = RowDataset(expanded, tokenizer, MAX_LENGTH)
    pairs = FourPairDataset(pair_rows, pair_indices)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_eval = build_eval_loaders(RowDataset(dev, tokenizer, MAX_LENGTH), tokenizer)
    train_eval = build_eval_loaders(rows, tokenizer)
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                  label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    config = dict(reference)
    config.update({'run_name': RUN_NAME, 'seed': SEED, 'model_name': MODEL_NAME, 'epochs': EPOCHS,
                   'max_length': MAX_LENGTH, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                   'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
                   'dev_batch_size': DEV_BATCH_SIZE, 'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
                   'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
                   'row_train_rows': len(train), 'pair_input_rows': len(expanded),
                   'pair_groups': len(pairs), 'pair_NN_groups': sum(target == 3 for _, _, target in pair_indices),
                   'real_pairs': sum(target != 3 for _, _, target in pair_indices),
                   'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
                   'optimizer_steps_per_epoch': len(row_loader) + len(pair_loader),
                   'training_order': 'all original Row CE, then shuffled FB/BF/EE/NN Pair CE; same AdamW',
                   'pair_ce_exposure': 'real F/B/E groups plus Train-only conflict-free N reverse groups',
                   'selection': f'best after_pair Dev {SELECT_BY} weighted F1; earliest epoch on ties',
                   'select_by': SELECT_BY, 'average_enabled': False, 'keep_loss_enabled': False,
                   'reference_run': str(REFERENCE_DIR), 'train_audit_epoch': TRAIN_AUDIT_EPOCH,
                   'comparison_note': 'Compared with run09: add NN candidates and N reverse groups; Pair update count increases. Compared with 0.9381: smoothing is also disabled.',
                   'generated_reverse_label': 'N->N is constructed Train supervision; conflicting Train labels are excluded only from NN Pair training.',
                   'evaluation': 'Only original 660 Dev/3290 Train rows are scored; reverse singleton inputs are unlabeled inference companions.',
                   'source_sha256': {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('每轮Row / Pair步数：', len(row_loader), '/', len(pair_loader))
    print('先Row CE，再0.5 × 四组合Pair CE；选模：', SELECT_BY)
    history, changes = [], []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第{epoch}/{EPOCHS}轮：Row =====')
        row_loss = train_rows(model, row_loader, optimizer, row_ce, device)
        for phase in ['after_row', 'after_pair']:
            if phase == 'after_pair':
                pair_loss = train_pairs(model, pair_loader, optimizer, pair_ce, device)
                print(f'Row CE：{row_loss:.6f}；四组合Pair CE：{pair_loss:.6f}')
            print(f'\n===== 第{epoch}轮 {phase}：Dev =====')
            phase_dir = RESULTS_DIR / f'epoch_{epoch}' / phase
            dev_summary = evaluate_phase(model, dev_eval, dev, device, phase_dir / 'dev', GOLD_PATH)
            summaries = [('dev', dev_summary)]
            if epoch == TRAIN_AUDIT_EPOCH:
                print(f'\n===== 第{epoch}轮 {phase}：Train =====')
                train_summary = evaluate_phase(model, train_eval, train, device, phase_dir / 'train',
                                               RESULTS_DIR / 'train_reference.csv')
                summaries.append(('train', train_summary))
            for split, summary in summaries:
                for record in summary.to_dict('records'):
                    record.update(epoch=epoch, phase=phase, split=split, row_loss=row_loss,
                                  pair_loss=pair_loss if phase == 'after_pair' else None)
                    history.append(record)
            pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        score = float(dev_summary.loc[dev_summary.method.eq(SELECT_BY), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            save_weights(model, MODEL_DIR / 'best_model.pth')
            dev_summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            (MODEL_DIR / 'best_metrics.json').write_text(json.dumps(
                {'best_epoch': epoch, 'weighted_f1': score, 'selection_method': SELECT_BY,
                 'selection_phase': 'after_pair'}, indent=2), encoding='utf-8')
            print(f'刷新最佳：第{epoch}轮，{SELECT_BY} F1={score:.6f}')
        for split in (['dev', 'train'] if epoch == TRAIN_AUDIT_EPOCH else ['dev']):
            changes.extend(stage_changes(epoch, split))
        pd.DataFrame(changes).to_csv(RESULTS_DIR / 'phase_changes.csv', index=False, encoding='utf-8-sig')
    comparison = []
    versions = [('reference_0p938', 2, REFERENCE_DIR / 'epoch_2/summary.csv'),
                ('best_pair_nn', best_epoch, RESULTS_DIR / 'best_summary.csv')]
    if EPOCHS >= 2:
        versions.append(('fixed_epoch_2', 2, RESULTS_DIR / 'epoch_2/after_pair/dev/summary.csv'))
    for version, epoch, path in versions:
        for record in pd.read_csv(path).to_dict('records'):
            comparison.append({'version': version, 'epoch': epoch, **record})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False, encoding='utf-8-sig')
    print('\n===== 最终比较：三种读出来自同一个最佳模型 =====')
    print(comparison[['version', 'epoch', 'method', 'weighted_f1', 'N_correct', 'FBE_correct']].to_string(index=False))
    print('最佳模型：', MODEL_DIR / 'best_model.pth')
    print('最终结果：', RESULTS_DIR / 'final_comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(prepare_only=args.prepare_only)
