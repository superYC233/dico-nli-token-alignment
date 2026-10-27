"""原Row→Pair训练：Pair CE + 四类输出保持；不做权重平均。"""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import (
    read_data, write_train_reference, train_rows, evaluate_phase, score_predictions, stage_changes,
    REFERENCE_DIR, GOLD_PATH, LABELS,
)


RUN_NAME = 'run01_keep1_t2'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

# 这次只增加这一项；KEEP_WEIGHT=0可作为同代码的原Pair CE对照。
KEEP_WEIGHT = 1.0
TEMPERATURE = 2.0
TRAIN_AUDIT_EPOCH = 2

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_row_anchor') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME


class AnchoredPairDataset(RealPairDataset):
    def __getitem__(self, index):
        first, second, _ = self.pairs[index]
        values = super().__getitem__(index)
        # 保存原训练行索引；shuffle只改变组顺序，不改变缓存与文本的对应。
        return (*values, torch.tensor([first, second], dtype=torch.long))


def cache_train_logits(model, loader, labels, device):
    """只预测Train，缓存本轮Row结束时的分数；不创建另一个教师模型。"""
    was_training = model.training
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            values = []
            with torch.inference_mode():
                for batch in tqdm(loader, desc='缓存本轮Row的Train分数'):
                    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
                    values.append(logits.float().cpu())
            # 在inference_mode外合并，得到可安全用作KL目标的普通无梯度张量。
            row_logits = torch.cat(values).detach()
            correct = row_logits.argmax(dim=1).eq(labels.cpu())
            return row_logits, correct
    finally:
        model.train(was_training)


def compute_keep_loss(student_logits, row_logits, correct, temperature):
    # 只保持Row阶段已经答对的行，错行不强迫复制旧判断。
    correct = correct.to(student_logits.device)
    if not correct.any():
        return student_logits.sum() * 0.0
    student = student_logits[correct].float()
    teacher = row_logits.detach().to(student_logits.device)[correct].float()
    # 必须在F/B/E/N四列上归一化：N分数漂移也会受到惩罚。
    teacher_probability = F.softmax(teacher / temperature, dim=-1)
    student_log_probability = F.log_softmax(student / temperature, dim=-1)
    # PyTorch参数顺序是student log-prob、teacher prob，计算KL(teacher || student)。
    return F.kl_div(student_log_probability, teacher_probability,
                    reduction='batchmean') * temperature ** 2


def compute_pair_stage_loss(first, second, targets, row_indices, cache, correct,
                            criterion, keep_weight, temperature):
    pair_loss = criterion(get_pair_logits(first, second), targets)
    if keep_weight == 0:
        keep_loss = first.new_zeros(())
    else:
        # first是全批第一行，second是全批第二行；索引必须按同样顺序拼接。
        indices = torch.cat([row_indices[:, 0], row_indices[:, 1]])
        student = torch.cat([first, second], dim=0)
        keep_loss = compute_keep_loss(student, cache[indices], correct[indices], temperature)
    loss = PAIR_WEIGHT * pair_loss + keep_weight * keep_loss
    return loss, pair_loss, keep_loss


def train_pairs(model, loader, optimizer, criterion, cache, correct, device,
                keep_weight=KEEP_WEIGHT, temperature=TEMPERATURE):
    model.train()
    totals = {'pair_loss': 0.0, 'keep_loss': 0.0, 'pair_stage_loss': 0.0}
    kept_rows = 0
    for ids0, mask0, ids1, mask1, targets, indices in tqdm(loader, desc='Pair CE + 四类输出保持'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        loss, pair_loss, keep_loss = compute_pair_stage_loss(
            first, second, targets.to(device), indices, cache, correct,
            criterion, keep_weight, temperature,
        )
        loss.backward()
        optimizer.step()
        totals['pair_loss'] += pair_loss.item()
        totals['keep_loss'] += keep_loss.item()
        totals['pair_stage_loss'] += loss.item()
        kept_rows += int(correct[indices].sum())
    result = {name: value / len(loader) for name, value in totals.items()}
    result['weighted_pair_loss'] = PAIR_WEIGHT * result['pair_loss']
    result['weighted_keep_loss'] = keep_weight * result['keep_loss']
    result['eligible_pair_rows'] = kept_rows
    return result


def prepare_data():
    train, dev, reference = read_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['training_data.csv', 'extra_train_rows.csv', 'relation_examples_source.csv',
                 'manual_label_notes.csv', 'train_word_sources.csv', 'overlap_report.json', 'data_config.json']:
        (RESULTS_DIR / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    (RESULTS_DIR / 'reference_run_config.json').write_bytes((REFERENCE_DIR / 'run_config.json').read_bytes())
    write_train_reference(train, RESULTS_DIR / 'train_reference.csv')
    print('Train / Dev：', len(train), '/', len(dev))
    print('数据：直接读取0.9381的3290行快照，不重新生成样本。')
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)
    return train, dev, reference


def save_cache(cache, correct, rows, output_dir):
    # 只保存小型分数表，不保存教师模型；每轮都能检查是哪一行参与保持。
    torch.save({'instance_ids': rows.frame.instance_id.tolist(),
                'logits': cache, 'correct': correct}, output_dir / 'train_row_cache.pt')
    records = []
    probabilities = F.softmax(cache / TEMPERATURE, dim=-1)
    for index, label in enumerate(LABELS):
        selected = rows.labels.eq(index)
        records.append({'label': label, 'rows': int(selected.sum()),
                        'row_correct': int((selected & correct).sum()),
                        'mean_max_probability_at_T': float(probabilities[selected].max(dim=1).values.mean()),
                        'directly_used_in_pair_keep': label != 'N'})
    pd.DataFrame(records).to_csv(output_dir / 'cache_summary.csv', index=False, encoding='utf-8-sig')
    print('Row缓存答对行数：', int(correct.sum()), '/', len(correct))
    return records


def main(prepare_only=False):
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换新名字，保留旧结果。')
    train, dev, reference = prepare_data()
    if prepare_only:
        print('仅准备数据，没有加载模型或训练。')
        return

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    groups = GroupDataset(train, tokenizer, MAX_LENGTH)
    rows, pairs = groups.rows, AnchoredPairDataset(groups.rows)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE,
                            shuffle=False, generator=torch.Generator().manual_seed(SEED + 1))
    train_eval_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
                                   generator=torch.Generator().manual_seed(SEED + 2))
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                        label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(MODEL_DIR / 'tokenizer')
    config = {'run_name': RUN_NAME, 'model_name': MODEL_NAME, 'seed': SEED, 'epochs': EPOCHS,
              'max_length': MAX_LENGTH, 'dtype': 'float32', 'learning_rate': LEARNING_RATE,
              'weight_decay': WEIGHT_DECAY, 'row_batch_size': ROW_BATCH_SIZE,
              'pair_batch_size': PAIR_BATCH_SIZE, 'dev_batch_size': DEV_BATCH_SIZE,
              'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
              'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
              'keep_weight': KEEP_WEIGHT, 'temperature': TEMPERATURE,
              'keep_definition': 'T^2 * KL(cached Row distribution || current Pair distribution), four labels',
              'keep_mask': 'cached argmax matches Train gold; recomputed after Row each epoch',
              'keep_scope': 'paired Train rows only; N singletons are not directly replayed in Pair stage',
              'cache_source': 'current epoch after Row; Train only; eval + inference_mode; detached CPU cache',
              'augmentation': reference['augmentation'], 'train_rows': len(rows), 'real_pairs': len(pairs),
              'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
              'training_order': 'all Row CE, then all Pair CE + keep; same optimizer',
              'average_enabled': False, 'use_lexical_features': False, 'freeze_first_n_layers': 0,
              'train_audit_epoch': TRAIN_AUDIT_EPOCH,
              'selection': {'best_row': 'after_row Dev independent weighted F1',
                            'best_pair': 'after_pair Dev pair-aware weighted F1'},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Row / Pair步数：', len(row_loader), '/', len(pair_loader))
    print(f'Pair阶段：{PAIR_WEIGHT} * Pair CE + {KEEP_WEIGHT} * 四类保持；T={TEMPERATURE}')

    history, changes, cache_history = [], [], []
    best = {'row': {'epoch': 0, 'score': -1.0}, 'pair': {'epoch': 0, 'score': -1.0}}
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第{epoch}/{EPOCHS}轮：Row =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        row_dir = RESULTS_DIR / f'epoch_{epoch}/after_row'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir / 'dev', GOLD_PATH)
        cache, correct = cache_train_logits(model, train_eval_loader, rows.labels, device)
        for item in save_cache(cache, correct, rows, row_dir):
            cache_history.append({'epoch': epoch, **item})
        pd.DataFrame(cache_history).to_csv(RESULTS_DIR / 'cache_summary.csv', index=False, encoding='utf-8-sig')
        summaries = [('after_row', 'dev', row_summary)]
        if epoch == TRAIN_AUDIT_EPOCH:
            train_summary = score_predictions(cache, train, row_dir / 'train', RESULTS_DIR / 'train_reference.csv')
            summaries.append(('after_row', 'train', train_summary))

        score = float(row_summary.loc[row_summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        if score > best['row']['score']:
            best['row'] = {'epoch': epoch, 'score': score}
            save_weights(model, MODEL_DIR / 'best_row_model.pth')
            row_summary.to_csv(RESULTS_DIR / 'best_row_summary.csv', index=False, encoding='utf-8-sig')

        print(f'\n===== 第{epoch}/{EPOCHS}轮：Pair + keep =====')
        losses = train_pairs(model, pair_loader, optimizer, pair_criterion, cache, correct, device)
        print('Pair阶段损失分量：', losses)
        pair_dir = RESULTS_DIR / f'epoch_{epoch}/after_pair'
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir / 'dev', GOLD_PATH)
        summaries.append(('after_pair', 'dev', pair_summary))
        if epoch == TRAIN_AUDIT_EPOCH:
            train_summary = evaluate_phase(model, train_eval_loader, train, device, pair_dir / 'train',
                                           RESULTS_DIR / 'train_reference.csv')
            summaries.append(('after_pair', 'train', train_summary))
        for phase, split, summary in summaries:
            for item in summary.to_dict('records'):
                item.update(epoch=epoch, phase=phase, split=split, row_loss=row_loss)
                if phase == 'after_pair':
                    item.update(losses)
                history.append(item)
        pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        for split in (['dev', 'train'] if epoch == TRAIN_AUDIT_EPOCH else ['dev']):
            changes.extend(stage_changes(epoch, split, row_dir / split, pair_dir / split))
        pd.DataFrame(changes).to_csv(RESULTS_DIR / 'phase_changes.csv', index=False, encoding='utf-8-sig')
        score = float(pair_summary.loc[pair_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best['pair']['score']:
            best['pair'] = {'epoch': epoch, 'score': score}
            save_weights(model, MODEL_DIR / 'best_pair_model.pth')
            pair_summary.to_csv(RESULTS_DIR / 'best_pair_summary.csv', index=False, encoding='utf-8-sig')
        (MODEL_DIR / 'best_metrics.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
        print('当前最佳：', best)

    comparison = []
    for version, epoch, path in [
        ('reference_0p938', 2, REFERENCE_DIR / 'epoch_2/summary.csv'),
        ('best_row', best['row']['epoch'], RESULTS_DIR / 'best_row_summary.csv'),
        ('best_pair', best['pair']['epoch'], RESULTS_DIR / 'best_pair_summary.csv'),
    ]:
        for item in pd.read_csv(path).to_dict('records'):
            comparison.append({'version': version, 'epoch': epoch, **item})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False, encoding='utf-8-sig')
    print('\n===== 最终比较 =====')
    print(comparison[['version', 'epoch', 'method', 'weighted_f1', 'N_correct', 'FBE_correct']].to_string(index=False))
    print('最佳Row模型：', MODEL_DIR / 'best_row_model.pth')
    print('最佳Pair模型：', MODEL_DIR / 'best_pair_model.pth')
    print('最终比较：', RESULTS_DIR / 'final_comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='只准备数据，不加载模型或训练')
    args = parser.parse_args()
    main(prepare_only=args.prepare_only)
