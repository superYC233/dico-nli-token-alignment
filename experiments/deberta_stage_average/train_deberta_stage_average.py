"""沿用0.9381配方：观察Row/Pair阶段，并等权平均第6—10轮。"""
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
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset, GroupDataset
from experiments.deberta_lexical.data import validate_groups
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, score_logits
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset, set_seed, save_weights


RUN_NAME = 'run02_3042_only'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
AVERAGE_START = 6
TRAIN_AUDIT_EPOCH = 2
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
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_stage_average') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME


# def read_data():
#     # 直接读取0.9381那次的训练快照，不重新造样本，也不加载旧权重续训。
#     reference = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
#     train = pd.read_csv(REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
#     dev = pd.read_csv(DEV_PATH, keep_default_na=False)
#     train_hash = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
#     assert train_hash == reference['augmentation']['effective_train_sha256']
#     assert hashlib.sha256(DEV_PATH.read_bytes()).hexdigest() == reference['dev_sha256']
#     validate_groups(train)
#     validate_groups(dev)
#     assert len(train) == 3290 and len(dev) == 660
#     return train, dev, reference

def read_data():
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    validate_groups(train)
    validate_groups(dev)
    assert len(train) == 3042 and len(dev) == 660
    return train, dev, None

def write_train_reference(train, path):
    # 官方评分需要两行之间的反向ID；N单例没有反向行。
    reverse_ids = {}
    for _, group in train.groupby('pair_id'):
        if len(group) == 2:
            first, second = group.instance_id.tolist()
            reverse_ids[first], reverse_ids[second] = second, first
    reference = train.copy()
    reference['reverse_pair_id'] = reference.instance_id.map(reverse_ids).fillna('')
    reference.to_csv(path, index=False, encoding='utf-8-sig')


def prepare_data():
    train, dev, reference = read_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_train_reference(train, RESULTS_DIR / 'train_reference.csv')
    print('训练数据：仅官方 Train =', len(train))
    print('Dev：', len(dev))
    print('模型目录：', MODEL_DIR)
    print('结果目录：', RESULTS_DIR)
    return train, dev, reference


def train_rows(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='四分类 Row CE'):
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
        loss = criterion(logits, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for ids0, mask0, ids1, mask1, targets in tqdm(loader, desc='真实 Pair CE'):
        optimizer.zero_grad()
        first = model(ids0.to(device), mask0.to(device), None)
        second = model(ids1.to(device), mask1.to(device), None)
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        total += pair_loss.item()
    return total / len(loader)


def score_predictions(logits, frame, output_dir, gold_path):
    summary = score_logits(logits, frame, output_dir, gold_path=gold_path)
    predictions = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    gold = predictions.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        predicted = predictions[column]
        index = summary.index[summary.method.eq(method)][0]
        summary.loc[index, 'nonN_to_N'] = int((gold.ne('N') & predicted.eq('N')).sum())
        for label in LABELS:
            summary.loc[index, f'{label}_predicted'] = int(predicted.eq(label).sum())
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    before = predictions.prediction.eq(gold)
    after = predictions.pair_prediction.eq(gold)
    changed = predictions.prediction.ne(predictions.pair_prediction)
    changes = {'changed': int(changed.sum()), 'fixed': int((~before & after).sum()),
               'broken': int((before & ~after).sum()),
               'changed_wrong': int((changed & ~before & ~after).sum())}
    (output_dir / 'decoding_changes.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    print('独立非N→N：', int(summary.loc[summary.method.eq('independent'), 'nonN_to_N'].iloc[0]))
    return summary


def evaluate_phase(model, loader, frame, device, output_dir, gold_path):
    was_training = model.training
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    # 额外评估不应改变接下来的shuffle/dropout随机序列；评分也不更新参数。
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            values = []
            with torch.inference_mode():
                for batch in tqdm(loader, desc='逐行预测'):
                    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
                    values.append(logits.float().cpu())
            return score_predictions(torch.cat(values), frame, output_dir, gold_path)
    finally:
        model.train(was_training)


def stage_changes(epoch, split, row_dir, pair_dir):
    row = pd.read_csv(row_dir / 'predictions_with_scores.csv', keep_default_na=False)
    pair = pd.read_csv(pair_dir / 'predictions_with_scores.csv', keep_default_na=False)
    assert row.instance_id.tolist() == pair.instance_id.tolist()
    gold = row.gold_label.map(dict(zip(FULL_LABELS, LABELS)))
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before, after = row[column].eq(gold), pair[column].eq(gold)
        item = {'epoch': epoch, 'split': split, 'method': method,
                'changed': int(row[column].ne(pair[column]).sum()),
                'fixed': int((~before & after).sum()), 'broken': int((before & ~after).sum()),
                'row_nonN_to_N': int((gold.ne('N') & row[column].eq('N')).sum()),
                'pair_nonN_to_N': int((gold.ne('N') & pair[column].eq('N')).sum())}
        for label in LABELS:
            item[f'{label}_correct_delta'] = int((gold.eq(label) & after).sum() - (gold.eq(label) & before).sum())
        records.append(item)
    print('Pair阶段前后变化：')
    print(pd.DataFrame(records).to_string(index=False))
    return records


def check_trajectory(epoch, current_dir):
    old = pd.read_csv(REFERENCE_DIR / f'epoch_{epoch}/predictions_with_scores.csv', keep_default_na=False)
    new = pd.read_csv(current_dir / 'predictions_with_scores.csv', keep_default_na=False)
    assert old.instance_id.tolist() == new.instance_id.tolist()
    return {'epoch': epoch, 'independent_differences': int(old.prediction.ne(new.prediction).sum()),
            'pair_aware_differences': int(old.pair_prediction.ne(new.pair_prediction).sum())}


def main(prepare_only=False):
    if MODEL_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请换一个RUN_NAME，保留旧模型。')
    train, dev, reference = prepare_data()
    if prepare_only:
        print('数据准备完成，没有加载模型或启动训练。')
        return

    set_seed(SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    groups = GroupDataset(train, tokenizer, MAX_LENGTH)
    rows, pairs = groups.rows, RealPairDataset(groups.rows)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(RowDataset(dev, tokenizer, MAX_LENGTH), batch_size=DEV_BATCH_SIZE,
                            shuffle=False, generator=torch.Generator().manual_seed(SEED + 1))
    train_eval_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False,
                                   generator=torch.Generator().manual_seed(SEED + 2))
    # 在CPU创建平均副本，再把训练模型送上GPU，避免复制整份GPU权重的峰值。
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    average = AveragedModel(model, device='cpu')  # 默认等权平均；第一次update才开始计数。
    model.to(device)
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
              'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS, 'pair_weight': PAIR_WEIGHT,
              'pair_label_smoothing': PAIR_LABEL_SMOOTHING, 'train_rows': len(rows), 'real_pairs': len(pairs),
              'freeze_first_n_layers': 0, 'use_lexical_features': False,
              'augmentation': 'none (official 3042 only)', 'reference_run': str(REFERENCE_DIR),
              'training_order': 'all Row CE batches, then all Pair CE batches; unchanged optimizer',
              'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
              'average': {'start_epoch': AVERAGE_START, 'end_epoch': EPOCHS, 'device': 'cpu',
                          'method': 'equal weight averaging after complete Pair stage; unchanged learning rate',
                          'selection': 'fixed final average; no Dev choice of window or average checkpoint'},
              'train_audit_epoch': TRAIN_AUDIT_EPOCH, 'evaluation_preserves_torch_rng': True,
              'online_selection': 'best Dev pair-aware F1 AFTER Pair; earliest epoch on ties',
              'evaluation_note': 'Train uses training labels for diagnosis only; Dev never enters either loss.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for folder in [MODEL_DIR, RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Row / Pair步数：', len(row_loader), '/', len(pair_loader))
    print('固定平均窗口：', AVERAGE_START, '—', EPOCHS, '轮；CPU等权平均。')

    history, changes, trajectory = [], [], []
    best_f1, best_epoch, best_summary = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        for phase in ['after_row', 'after_pair']:
            if phase == 'after_pair':
                pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
            print(f'\n===== 第{epoch}轮 {phase}：Dev =====')
            output_dir = RESULTS_DIR / f'epoch_{epoch}' / phase
            summary = evaluate_phase(model, dev_loader, dev, device, output_dir / 'dev', GOLD_PATH)
            for item in summary.to_dict('records'):
                item.update(epoch=epoch, phase=phase, split='dev', row_loss=row_loss,
                            pair_loss=pair_loss if phase == 'after_pair' else None)
                history.append(item)
            if epoch == TRAIN_AUDIT_EPOCH:
                print(f'\n===== 第{epoch}轮 {phase}：完整Train诊断 =====')
                train_summary = evaluate_phase(model, train_eval_loader, train, device, output_dir / 'train',
                                               RESULTS_DIR / 'train_reference.csv')
                for item in train_summary.to_dict('records'):
                    item.update(epoch=epoch, phase=phase, split='train', row_loss=row_loss,
                                pair_loss=pair_loss if phase == 'after_pair' else None)
                    history.append(item)
            pd.DataFrame(history).to_csv(RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')

        # summary此时对应本轮Pair结束后的Dev；Row阶段结果不参与最佳单轮选模。
        score = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_f1:
            best_f1, best_epoch, best_summary = score, epoch, summary.copy()
            save_weights(model, MODEL_DIR / 'best_model.pth')
            (MODEL_DIR / 'best_metrics.json').write_text(
                json.dumps({'best_epoch': epoch, 'weighted_f1': score, 'selection_phase': 'after_pair'}, indent=2),
                encoding='utf-8')
            summary.to_csv(RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print(f'最佳单轮更新：epoch {epoch}，pair-aware F1={score:.6f}')
        if epoch >= AVERAGE_START:
            average.update_parameters(model)
            print('已平均的轮数：', int(average.n_averaged))
        for split in (['dev', 'train'] if epoch == TRAIN_AUDIT_EPOCH else ['dev']):
            changes.extend(stage_changes(epoch, split, RESULTS_DIR / f'epoch_{epoch}/after_row' / split,
                                         RESULTS_DIR / f'epoch_{epoch}/after_pair' / split))
        pd.DataFrame(changes).to_csv(RESULTS_DIR / 'phase_changes.csv', index=False, encoding='utf-8-sig')
        trajectory.append(check_trajectory(epoch, RESULTS_DIR / f'epoch_{epoch}/after_pair/dev'))
        pd.DataFrame(trajectory).to_csv(RESULTS_DIR / 'trajectory_comparison.csv', index=False, encoding='utf-8-sig')

    last_summary = summary.copy()
    assert int(average.n_averaged) == EPOCHS - AVERAGE_START + 1
    # 训练已结束：平均参数复制进同一GPU模型，不再创建第二份GPU模型。
    model.load_state_dict(average.module.state_dict(), strict=True)
    save_weights(model, MODEL_DIR / 'averaged_model.pth')
    print(f'\n===== 固定第{AVERAGE_START}—{EPOCHS}轮平均模型：Dev =====')
    averaged_summary = evaluate_phase(model, dev_loader, dev, device, RESULTS_DIR / 'averaged_final/dev', GOLD_PATH)
    versions = [('best_online', best_epoch, best_summary), ('last_online', EPOCHS, last_summary),
                ('fixed_average', f'{AVERAGE_START}-{EPOCHS}', averaged_summary)]
    comparison = []
    for version, epoch, frame in versions:
        for item in frame.to_dict('records'):
            comparison.append({'version': version, 'epoch_or_window': epoch, **item})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False, encoding='utf-8-sig')
    print('\n===== 最终比较 =====')
    print(comparison[['version', 'epoch_or_window', 'method', 'weighted_f1', 'N_correct',
                      'FBE_correct', 'nonN_to_N']].to_string(index=False))
    print('最佳单轮：', MODEL_DIR / 'best_model.pth')
    print('固定平均：', MODEL_DIR / 'averaged_model.pth')
    print('阶段诊断：', RESULTS_DIR / 'phase_changes.csv')
    print('最终比较：', RESULTS_DIR / 'final_comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='只复制并核对数据，不加载模型或训练')
    args = parser.parse_args()
    train, dev, reference = prepare_data()
    main(prepare_only=args.prepare_only)
