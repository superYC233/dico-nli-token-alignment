"""右键运行：官方3042 + 原Lexical + 逐样本margin保护；默认先验证1轮。"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.data import DiCoDataset, PairDataset, build_feature_tensor, save_feature_table
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical.margin_constraints import MarginProtection
from experiments.deberta_lexical.evaluation import get_pair_logits, predict_logits
from experiments.deberta_lexical.train_deberta_lexical import set_seed, train_rows
from experiments.deberta_lexical.train_deberta_lexical_anchor import (
    MODEL_NAME, BASE_REVISION, TRAIN_PATH, DEV_PATH, GOLD_PATH, REFERENCE_DIR,
    SEED, ROW_BATCH_SIZE, PAIR_BATCH_SIZE, DEV_BATCH_SIZE, MAX_LENGTH,
    LEARNING_RATE, LEXICAL_LEARNING_RATE, LEXICAL_HIDDEN_SIZE, WEIGHT_DECAY,
    ROW_WEIGHTS, PAIR_WEIGHTS, PAIR_WEIGHT, PAIR_LABEL_SMOOTHING,
    build_model, build_optimizer, read_official_data, file_hash, load_saved_model,
)
from experiments.deberta_stage_average.train_deberta_stage_average import score_predictions, stage_changes
from experiments.deberta_bidirectional.data import write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights


RUN_NAME = 'run08_row_margin_3epochs'
EPOCHS = 10
MARGIN_CAP = 1.0             # Row判对且间隔>0；保护下限=min(原间隔,1)。
INITIAL_DUAL = 0.25          # 四个约束各从0.25开始，不是固定的margin损失权重。
DUAL_LEARNING_RATE = 0.1
PROTECTION_MICROBATCH = 16   # 保护批次分块反向，不与Pair保留大图叠加。
HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical')


class IndexedRows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (*self.rows[index], index)


def predict_preserving_rng(model, loader, device):
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        return predict_logits(model, loader, device)


def save_reference(protection, train, path):
    frame = train[['instance_id', 'label']].copy()
    frame['row_prediction'] = ['FBEN'[i] for i in protection.reference_logits.argmax(1).tolist()]
    frame['row_correct'] = protection.row_correct.tolist()
    frame['protected'] = protection.protected.tolist()
    frame['reference_margin'] = protection.reference_margins.tolist()
    frame['margin_floor'] = protection.floors.tolist()
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def protection_backward(model, batch, protection, device, microbatch=PROTECTION_MICROBATCH):
    ids, masks, features, labels, indices = batch
    assert torch.equal(labels.cpu(), protection.labels[indices.cpu()])
    denominators = protection.counts(indices)
    violations = torch.zeros(4, device=device)
    total_loss = 0.0
    was_training = model.training
    # 参照与当前保护分支都关闭dropout；eval不关闭梯度。
    model.eval()
    try:
        for start in range(0, len(indices), microbatch):
            part = slice(start, start + microbatch)
            active = protection.protected[indices[part].cpu()]
            if not active.any():
                continue
            logits = model(ids[part].to(device), masks[part].to(device), features[part].to(device))
            contribution = protection.constraint_parts(logits, indices[part], denominators)
            loss = protection.primal_term(contribution)
            violations += contribution.detach()
            value = loss.detach().item()
            # 零损坏时梯度也为0，跳过无效反传，保留之前的Pair梯度。
            if value > 0:
                loss.backward()
            total_loss += value
            del logits, loss, contribution
    finally:
        model.train(was_training)
    return violations, total_loss


def train_protected_pairs(model, pair_loader, guard_loader, optimizer, criterion,
                          device, protection, enabled=True):
    model.train()
    guard_batches = iter(guard_loader) if enabled else None
    records = []
    total_pair, total_guard, guard_rows = 0.0, 0.0, 0
    started = time.perf_counter()
    for step, batch in enumerate(tqdm(pair_loader, desc='Pair CE + Train margin约束'), 1):
        ids0, mask0, features0, ids1, mask1, features1, targets = batch
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        pair_value = pair_loss.item()
        del first, second, pair_loss
        before = protection.multipliers.detach().cpu().tolist()
        if enabled:
            guard_batch = next(guard_batches)
            violations, guard_value = protection_backward(model, guard_batch, protection, device)
            this_guard_rows = len(guard_batch[-1])
        else:
            violations, guard_value, this_guard_rows = torch.zeros(4, device=device), 0.0, 0
        # 所有反传都在同一个theta_t上；保护microbatch共享同一组lambda_t。
        optimizer.step()
        if enabled:
            protection.dual_step(violations)
        after = protection.multipliers.detach().cpu().tolist()
        record = {'step': step, 'pair_loss': pair_value, 'guard_lagrangian': guard_value,
                  'guard_rows': this_guard_rows, 'enabled': enabled}
        for k, label in enumerate('FBEN'):
            record.update({f'{label}_violation': float(violations[k]),
                           f'{label}_lambda_before': before[k], f'{label}_lambda_after': after[k]})
        records.append(record)
        total_pair += pair_value
        total_guard += guard_value
        guard_rows += this_guard_rows
    if enabled:
        assert guard_rows == len(guard_loader.dataset)
    return {'pair_loss': total_pair / len(pair_loader),
            'guard_lagrangian': total_guard / len(pair_loader),
            'pair_train_seconds': time.perf_counter() - started,
            'protection_rows_seen': guard_rows, 'pair_optimizer_steps': len(pair_loader)}, records


def main(run_name=RUN_NAME, epochs=EPOCHS, prepare_only=False, no_protection=False):
    train, dev = read_official_data()
    features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=get_model_cache_dir(),
                                              revision=BASE_REVISION, local_files_only=True)
    rows = DiCoDataset(train, tokenizer, features, MAX_LENGTH)
    pairs = PairDataset(train, tokenizer, features, MAX_LENGTH)
    dev_rows = DiCoDataset(dev, tokenizer, dev_features, MAX_LENGTH)
    pair_steps = math.ceil(len(pairs) / PAIR_BATCH_SIZE)
    guard_batch_size = math.ceil(len(rows) / pair_steps)
    assert math.ceil(len(rows) / guard_batch_size) == pair_steps
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    print('数据：官方Train3042 / Dev660；原Lexical；真实Pair1282组。')
    print('每轮：Row CE → 记录Train正确样本margin → Pair CE及margin约束 → Train/Dev评估。')
    print('保护：逐样本非负损坏，F/B/E/N四个乘子；包含N单例，不读取Dev保护目标。')
    print('保护批次 / microbatch：', guard_batch_size, '/', PROTECTION_MICROBATCH)
    print('margin上限 / 乘子初值 / dual lr：', MARGIN_CAP, '/', INITIAL_DUAL, '/', DUAL_LEARNING_RATE)
    print('训练轮数：', epochs, '；启用保护：', not no_protection)
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('准备检查完成，没有加载模型权重或训练。')
        return
    if (model_dir / 'run_config.json').exists() or (output_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有运行记录，请修改RUN_NAME，保留旧结果。')
    assert epochs >= 1
    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False)
    dev_loader = DataLoader(dev_rows, batch_size=DEV_BATCH_SIZE, shuffle=False)
    # 保护顺序独立，额外DataLoader不改变原Row/Pair的shuffle或dropout序列。
    guard_generator = torch.Generator().manual_seed(SEED + 1009)
    guard_loader = DataLoader(IndexedRows(rows), batch_size=guard_batch_size, shuffle=True,
                              generator=guard_generator)
    labels = torch.tensor(train.label.map(dict(zip([
        'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER',
    ], range(4)))).to_numpy())
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                         label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    reference_path = output_dir / 'train_reference.csv'
    write_train_reference(train, reference_path)
    save_feature_table(train, features, output_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, output_dir / 'dev_features.csv')
    sources = [Path(__file__).resolve()] + [HERE / name for name in [
        'margin_constraints.py', 'model.py', 'data.py', 'lexical_features.py',
        'evaluation.py', 'train_deberta_lexical.py', 'train_deberta_lexical_anchor.py',
    ]]
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION,
        'seed': SEED, 'epochs': epochs, 'train_rows': len(train), 'dev_rows': len(dev),
        'real_pairs': len(pairs), 'official_train_only': True, 'data_augmentation': False,
        'train_sha256': file_hash(TRAIN_PATH), 'dev_sha256': file_hash(DEV_PATH),
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'feature_source_sha256': file_hash(HERE / 'lexical_features.py'),
        'lexical_hidden_size': LEXICAL_HIDDEN_SIZE, 'use_lexical_features': True,
        'learning_rate': LEARNING_RATE, 'lexical_learning_rate': LEXICAL_LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
        'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'dev_batch_size': DEV_BATCH_SIZE, 'max_length': MAX_LENGTH, 'dtype': 'float32',
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
        'guard_batch_size': guard_batch_size, 'guard_microbatch': PROTECTION_MICROBATCH,
        'guard_rows_per_epoch': len(train) if not no_protection else 0,
        'protection_enabled': not no_protection, 'margin_cap': MARGIN_CAP,
        'initial_dual': INITIAL_DUAL, 'dual_learning_rate': DUAL_LEARNING_RATE,
        'constraint': 'class mean of nonnegative per-row margin damage <= 0; only Row-correct Train rows with positive margin',
        'dual': 'four class multipliers; simultaneous primal-dual; reset after every Row reference refresh',
        'reference': 'full official Train after Row, eval mode; no Dev data or reference parameters',
        'primal': 'unchanged shared AdamW on 0.5*Pair CE + sum(lambda_t * class_damage)',
        'dual_update': 'lambda_next=clamp_min(lambda_t+dual_lr*pre_update_violation,0)',
        'implementation': 'self-contained Cooper-style simultaneous primal-dual adaptation, not the cooper package or a paper reproduction',
        'source': 'https://cooper.readthedocs.io/en/latest/_modules/cooper/optim/constrained_optimizers/simultaneous_optimizer.html',
        'guarantee': 'sampled stochastic constraints; no per-step or full-Train feasibility guarantee; measured by audits',
        'selection': 'best AFTER-PAIR Dev pair-aware F1; earliest on ties; same-checkpoint independent reported',
        'inference': 'unchanged FB/BF/EE pair-aware; singleton remains four-class model argmax',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'reference_run': str(REFERENCE_DIR),
        'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    stages, batches, audits, changes = [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        print(f'\n===== 第{epoch}/{epochs}轮 =====')
        model.train()
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        print('记录本轮完整Train参照；dropout关闭，不更新参数。')
        train_before = predict_preserving_rng(model, train_loader, device)
        protection = MarginProtection(train_before, labels, device, MARGIN_CAP,
                                      INITIAL_DUAL, DUAL_LEARNING_RATE)
        row_dir.mkdir(parents=True, exist_ok=True)
        save_reference(protection, train, row_dir / 'train_margin_reference.csv')
        print('Train保护样本数量：', dict(zip('FBEN', protection.counts(torch.arange(len(rows))).int().cpu().tolist())))
        print(f'\n===== 第{epoch}轮 Row后：Train =====')
        row_train_summary = score_predictions(train_before, train, row_dir / 'train', reference_path)
        print(f'\n===== 第{epoch}轮 Row后：Dev =====')
        row_dev_summary = score_predictions(predict_preserving_rng(model, dev_loader, device),
                                            dev, row_dir / 'dev', GOLD_PATH)
        result, records = train_protected_pairs(model, pair_loader, guard_loader, optimizer,
                                                pair_criterion, device, protection, not no_protection)
        print('检查Pair后完整Train，验证保护是否实际做到。')
        train_after = predict_preserving_rng(model, train_loader, device)
        audit = protection.audit(train_after)
        print(pd.DataFrame(audit).to_string(index=False))
        for item in audit:
            item['epoch'] = epoch
        audits.extend(audit)
        pair_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit).to_csv(pair_dir / 'margin_audit.csv', index=False, encoding='utf-8-sig')
        print(f'\n===== 第{epoch}轮 Pair后：Train =====')
        pair_train_summary = score_predictions(train_after, train, pair_dir / 'train', reference_path)
        print(f'\n===== 第{epoch}轮 Pair后：Dev =====')
        pair_dev_summary = score_predictions(predict_preserving_rng(model, dev_loader, device),
                                             dev, pair_dir / 'dev', GOLD_PATH)
        print('本轮：', {'row_loss': row_loss, **result})
        for phase, split, summary in [
            ('after_row', 'train', row_train_summary), ('after_row', 'dev', row_dev_summary),
            ('after_pair', 'train', pair_train_summary), ('after_pair', 'dev', pair_dev_summary),
        ]:
            for item in summary.to_dict('records'):
                item.update(epoch=epoch, phase=phase, split=split, row_loss=row_loss, **result)
                stages.append(item)
        for record in records:
            record['epoch'] = epoch
        batches.extend(records)
        for split in ['train', 'dev']:
            changes.extend(stage_changes(epoch, split, row_dir / split, pair_dir / split))
        pd.DataFrame(stages).to_csv(output_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(batches).to_csv(output_dir / 'constraint_batches.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(audits).to_csv(output_dir / 'margin_audit.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(changes).to_csv(output_dir / 'stage_changes.csv', index=False, encoding='utf-8-sig')
        score = float(pair_dev_summary.loc[pair_dev_summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected = {'epoch': epoch, 'phase': 'after_pair', 'pair_aware_f1': score,
                        'independent_f1': float(pair_dev_summary.loc[pair_dev_summary.method.eq('independent'), 'weighted_f1'].iloc[0]),
                        'train_protected_broken': sum(item['protected_broken'] for item in audit),
                        'train_protected': sum(item['protected'] for item in audit)}
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            pair_dev_summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新最佳模型：', selected)
        del protection, train_before, train_after
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_weights(model, model_dir / 'best_model.pth')
    score_predictions(predict_preserving_rng(model, dev_loader, device),
                       dev, output_dir / 'best/dev', GOLD_PATH)
    old = pd.read_csv(output_dir / f'epoch_{best_epoch}/after_pair/dev/predictions_with_scores.csv')
    new = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert old.instance_id.tolist() == new.instance_id.tolist()
    assert old[['prediction', 'pair_prediction']].equals(new[['prediction', 'pair_prediction']])
    print('最佳轮次：', best_epoch, '；pair-aware F1：', best_score)
    print('最佳完整模型：', model_dir / 'best_model.pth')
    print('约束审计：', output_dir / 'margin_audit.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-protection', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.epochs, args.prepare_only, args.no_protection)
