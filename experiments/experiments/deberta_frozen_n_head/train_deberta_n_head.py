"""右键运行：冻结0.9381，只训练N头；同一轮同时评估独立与pair-aware。"""
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.evaluation import pair_aware
from experiments.deberta_manual_aug.train_boundary_manual import set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import (
    read_data, write_train_reference, score_predictions,
)
from experiments.deberta_frozen_n_head import config as cfg
from experiments.deberta_frozen_n_head.model import build_model


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_cache(model, frame, tokenizer, device, description):
    rows = RowDataset(frame, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(
        rows, batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )
    model.eval()
    logits, features = [], []
    # 使用no_grad，不使用inference_mode：缓存还要作为新头反传的输入。
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            values, vectors = model.extract_features(
                batch['input_ids'].to(device), batch['attention_mask'].to(device),
            )
            logits.append(values.float().cpu())
            features.append(vectors.float().cpu())
    # 不缓存标签。训练标签由Train另行提供，Dev只用于评分。
    return {'base_logits': torch.cat(logits), 'features': torch.cat(features)}


@torch.no_grad()
def predict_cache(model, cache, device):
    model.eval()
    values = []
    for start in range(0, len(cache['features']), cfg.DEV_BATCH_SIZE):
        stop = start + cfg.DEV_BATCH_SIZE
        logits = model.logits_from_features(
            cache['base_logits'][start:stop].to(device),
            cache['features'][start:stop].to(device),
        )
        values.append(logits.float().cpu())
    return torch.cat(values)


def train_head(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        n_logit = model.n_head(features.to(device)).squeeze(-1)
        loss = criterion(n_logit, targets.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


def selection_record(summary, epoch, target_independent, target_pair):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    # 两个指标取同一轮、同一套分数；优先改善距离目标更远的那一项。
    score = min(independent / target_independent, paired / target_pair)
    return {'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': paired,
            'joint_score': score,
            'targets_met': independent >= target_independent and paired >= target_pair}


def check_reference(cache, dev):
    output_dir = cfg.RESULTS_DIR / 'reference_check'
    summary = score_predictions(cache['base_logits'], dev, output_dir, cfg.GOLD_PATH)
    old = pd.read_csv(cfg.REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    current = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    assert old.instance_id.tolist() == current.instance_id.tolist()
    differences = {
        'independent_differences': int(old.prediction.ne(current.prediction).sum()),
        'pair_aware_differences': int(old.pair_prediction.ne(current.pair_prediction).sum()),
    }
    assert all(value == 0 for value in differences.values()), '参考模型未复现，请核对旧checkpoint。'
    (output_dir / 'reference_check.json').write_text(
        json.dumps(differences, indent=2), encoding='utf-8',
    )
    print('与0.9381历史预测逐条比较：', differences)
    return summary


def check_fixed_relations(logits, cache, frame):
    # 这是评估后的核查，组大小不会作为N头的输入。
    assert torch.equal(logits[:, :3], cache['base_logits'][:, :3])
    paired_rows = torch.tensor(frame.groupby('pair_id').instance_id.transform('size').eq(2).to_numpy())
    before = pair_aware(cache['base_logits'], frame)
    after = pair_aware(logits, frame)
    assert torch.equal(before[paired_rows], after[paired_rows]), '配对行的F/B/E排序发生变化。'


def main(check_only=False):
    if not check_only and (cfg.MODEL_DIR.exists() or (cfg.RESULTS_DIR / 'run_config.json').exists()):
        raise FileExistsError('此RUN_NAME已有模型或训练记录，请换RUN_NAME保留旧实验。')

    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    train, dev, reference = read_data()
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('Train / Dev：', len(train), '/', len(dev))
    print('数据：原3290行不变，没有新增或反转样本。')
    print('参考权重：', cfg.REFERENCE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone_config = AutoConfig.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    n_targets = torch.tensor(train.label.eq('NEGATIVE_OTHER').to_numpy(), dtype=torch.float32)
    n_prior = n_targets.mean().item()
    model = build_model(backbone_config, tokenizer, cfg.HEAD_HIDDEN_SIZE, n_prior)
    model.relation_model.load_state_dict(torch.load(cfg.REFERENCE_MODEL, map_location='cpu', weights_only=True))
    model.to(device)
    print('可训练参数：', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('冻结参数：', sum(p.numel() for p in model.parameters() if not p.requires_grad))

    # 先验证旧模型，避免在加载了错误checkpoint时继续训练。
    dev_cache = extract_cache(model, dev, tokenizer, device, '提取Dev特征：只推理')
    baseline = check_reference(dev_cache, dev)
    check_fixed_relations(predict_cache(model, dev_cache, device), dev_cache, dev)
    if check_only:
        print('参考模型核查通过；未训练N头，未保存新模型。')
        return

    train_cache = extract_cache(model, train, tokenizer, device, '提取Train特征：只推理一次')
    cache_dir = cfg.RESULTS_DIR / 'feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, cache_dir / 'train.pt')
    torch.save(dev_cache, cache_dir / 'dev.pt')
    train.to_csv(cfg.RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, cfg.RESULTS_DIR / 'train_reference.csv')

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    backbone_config.save_pretrained(cfg.MODEL_DIR / 'backbone_config')
    config = {
        'run_name': cfg.RUN_NAME, 'seed': cfg.SEED, 'model_name': cfg.MODEL_NAME,
        'reference_model': str(cfg.REFERENCE_MODEL), 'reference_sha256': file_hash(cfg.REFERENCE_MODEL),
        'reference_run': str(cfg.REFERENCE_DIR), 'reference_epoch': 2,
        'head_hidden_size': cfg.HEAD_HIDDEN_SIZE, 'n_prior': n_prior,
        'epochs': cfg.EPOCHS, 'learning_rate': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'n_pos_weight': cfg.N_POS_WEIGHT, 'train_batch_size': cfg.TRAIN_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'max_length': cfg.MAX_LENGTH,
        'target_independent': cfg.TARGET_INDEPENDENT, 'target_pair': cfg.TARGET_PAIR,
        'train_rows': len(train), 'train_N': int(n_targets.sum().item()),
        'train_sha256': reference['augmentation']['effective_train_sha256'],
        'dev_sha256': file_hash(cfg.DEV_PATH),
        'selection': 'max min(independent/target_independent, pair_aware/target_pair); earliest tie',
        'trainable': 'n_head only; frozen encoder and F/B/E outputs remain in eval mode',
        'loss': 'BCEWithLogitsLoss on Train N vs non-N; no Pair CE in this repair stage',
        'feature_inputs': 'single-row CLS and existing alignment pooled features; no pair_id or group size',
        'probabilities': 'P(N)=sigmoid(g); P(F/B/E)=(1-P(N))*softmax(frozen F/B/E)',
        'reference_selection': 'reference checkpoint was previously selected on Dev pair-aware F1',
        'swa_enabled': False, 'mixed_pair_training': False,
        'source_sha256': {name: file_hash(cfg.HERE / name) for name in
                          ['config.py', 'model.py', 'train_deberta_n_head.py']},
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    loader = DataLoader(
        TensorDataset(train_cache['features'], n_targets),
        batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.N_POS_WEIGHT, device=device))
    optimizer = torch.optim.AdamW(model.n_head.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    history, selections = [], []
    best_score, best = -1.0, None
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = train_head(model, loader, optimizer, criterion, device)
        print(f'\n===== N头第{epoch}/{cfg.EPOCHS}轮，Train BCE={loss:.6f} =====')
        logits = predict_cache(model, dev_cache, device)
        check_fixed_relations(logits, dev_cache, dev)
        summary = score_predictions(logits, dev, cfg.RESULTS_DIR / f'epoch_{epoch}', cfg.GOLD_PATH)
        selected = selection_record(summary, epoch, cfg.TARGET_INDEPENDENT, cfg.TARGET_PAIR)
        selections.append({**selected, 'train_bce': loss})
        for item in summary.to_dict('records'):
            history.append({'epoch': epoch, 'train_bce': loss, **item})
        pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(selections).to_csv(cfg.RESULTS_DIR / 'joint_selection.csv', index=False, encoding='utf-8-sig')
        print(f"同轮：独立={selected['independent_f1']:.6f}，pair={selected['pair_aware_f1']:.6f}；"
              f"同时达标={selected['targets_met']}")
        if selected['joint_score'] > best_score:
            best_score, best = selected['joint_score'], selected
            # 每次只覆盖这个很小的最佳头；完整184M模型在最后保存一次。
            save_weights(model.n_head, cfg.MODEL_DIR / 'best_n_head.pth')
            (cfg.MODEL_DIR / 'best_metrics.json').write_text(json.dumps(best, indent=2), encoding='utf-8')

    model.n_head.load_state_dict(torch.load(cfg.MODEL_DIR / 'best_n_head.pth', map_location=device, weights_only=True))
    model.eval()
    save_weights(model, cfg.MODEL_DIR / 'best_model.pth')
    final_summaries = []
    for split, cache, frame, gold_path in [
        ('dev', dev_cache, dev, cfg.GOLD_PATH),
        ('train', train_cache, train, cfg.RESULTS_DIR / 'train_reference.csv'),
    ]:
        print(f'\n===== 同一个最佳模型：{split} =====')
        logits = predict_cache(model, cache, device)
        check_fixed_relations(logits, cache, frame)
        summary = score_predictions(logits, frame, cfg.RESULTS_DIR / 'best' / split, gold_path)
        for item in summary.to_dict('records'):
            final_summaries.append({'epoch': best['epoch'], 'split': split, **item})
    pd.DataFrame(final_summaries).to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
    baseline.to_csv(cfg.RESULTS_DIR / 'reference_summary.csv', index=False, encoding='utf-8-sig')
    print('\n最佳同轮结果：', best)
    print('同时达到0.87和0.94。' if best['targets_met'] else '本次尚未同时达到两个目标。')
    print('完整最佳模型：', cfg.MODEL_DIR / 'best_model.pth')
    print('逐轮同轮成绩：', cfg.RESULTS_DIR / 'joint_selection.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='只复现0.9381，不训练新头')
    args = parser.parse_args()
    main(check_only=args.check_only)
