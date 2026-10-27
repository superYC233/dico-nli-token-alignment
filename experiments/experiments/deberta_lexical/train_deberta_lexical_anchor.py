"""右键运行：官方3042 + 原Lexical；Row照旧，Pair后向本轮Row参数锚点回拉。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.data import (
    validate_groups, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, predict_logits
from experiments.deberta_lexical.train_deberta_lexical import set_seed, train_rows
from experiments.deberta_lexical.parameter_anchor import RowParameterAnchor
from experiments.deberta_stage_average.train_deberta_stage_average import score_predictions, stage_changes
from experiments.deberta_bidirectional.data import write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import save_weights, load_weights


RUN_NAME = 'run06_row_parameter_anchor'
ANCHOR_STRENGTH = 1000.0  # 首轮固定值；0为同代码无回拉对照，不自动扫参数。
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
LEXICAL_HIDDEN_SIZE = 32
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05
HERE = Path(__file__).resolve().parent
REFERENCE_DIR = HERE / 'outputs/run02_branch_lr/lexical_seed_42'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_official_data():
    reference = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    assert file_hash(TRAIN_PATH) == reference['train_sha256']
    assert file_hash(DEV_PATH) == reference['dev_sha256']
    assert file_hash(HERE / 'lexical_features.py') == reference['feature_source_sha256']
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    assert len(train) == 3042 and len(dev) == 660
    return train, dev


def build_model(tokenizer, model_name=MODEL_NAME, revision=BASE_REVISION,
                lexical_hidden=LEXICAL_HIDDEN_SIZE):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    return DebertaDiCoModel(backbone, tokenizer, feature_size=len(FEATURE_NAMES),
                            lexical_hidden=lexical_hidden, use_lexical_features=True)


def build_optimizer(model):
    original, lexical = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (lexical if name.startswith('lexical_output.') else original).append(parameter)
    return torch.optim.AdamW([
        {'params': original, 'lr': LEARNING_RATE},
        {'params': lexical, 'lr': LEXICAL_LEARNING_RATE},
    ], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_anchored_pairs(model, loader, optimizer, criterion, device, anchor,
                         strength=ANCHOR_STRENGTH):
    model.train()
    started = time.perf_counter()
    total, pair_count, records = 0.0, 0, []
    for step, (ids0, mask0, features0, ids1, mask1, features1, targets) in enumerate(
        tqdm(loader, desc='原Pair CE + Row参数锚定'), 1,
    ):
        optimizer.zero_grad(set_to_none=True)
        first = model(ids0.to(device), mask0.to(device), features0.to(device))
        second = model(ids1.to(device), mask1.to(device), features1.to(device))
        pair_loss = criterion(get_pair_logits(first, second), targets.to(device))
        (PAIR_WEIGHT * pair_loss).backward()
        optimizer.step()
        # 回拉不写入grad，不混入Adam的m/v；仍只有一次编码器反向。
        stats = anchor.apply(optimizer, strength)
        value = pair_loss.item()
        total += value
        pair_count += len(targets)
        stats.update(step=step, pair_groups=len(targets), pair_loss=value,
                     weighted_pair_loss=PAIR_WEIGHT * value)
        records.append(stats)
        del first, second, pair_loss
    return {'pair_loss': total / len(loader), 'pair_ce_groups': pair_count,
            'pair_optimizer_steps': len(loader), 'pair_train_seconds': time.perf_counter() - started,
            'anchor_strength': strength,
            'final_original_distance': records[-1]['original_distance_after_pull'],
            'final_lexical_distance': records[-1]['lexical_distance_after_pull']}, records


def evaluate_phase(model, loader, frame, device, output_dir, gold_path):
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    # 与原Lexical相同的四类预测，额外评估不改变下一阶段shuffle/dropout。
    with torch.random.fork_rng(devices=devices):
        logits = predict_logits(model, loader, device)
        return score_predictions(logits, frame, output_dir, gold_path)


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    assert file_hash(HERE / 'lexical_features.py') == config['feature_source_sha256']
    assert config['feature_names'] == FEATURE_NAMES
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, config['model_name'], config['base_revision'], config['lexical_hidden_size'])
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config


def main(run_name=RUN_NAME, strength=ANCHOR_STRENGTH, prepare_only=False):
    if strength < 0:
        raise ValueError('ANCHOR_STRENGTH必须大于等于0。')
    train, dev = read_official_data()
    train_features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    rows = DiCoDataset(train, tokenizer, train_features, MAX_LENGTH)
    pairs = PairDataset(train, tokenizer, train_features, MAX_LENGTH)
    dev_rows = DiCoDataset(dev, tokenizer, dev_features, MAX_LENGTH)
    assert len(pairs) == 1282
    model_dir, output_dir = MODEL_ROOT / run_name, HERE / 'outputs' / run_name
    print('数据：只用官方Train3042行，Dev660行，真实Pair1282组。')
    print('模型：原Lexical20维分支，804参数；从相同NLI初始化重训。')
    print('每轮：完整Row CE → 保存本轮参数锚点/评估 → 原Pair CE及参数回拉 → 评估。')
    print('原模型 / 词汇分支学习率：', LEARNING_RATE, '/', LEXICAL_LEARNING_RATE)
    print('锚定强度：', strength, '；原模型 / 词汇分支每步回拉比例：',
          LEARNING_RATE * strength / (1 + LEARNING_RATE * strength), '/',
          LEXICAL_LEARNING_RATE * strength / (1 + LEXICAL_LEARNING_RATE * strength))
    print('Row191步 + Pair81步；固定padding128，原Pair系数0.5、平滑0.05。')
    print('没有人工增强、SAM、BLOOP、NN或额外N回放。')
    print('选模：Pair后Dev pair-aware F1最高，只保存一个最佳完整模型。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据/词汇特征/路径检查完成，没有加载模型权重或训练。')
        return
    if (model_dir / 'run_config.json').exists() or (output_dir / 'run_config.json').exists():
        raise FileExistsError('此RUN_NAME已有训练记录，请修改RUN_NAME，保留旧结果。')
    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(tokenizer).to(device)
    row_loader = DataLoader(rows, batch_size=ROW_BATCH_SIZE, shuffle=True)
    pair_loader = DataLoader(pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_rows, batch_size=DEV_BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(rows, batch_size=DEV_BATCH_SIZE, shuffle=False)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(PAIR_WEIGHTS, device=device),
                                         label_smoothing=PAIR_LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    save_feature_table(train, train_features, output_dir / 'train_features.csv')
    save_feature_table(dev, dev_features, output_dir / 'dev_features.csv')
    source_paths = [Path(__file__).resolve()] + [HERE / name for name in [
        'parameter_anchor.py', 'model.py', 'data.py', 'lexical_features.py', 'evaluation.py',
        'train_deberta_lexical.py',
    ]]
    config = {
        'run_name': run_name, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION, 'seed': SEED,
        'architecture': 'original CLS + Token Alignment + lexical_v1 additive four-class branch',
        'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs': len(pairs),
        'data_augmentation': False, 'official_train_only': True,
        'train_sha256': file_hash(TRAIN_PATH), 'dev_sha256': file_hash(DEV_PATH),
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'feature_source_sha256': file_hash(HERE / 'lexical_features.py'),
        'lexical_hidden_size': LEXICAL_HIDDEN_SIZE, 'use_lexical_features': True,
        'learning_rate': LEARNING_RATE, 'lexical_learning_rate': LEXICAL_LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY, 'epochs': EPOCHS, 'dtype': 'float32',
        'row_batch_size': ROW_BATCH_SIZE, 'pair_batch_size': PAIR_BATCH_SIZE,
        'dev_batch_size': DEV_BATCH_SIZE, 'max_length': MAX_LENGTH, 'dynamic_padding': False,
        'gradient_checkpointing': False, 'row_weights': ROW_WEIGHTS, 'pair_weights': PAIR_WEIGHTS,
        'pair_weight': PAIR_WEIGHT, 'pair_label_smoothing': PAIR_LABEL_SMOOTHING,
        'anchor_strength': strength, 'anchor_reference': 'detached parameters immediately after each Row stage',
        'anchor_scope': 'all trainable parameters, including original encoder/CLS/alignment and lexical branch',
        'anchor_penalty': 'strength/2 * sum((theta - theta_Row)^2); global sum, not parameter mean',
        'anchor_update': 'after AdamW candidate u: theta=theta_Row+(u-theta_Row)/(1+lr*strength)',
        'anchor_original_retention': 1 / (1 + LEARNING_RATE * strength),
        'anchor_lexical_retention': 1 / (1 + LEXICAL_LEARNING_RATE * strength),
        'anchor_momentum': 'anchor does not enter Adam moments; original shared AdamW states are retained',
        'anchor_source': 'RecAdam-inspired parameter retention; task-specific proximal adaptation, NOT original RecAdam',
        'anchor_paper': 'https://aclanthology.org/2020.emnlp-main.634/',
        'anchor_code_reference': 'https://github.com/Sanyuan-Chen/RecAdam/blob/master/RecAdam.py',
        'row_steps_per_epoch': len(row_loader), 'pair_steps_per_epoch': len(pair_loader),
        'optimizer_steps_per_epoch': len(row_loader) + len(pair_loader),
        'forward_rows_per_epoch': len(train) + 2 * len(pairs),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_order': 'full Row CE; snapshot Row parameters; evaluate; Pair CE with proximal parameter anchoring; evaluate',
        'selection': 'best AFTER-PAIR Dev pair-aware F1; earliest on ties',
        'inference': 'unchanged four-class argmax and FB/BF/EE; singleton predicted by model',
        'reference_run': str(REFERENCE_DIR),
        'comparison_note': 'Original Lexical data/features/lrs/losses/fixed padding retained. Anchor0 disables pull. Model weights are freshly initialized from the same NLI checkpoint, not the Dev-selected0.9306 model.',
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    history, stages, changes, anchor_records = [], [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        model.train()
        row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
        anchor = RowParameterAnchor(model)
        row_dir, pair_dir = output_dir / f'epoch_{epoch}/after_row', output_dir / f'epoch_{epoch}/after_pair'
        row_summary = evaluate_phase(model, dev_loader, dev, device, row_dir, GOLD_PATH)
        pair_losses, records = train_anchored_pairs(model, pair_loader, optimizer, pair_criterion, device, anchor, strength)
        del anchor
        pair_summary = evaluate_phase(model, dev_loader, dev, device, pair_dir, GOLD_PATH)
        losses = dict(row_loss=row_loss, **pair_losses)
        print('本轮损失和参数距离：', losses)
        for phase, summary in [('after_row', row_summary), ('after_pair', pair_summary)]:
            for record in summary.to_dict('records'):
                record.update(epoch=epoch, phase=phase, **losses)
                stages.append(record)
                if phase == 'after_pair':
                    history.append(record.copy())
        for record in records:
            record['epoch'] = epoch
        anchor_records.extend(records)
        pd.DataFrame(anchor_records).to_csv(output_dir / 'anchor_batches.csv', index=False, encoding='utf-8-sig')
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
    parser.add_argument('--anchor-strength', type=float, default=ANCHOR_STRENGTH)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.anchor_strength, args.prepare_only)
