"""右键运行：顶部2层+N头，四类Row CE与F/B/E保持约束同一步训练。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_manual_aug.train_boundary_manual import set_seed, save_weights
from experiments.deberta_stage_average.train_deberta_stage_average import (
    read_data, write_train_reference, evaluate_phase,
)
from experiments.deberta_frozen_n_head.train_deberta_n_head import file_hash, selection_record
from experiments.deberta_frozen_n_head import config_top2 as cfg
from experiments.deberta_frozen_n_head.model_top2 import build_model


def compute_losses(logits, labels, teacher_logits, keep_weight, temperature):
    # 每一行都受四分类监督，包括N，不只约束两个答案加起来是否正确。
    row_loss = F.cross_entropy(logits, labels)
    teacher = teacher_logits.detach()[:, :3]
    # 只保护教师在Train上条件分类答对的F/B/E行，不保留教师的N偏置，
    # 也不要求学生模仿教师在Train上已经判错的关系。
    keep_mask = labels.ne(3) & teacher.argmax(dim=1).eq(labels)
    if keep_mask.any():
        teacher_log_p = F.log_softmax(teacher[keep_mask] / temperature, dim=1)
        student_log_p = F.log_softmax(logits[keep_mask, :3] / temperature, dim=1)
        keep_loss = (teacher_log_p.exp() * (teacher_log_p - student_log_p)).sum(dim=1).mean()
        keep_loss = keep_loss * temperature ** 2
    else:
        keep_loss = logits[:, :3].sum() * 0.0
    loss = row_loss + keep_weight * keep_loss
    return loss, row_loss, keep_loss, int(keep_mask.sum().item())


def make_optimizer(model, encoder_lr, head_lr, weight_decay):
    encoder_parameters = [parameter for parameter in model.relation_model.deberta_model.parameters()
                          if parameter.requires_grad]
    return torch.optim.AdamW([
        {'params': encoder_parameters, 'lr': encoder_lr},
        {'params': model.n_head.parameters(), 'lr': head_lr},
    ], weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, device, keep_weight, temperature, max_grad_norm):
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    totals = {'loss': 0.0, 'row_loss': 0.0, 'keep_loss': 0.0, 'keep_rows': 0}
    count = 0
    for ids, mask, labels, teacher in tqdm(loader, desc='Row CE + F/B/E保持'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids.to(device), mask.to(device), None)
        loss, row_loss, keep_loss, kept = compute_losses(
            logits, labels.to(device), teacher.to(device), keep_weight, temperature,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        size = len(labels)
        totals['loss'] += loss.item() * size
        totals['row_loss'] += row_loss.item() * size
        totals['keep_loss'] += keep_loss.item() * size
        totals['keep_rows'] += kept
        count += size
    # keep_loss记录各批保持损失按批行数加权的均值，与本轮优化目标对应。
    for name in ['loss', 'row_loss', 'keep_loss']:
        totals[name] /= count
    return totals


def read_teacher(train, source_config):
    source_train = pd.read_csv(cfg.SOURCE_RESULTS_DIR / 'training_data.csv', keep_default_na=False)
    train_hash = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    source_hash = hashlib.sha256(source_train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert train_hash == source_hash == source_config['train_sha256']
    assert source_train.instance_id.tolist() == train.instance_id.tolist()
    cache = torch.load(cfg.TEACHER_CACHE, map_location='cpu', weights_only=True)
    teacher = cache['base_logits'].float().clone()
    assert teacher.shape == (len(train), 4)
    assert torch.isfinite(teacher).all()
    # base_logits是原0.9381的输出，不是上一轮N头修正后的输出。
    return teacher


def compare_source(prediction_path):
    source = pd.read_csv(cfg.SOURCE_RESULTS_DIR / 'best/dev/predictions_with_scores.csv', keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    assert source.instance_id.tolist() == current.instance_id.tolist()
    changes = {column: int(source[column].ne(current[column]).sum())
               for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in changes.values()), '微调起点未复现run01最佳模型。'
    print('与run01最佳模型逐条比较：', changes)
    return changes


def write_metrics(metrics):
    (cfg.MODEL_DIR / 'best_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def main(check_only=False):
    if not check_only and (cfg.MODEL_DIR.exists() or (cfg.RESULTS_DIR / 'run_config.json').exists()):
        raise FileExistsError('本RUN_NAME已有训练记录，请换RUN_NAME，保留旧模型。')
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    train, dev, reference = read_data()
    source_config = json.loads((cfg.SOURCE_MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    source_best = json.loads((cfg.SOURCE_MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    assert cfg.MAX_LENGTH == source_config['max_length']
    assert cfg.DEV_BATCH_SIZE == source_config['dev_batch_size']
    teacher_logits = read_teacher(train, source_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.SOURCE_MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone_config = AutoConfig.from_pretrained(cfg.SOURCE_MODEL_DIR / 'backbone_config', local_files_only=True)
    model = build_model(
        backbone_config, tokenizer, source_config['head_hidden_size'], source_config['n_prior'],
        cfg.UNFREEZE_LAST_LAYERS,
    )
    source_path = cfg.SOURCE_MODEL_DIR / 'best_model.pth'
    model.load_state_dict(torch.load(source_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    parameter_rows = [{'name': name, 'parameters': parameter.numel(), 'trainable': parameter.requires_grad}
                      for name, parameter in model.named_parameters()]
    print('微调起点：', source_path, '；来源最佳轮：', source_best['epoch'])
    print('Train / Dev：', len(train), '/', len(dev))
    print('解冻顶部层数：', cfg.UNFREEZE_LAST_LAYERS)
    print('可训练参数：', sum(item['parameters'] for item in parameter_rows if item['trainable']))
    print('学习率：编码层', cfg.ENCODER_LEARNING_RATE, '；N头', cfg.HEAD_LEARNING_RATE)
    print('损失：四类Row CE +', cfg.KEEP_WEIGHT, '× Train F/B/E条件KL；没有Pair-only阶段。')

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dev_loader = DataLoader(
        RowDataset(dev, tokenizer, cfg.MAX_LENGTH), batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.SEED + 1),
    )
    initial = evaluate_phase(model, dev_loader, dev, device, cfg.RESULTS_DIR / 'epoch_0/dev', cfg.GOLD_PATH)
    source_check = compare_source(cfg.RESULTS_DIR / 'epoch_0/dev/predictions_with_scores.csv')
    best = selection_record(initial, 0, cfg.TARGET_INDEPENDENT, cfg.TARGET_PAIR)
    train_rows = RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    keep_rows = int((train_rows.labels.ne(3) & teacher_logits[:, :3].argmax(1).eq(train_rows.labels)).sum().item())
    print('保持监督覆盖Train行数：', keep_rows, '；N行不参与KL。')
    if check_only:
        print('起点和教师缓存核查通过；没有更新真实模型，没有保存新checkpoint。')
        return

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    backbone_config.save_pretrained(cfg.MODEL_DIR / 'backbone_config')
    pd.DataFrame(parameter_rows).to_csv(cfg.RESULTS_DIR / 'parameter_groups.csv', index=False, encoding='utf-8-sig')
    train.to_csv(cfg.RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, cfg.RESULTS_DIR / 'train_reference.csv')
    config = {
        'run_name': cfg.RUN_NAME, 'seed': cfg.SEED, 'model_name': source_config['model_name'],
        'source_model': str(source_path), 'source_sha256': file_hash(source_path),
        'source_results': str(cfg.SOURCE_RESULTS_DIR), 'source_epoch': source_best['epoch'],
        'source_check': source_check, 'head_hidden_size': source_config['head_hidden_size'],
        'n_prior': source_config['n_prior'], 'unfreeze_last_layers': cfg.UNFREEZE_LAST_LAYERS,
        'encoder_learning_rate': cfg.ENCODER_LEARNING_RATE, 'head_learning_rate': cfg.HEAD_LEARNING_RATE,
        'weight_decay': cfg.WEIGHT_DECAY, 'max_grad_norm': cfg.MAX_GRAD_NORM, 'epochs': cfg.EPOCHS,
        'train_batch_size': cfg.TRAIN_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'max_length': cfg.MAX_LENGTH, 'dtype': 'float32',
        'keep_weight': cfg.KEEP_WEIGHT, 'keep_temperature': cfg.KEEP_TEMPERATURE,
        'teacher_cache': str(cfg.TEACHER_CACHE), 'teacher_cache_sha256': file_hash(cfg.TEACHER_CACHE),
        'teacher_reference_sha256': source_config['reference_sha256'], 'keep_train_rows': keep_rows,
        'teacher': 'original 0.9381 Train logits; only F/B/E columns; no Dev training targets',
        'keep_mask': 'non-N Train rows where teacher conditional argmax equals Train gold',
        'loss': 'unweighted four-class Row CE + keep_weight*T^2*KL(teacher_FBE || student_FBE)',
        'train_rows': len(train), 'train_sha256': reference['augmentation']['effective_train_sha256'],
        'dev_sha256': file_hash(cfg.DEV_PATH),
        'target_independent': cfg.TARGET_INDEPENDENT, 'target_pair': cfg.TARGET_PAIR,
        'selection': 'max min(independent/target_independent, pair_aware/target_pair); include epoch0; earliest tie',
        'frozen': 'embeddings, bottom encoder layers, original CLS classifier and Token Alignment parameters',
        'dropout': 'trainable top encoder layers only; frozen modules remain eval',
        'pair_ce_enabled': False, 'nn_augmentation_enabled': False, 'swa_enabled': False,
        'reference_note': 'source and its frozen reference were previously selected using Dev',
        'source_code_sha256': {name: file_hash(cfg.HERE / name) for name in
                               ['config_top2.py', 'model_top2.py', 'train_deberta_top2.py']},
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    dataset = TensorDataset(train_rows.encoded['input_ids'], train_rows.encoded['attention_mask'],
                            train_rows.labels, teacher_logits)
    train_loader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(cfg.SEED))
    optimizer = make_optimizer(model, cfg.ENCODER_LEARNING_RATE, cfg.HEAD_LEARNING_RATE, cfg.WEIGHT_DECAY)
    # 起点也参与比较，若所有微调轮次更差，仍保留起点而不冒充有提升。
    save_weights(model, cfg.MODEL_DIR / 'best_model.pth')
    write_metrics(best)
    history = [{'epoch': 0, **item} for item in initial.to_dict('records')]
    selections = [best.copy()]
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f'\n===== 顶部微调第{epoch}/{cfg.EPOCHS}轮 =====')
        losses = train_epoch(model, train_loader, optimizer, device, cfg.KEEP_WEIGHT,
                             cfg.KEEP_TEMPERATURE, cfg.MAX_GRAD_NORM)
        print('训练损失：', losses)
        summary = evaluate_phase(model, dev_loader, dev, device, cfg.RESULTS_DIR / f'epoch_{epoch}/dev', cfg.GOLD_PATH)
        selected = selection_record(summary, epoch, cfg.TARGET_INDEPENDENT, cfg.TARGET_PAIR)
        selections.append({**selected, **losses})
        for item in summary.to_dict('records'):
            history.append({'epoch': epoch, **losses, **item})
        pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(selections).to_csv(cfg.RESULTS_DIR / 'joint_selection.csv', index=False, encoding='utf-8-sig')
        print('同轮评估：', selected)
        if selected['joint_score'] > best['joint_score']:
            best = selected.copy()
            save_weights(model, cfg.MODEL_DIR / 'best_model.pth')
            write_metrics(best)
            print('刷新最佳：只覆盖同一个best_model.pth。')

    model.load_state_dict(torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True))
    model.eval()
    train_eval_loader = DataLoader(train_rows, batch_size=cfg.DEV_BATCH_SIZE, shuffle=False,
                                   generator=torch.Generator().manual_seed(cfg.SEED + 2))
    results = []
    for split, loader, frame, gold_path in [
        ('dev', dev_loader, dev, cfg.GOLD_PATH),
        ('train', train_eval_loader, train, cfg.RESULTS_DIR / 'train_reference.csv'),
    ]:
        print(f'\n===== 同一个最佳模型：{split} =====')
        summary = evaluate_phase(model, loader, frame, device, cfg.RESULTS_DIR / 'best' / split, gold_path)
        if split == 'dev':
            reproduced = selection_record(summary, best['epoch'], cfg.TARGET_INDEPENDENT, cfg.TARGET_PAIR)
            assert abs(reproduced['independent_f1'] - best['independent_f1']) < 1e-10
            assert abs(reproduced['pair_aware_f1'] - best['pair_aware_f1']) < 1e-10
        results.extend({'epoch': best['epoch'], 'split': split, **item} for item in summary.to_dict('records'))
    pd.DataFrame(results).to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
    print('\n最佳微调轮次：', best)
    if best['epoch'] == 0:
        print('微调没有超过起点，保存的最佳模型仍为起点。')
    print('同时达到两个目标。' if best['targets_met'] else '本次尚未同时达到0.87和0.94。')
    print('完整最佳模型：', cfg.MODEL_DIR / 'best_model.pth')
    print('同轮结果：', cfg.RESULTS_DIR / 'joint_selection.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='只检查起点、冻结范围和Train教师，不训练')
    args = parser.parse_args()
    main(check_only=args.check_only)
