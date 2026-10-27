"""右键运行：原生cross-encoder，四分类CE + 真实反向KL。"""
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_reversal_kl import config as cfg
from experiments.deberta_reversal_kl.data import read_data, GroupDataset, collate_groups, make_dev_loader
from experiments.deberta_reversal_kl.evaluation import predict_logits, score_logits


def load_model():
    # 保留原NLI编码器和pooler，只重新初始化尺寸不相同的三分类输出层。
    model_config = AutoConfig.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    model_config.id2label = dict(enumerate(cfg.FULL_LABELS))
    model_config.label2id = cfg.LABEL_TO_ID.copy()
    model_config.num_labels = 4
    model_config.problem_type = 'single_label_classification'
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION, cache_dir=get_model_cache_dir(),
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
        config=model_config, ignore_mismatched_sizes=True,
    )


def compute_loss(logits, labels, pair_indices, criterion, consistency_weight=0.5):
    ce_loss = criterion(logits, labels)
    if len(pair_indices) > 0:
        first = logits[pair_indices[:, 0]]
        # 顺序为F/B/E/N；真实F/B/E反向组中，N分量代表双方的错误概率。
        second = logits[pair_indices[:, 1]][:, [1, 0, 2, 3]]
        log_p = F.log_softmax(first, dim=-1)
        log_q = F.log_softmax(second, dim=-1)
        # 0.5 * [KL(p||q) + KL(q||p)]，两边都接收梯度，不detach。
        kl_loss = 0.5 * ((log_p.exp() - log_q.exp()) * (log_p - log_q)).sum(dim=-1).mean()
    else:
        kl_loss = logits.new_zeros(())
    return ce_loss + consistency_weight * kl_loss, ce_loss, kl_loss


def main():
    set_seed(cfg.SEED)
    train, dev = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    dataset = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    train_loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    dev_loader = make_dev_loader(dev, tokenizer, cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE)
    model = load_model()
    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = len(train_loader) * cfg.EPOCHS
    warmup_steps = math.ceil(total_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    eval_every = max(1, math.ceil(len(train_loader) * cfg.EVAL_EPOCH_FRACTION))

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=False)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(cfg.MODEL_DIR / 'tokenizer')
    source_paths = [Path(__file__), Path(cfg.__file__), Path(__file__).with_name('data.py'),
                    Path(__file__).with_name('evaluation.py'),
                    PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe/data.py']
    run_config = {
        'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
        'model_config': model.config.to_dict(), 'seed': cfg.SEED, 'dtype': 'float32',
        'architecture': 'native AutoModelForSequenceClassification; no Alignment or Lexical',
        'training': 'four-class CE + symmetric reversal KL on true reversible groups; no Pair CE',
        'learning_rate': cfg.LEARNING_RATE, 'weight_decay': cfg.WEIGHT_DECAY,
        'row_weights': [1.0, 1.0, 1.0, 1.0], 'label_smoothing': cfg.LABEL_SMOOTHING,
        'consistency_weight': cfg.CONSISTENCY_WEIGHT, 'max_grad_norm': cfg.MAX_GRAD_NORM,
        'epochs': cfg.EPOCHS, 'warmup_ratio': cfg.WARMUP_RATIO,
        'warmup_steps': warmup_steps, 'planned_optimizer_steps': total_steps,
        'scheduler': 'linear warmup then linear decay',
        'group_batch_size': cfg.GROUP_BATCH_SIZE, 'dev_batch_size': cfg.DEV_BATCH_SIZE,
        'max_length': cfg.MAX_LENGTH, 'train_rows': len(train), 'dev_rows': len(dev),
        'train_groups': len(dataset), 'train_pairs': dataset.pair_count,
        'steps_per_epoch': len(train_loader), 'eval_every_steps': eval_every,
        'early_stopping_patience_evaluations': cfg.EARLY_STOPPING_PATIENCE,
        'selection': cfg.SELECT_BY, 'labels': cfg.FULL_LABELS, 'data_augmentation': False,
        'adaptation_note': 'English base NLI adaptation of public clean CE/reversal-KL recipe; not a reproduction of multilingual ensembles or structural-prior scores',
        'source_repo': 'https://github.com/MinLee0210/dico-nli-challenge',
        'train_sha256': hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_sha256': hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
        (folder / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('模型：原生DeBERTa四分类，保留NLI pooler；没有Alignment、Lexical、Pair CE')
    print('Train / Dev：', len(train), '/', len(dev))
    print('真实反向组：', dataset.pair_count, '；每轮步数：', len(train_loader))
    print('损失：完整四类CE +', cfg.CONSISTENCY_WEIGHT, '* 真实反向KL')
    print('lr：', cfg.LEARNING_RATE, '；warmup：', warmup_steps, '步；然后线性衰减')
    print('每', eval_every, '步评估，按', cfg.SELECT_BY, '选模；耐心值：', cfg.EARLY_STOPPING_PATIENCE)
    print('模型目录：', cfg.MODEL_DIR)
    print('结果目录：', cfg.RESULTS_DIR)

    history = []
    best_f1 = -1.0
    best_step = 0
    global_step = 0
    evaluations_without_improvement = 0
    stop = False
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        ce_sum = 0.0
        kl_sum = 0.0
        interval_steps = 0
        print(f'\n===== 第 {epoch}/{cfg.EPOCHS} 轮 =====')
        for step, batch in enumerate(tqdm(train_loader, desc='四分类CE + 反向KL'), start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            ).logits
            loss, ce_loss, kl_loss = compute_loss(
                logits, batch['labels'].to(device), batch['pair_indices'].to(device),
                criterion, cfg.CONSISTENCY_WEIGHT,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1
            ce_sum += ce_loss.item()
            kl_sum += kl_loss.item()
            interval_steps += 1
            if global_step % eval_every != 0 and step != len(train_loader):
                continue
            epoch_fraction = epoch - 1 + step / len(train_loader)
            print(f'\nStep {global_step} / epoch {epoch_fraction:.2f}；'
                  f'CE={ce_sum / interval_steps:.6f}，KL={kl_sum / interval_steps:.6f}，'
                  f'lr={scheduler.get_last_lr()[0]:.3e}')
            summary = score_logits(
                predict_logits(model, dev_loader, device), dev,
                cfg.RESULTS_DIR / f'step_{global_step}', gold_path=cfg.GOLD_PATH,
            )
            for record in summary.to_dict('records'):
                record.update(global_step=global_step, epoch=epoch_fraction,
                              ce_loss=ce_sum / interval_steps, kl_loss=kl_sum / interval_steps,
                              learning_rate=scheduler.get_last_lr()[0])
                history.append(record)
            pd.DataFrame(history).to_csv(cfg.RESULTS_DIR / 'eval_summary.csv', index=False, encoding='utf-8-sig')
            ce_sum = kl_sum = 0.0
            interval_steps = 0
            score = float(summary.loc[summary['method'] == cfg.SELECT_BY, 'weighted_f1'].iloc[0])
            if score > best_f1:
                best_f1, best_step = score, global_step
                evaluations_without_improvement = 0
                torch.save(model.state_dict(), cfg.MODEL_DIR / 'best_model.pth')
                best = {
                    'best_step': best_step, 'best_epoch': epoch_fraction,
                    'selection': cfg.SELECT_BY, 'weighted_f1': best_f1,
                    'same_checkpoint_metrics': summary.to_dict('records'),
                }
                for folder in [cfg.MODEL_DIR, cfg.RESULTS_DIR]:
                    (folder / 'best_metrics.json').write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
                summary.to_csv(cfg.RESULTS_DIR / 'best_summary.csv', index=False, encoding='utf-8-sig')
                print(f'刷新最佳：step {best_step}，independent F1={best_f1:.6f}')
            else:
                evaluations_without_improvement += 1
            if evaluations_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
                print('独立F1连续', evaluations_without_improvement, '次评估未提升，提前停止。')
                stop = True
                break
        if stop:
            break
    print(f'\n最佳step {best_step}，independent F1={best_f1:.6f}')
    print('完整最佳权重：', cfg.MODEL_DIR / 'best_model.pth')
    print('同一权重的三种成绩：', cfg.RESULTS_DIR / 'best_summary.csv')


if __name__ == '__main__':
    main()
