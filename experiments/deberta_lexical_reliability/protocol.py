"""记录训练配方和源码指纹，防止复验中途悄悄改变实验。"""
import hashlib
import json
from pathlib import Path

from experiments.deberta_lexical_reliability import config as cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_definition():
    original = cfg.PROJECT_ROOT / 'experiments/deberta_lexical'
    source_files = [original / name for name in [
        'model.py', 'data.py', 'lexical_features.py', 'evaluation.py', 'train_deberta_lexical.py',
    ]]
    source_files += [cfg.EXPERIMENT_DIR / name for name in ['config.py', 'run_one.py', 'protocol.py']]
    return {
        'new_paired_seeds': cfg.SEEDS, 'arms': ['baseline', 'lexical'],
        'primary_metric': 'best-by-Dev-pair-aware weighted F1 within 10 epochs',
        'primary_comparison': 'mean of (lexical minus baseline) over BOTH new seeds 7 and 21',
        'secondary_metric': 'final epoch pair-aware weighted F1',
        'historical_seed_42': 'reused, already used for recipe selection; not an independent confirmation',
        'selection_note': 'Dev was repeatedly used; these results do not substitute for an untouched test set',
        'bootstrap_note': 'paired pair_id group resampling; conditional on fitted models; excludes training variance',
        'model_name': cfg.MODEL_NAME, 'epochs': cfg.EPOCHS,
        'row_batch_size': cfg.ROW_BATCH_SIZE, 'pair_batch_size': cfg.PAIR_BATCH_SIZE,
        'dev_batch_size': cfg.DEV_BATCH_SIZE, 'max_length': cfg.MAX_LENGTH,
        'learning_rate': cfg.LEARNING_RATE, 'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE,
        'weight_decay': cfg.WEIGHT_DECAY, 'row_weights': cfg.ROW_WEIGHTS,
        'pair_weights': cfg.PAIR_WEIGHTS, 'pair_weight': cfg.PAIR_WEIGHT,
        'pair_label_smoothing': cfg.PAIR_LABEL_SMOOTHING,
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE, 'dtype': 'float32',
        'model_root': str(cfg.MODEL_ROOT),
        'data_sha256': {str(p.relative_to(cfg.PROJECT_ROOT)): digest(p)
                        for p in [cfg.TRAIN_PATH, cfg.DEV_PATH, cfg.GOLD_PATH]},
        'source_sha256': {str(p.relative_to(cfg.PROJECT_ROOT)): digest(p) for p in source_files},
    }


def ensure_protocol():
    # 与原最佳运行记录核对关键训练参数，避免把已调坏的配方拿来复验。
    old = json.loads((cfg.PROJECT_ROOT / 'experiments/deberta_lexical/outputs'
                     / 'run02_branch_lr/lexical_seed_42/run_config.json').read_text(encoding='utf-8'))
    current = protocol_definition()
    for name in ['model_name', 'epochs', 'row_batch_size', 'pair_batch_size', 'dev_batch_size',
                 'max_length', 'learning_rate', 'lexical_learning_rate', 'weight_decay', 'row_weights',
                 'pair_weights', 'pair_weight', 'pair_label_smoothing', 'lexical_hidden_size', 'dtype']:
        if current[name] != old[name]:
            raise ValueError(f'{name} 不同于 0.930569 配方，停止复验。')
    if digest(cfg.TRAIN_PATH) != old['train_sha256'] or digest(cfg.DEV_PATH) != old['dev_sha256']:
        raise ValueError('数据文件与原最佳运行不同。')
    text = json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True)
    folder = cfg.RESULTS_ROOT / cfg.RUN_NAME
    path = folder / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != current:
            raise ValueError('配方、源码或数据在复验中改变了。请恢复版本，或用新 RUN_NAME 建立新实验。')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
