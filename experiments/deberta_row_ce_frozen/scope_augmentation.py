"""为冻结 Row CE 实验准备数据，并评估该冻结实验的保留诊断。"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_row_ce_frozen import config as cfg
from experiments.deberta_original_aug import scope_augmentation as augmentation
# 数据构造函数共用一份实现；这里仅保留冻结实验的保存路径与评估流程。
from experiments.deberta_original_aug.scope_augmentation import (
    FAMILIES, DIAGNOSTIC_PATH, build_train_variant,
)


def prepare_training_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    effective, extra, audit = build_train_variant(train, dev, cfg.AUGMENT_MODE, cfg.SEED)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if (cfg.RESULTS_DIR / 'run_config.json').exists() or (cfg.RESULTS_DIR / 'epoch_summary.csv').exists():
        raise FileExistsError('此 RUN_NAME 已有训练记录，请换新的 RUN_NAME。')
    effective.to_csv(cfg.RESULTS_DIR / 'training_data.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(cfg.RESULTS_DIR / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(cfg.RESULTS_DIR / 'augmentation_audit.csv', index=False, encoding='utf-8-sig')
    info = {'mode': cfg.AUGMENT_MODE, 'seed': cfg.SEED,
            'original_rows': len(train), 'extra_rows': len(extra), 'effective_rows': len(effective),
            'extra_class_counts': extra.label.value_counts().to_dict(),
            'effective_class_counts': effective.label.value_counts().to_dict(),
            'extra_groups': extra.pair_id.nunique(),
            'effective_train_sha256': hashlib.sha256(effective.to_csv(index=False).encode('utf-8')).hexdigest(),
            'augmentation_code_sha256': hashlib.sha256(Path(augmentation.__file__).read_bytes()).hexdigest(),
            'diagnostic_sha256': hashlib.sha256(DIAGNOSTIC_PATH.read_bytes()).hexdigest(),
            'annotation_note': 'Scope labels use explicit literal-semantic templates; no independent human adjudication.',
            'budget_note': 'scope/copy match rows, class counts, group shapes and optimizer steps; original none has fewer steps.'}
    (cfg.RESULTS_DIR / 'data_config.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('新增数据方案：', cfg.AUGMENT_MODE)
    print('原 Train / 新增 / 构造后：', len(train), '/', len(extra), '/', len(effective))
    print('新增类别数量：', info['extra_class_counts'])
    print('可查看新增样本：', cfg.RESULTS_DIR / 'extra_train_rows.csv')
    return effective, dev, info


def evaluate_preserved_diagnostics(model, tokenizer, device, output_dir):
    # 仅在冻结模型按 Dev independent 选好后测一次，不用于选择 epoch。
    from experiments.deberta_minimal_contrasts.evaluate_diagnostics import read_review, save_summaries
    from experiments.deberta_generalization_audit.export_predictions import predict

    review = read_review(DIAGNOSTIC_PATH.parent)
    selected = review[review.eligible & (review.kind != 'anchor') &
                      (review.review_status != 'rejected')].copy().reset_index(drop=True)
    inputs = selected.copy()
    inputs['label'] = inputs.proposed_label.map(dict(zip(cfg.LABELS, cfg.FULL_LABELS)))
    model.eval()
    predicted = predict(inputs, tokenizer, model, device)
    columns = ['prediction', 'max_probability']
    columns += [f'{prefix}_{label}' for prefix in ['logit', 'prob'] for label in cfg.LABELS]
    result = pd.concat([selected, predicted[columns]], axis=1)
    save_summaries(result, output_dir=output_dir)
    print('保留诊断结果：', output_dir)
    print('未审核的建议标签匹配率与已批准成绩分开保存；诊断题没有进入训练或选模。')


if __name__ == '__main__':
    prepare_training_data()
