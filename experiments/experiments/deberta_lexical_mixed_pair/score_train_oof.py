"""三折训练并给Train评分。只读Train，不读取官方Dev，不保存三折模型权重。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config_oof as cfg
from experiments.deberta_lexical_mixed_pair.oof_data import (
    load_train_and_folds, build_weight_table, oof_signature,
)
from experiments.deberta_lexical_mixed_pair.train_deberta_oof_hard import (
    build_model, build_optimizer, make_criteria, make_train_loader, make_dev_loader, train_epoch,
)
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits


def make_score_table(logits, held, fold):
    gold = torch.tensor(held['label'].map(cfg.LABEL_TO_ID).tolist())
    probabilities = torch.softmax(logits, dim=1)
    rows = torch.arange(len(held))
    other_logits = logits.clone()
    other_logits[rows, gold] = -torch.inf
    result = held.copy().reset_index(drop=True)
    result['fold'] = fold
    result['prediction'] = np.array(cfg.LABELS)[logits.argmax(dim=1).numpy()]
    result['gold_probability'] = probabilities[rows, gold].numpy()
    result['gold_margin'] = (logits[rows, gold] - other_logits.max(dim=1).values).numpy()
    for index, label in enumerate(cfg.LABELS):
        result[f'logit_{label}'] = logits[:, index].numpy()
        result[f'prob_{label}'] = probabilities[:, index].numpy()
    return result


def main():
    train, assignments = load_train_and_folds()
    signature = oof_signature()
    cfg.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config_path = cfg.OOF_DIR / 'oof_config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        assert saved == signature, 'OOF配置或Train已改变，请更换OOF_RUN_NAME。'
    else:
        config_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
    assignments.to_csv(cfg.OOF_DIR / 'fold_assignments.csv', index=False, encoding='utf-8-sig')
    assignments[assignments['label_conflict']].to_csv(
        cfg.OOF_DIR / 'label_conflicts_for_review.csv', index=False, encoding='utf-8-sig',
    )
    print('OOF只使用Train：', len(train), '行；', cfg.OOF_FOLDS, '折；每折固定', cfg.OOF_EPOCHS, '轮')
    print('可逆对和规范化重复文本不会跨折；每条行仅由未训练过它的模型评分。')
    print('OOF结果目录：', cfg.OOF_DIR)

    tokenizer = None
    features = None
    fold_results = []
    history_path = cfg.OOF_DIR / 'fold_training_history.csv'
    history = pd.read_csv(history_path).to_dict('records') if history_path.exists() else []
    for fold in range(1, cfg.OOF_FOLDS + 1):
        held_indices = np.flatnonzero(assignments['fold'].to_numpy() == fold)
        fit_indices = np.flatnonzero(assignments['fold'].to_numpy() != fold)
        fit = train.iloc[fit_indices].reset_index(drop=True)
        held = assignments.iloc[held_indices].reset_index(drop=True)
        assert set(assignments.iloc[fit_indices]['split_group']).isdisjoint(held['split_group'])
        assert set(fit['pair_id']).isdisjoint(held['pair_id'])
        prediction_path = cfg.OOF_DIR / f'fold_{fold}_scores.csv'
        if prediction_path.exists():
            result = pd.read_csv(prediction_path, keep_default_na=False)
            assert result['instance_id'].is_unique
            assert set(result['instance_id']) == set(held['instance_id'])
            print(f'第{fold}折已有评分，跳过训练。')
        else:
            print(f'\n===== 第 {fold}/{cfg.OOF_FOLDS} 折：训练 {len(fit)} 行，留出 {len(held)} 行 =====')
            # 每折重新初始化同一个NLI主干和任务头，不加载任何DiCo已训练权重。
            set_seed(cfg.SEED)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
                )
                # 特征仅依赖各行文本，没有用整个语料拟合词表或标准化器。
                features = build_feature_tensor(train)
            device = torch.device('cuda')
            fit_loader = make_train_loader(fit, tokenizer, features[fit_indices], np.ones(len(fit)))
            held_loader = make_dev_loader(
                held, tokenizer, features[held_indices], cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE,
            )
            model = build_model(tokenizer, device)
            optimizer = build_optimizer(model)
            row_criterion, pair_criterion = make_criteria(device)
            history = [record for record in history if record['fold'] != fold]
            for epoch in range(1, cfg.OOF_EPOCHS + 1):
                print(f'第{fold}折，第{epoch}/{cfg.OOF_EPOCHS}轮')
                losses = train_epoch(model, fit_loader, optimizer, row_criterion, pair_criterion, device)
                history.append({'fold': fold, 'epoch': epoch, **losses})
                pd.DataFrame(history).to_csv(history_path, index=False, encoding='utf-8-sig')
                print(f"Row CE：{losses['row_loss']:.6f}；Pair CE：{losses['pair_loss']:.6f}")
            # 只评固定最后一轮，不用留出标签或官方Dev挑checkpoint。
            logits = predict_logits(model, held_loader, device)
            result = make_score_table(logits, held, fold)
            result.to_csv(prediction_path, index=False, encoding='utf-8-sig')
            del model, optimizer, row_criterion, pair_criterion
            torch.cuda.empty_cache()
        gold_short = result['label'].map(dict(zip(cfg.FULL_LABELS, cfg.LABELS)))
        print(f"第{fold}折留出独立W-F1：{f1_score(gold_short, result['prediction'], average='weighted'):.6f}")
        fold_results.append(result)

    oof = pd.concat(fold_results, ignore_index=True)
    assert oof['instance_id'].is_unique and set(oof['instance_id']) == set(train['instance_id'])
    oof = oof.set_index('instance_id').loc[train['instance_id']].reset_index()
    oof.to_csv(cfg.OOF_DIR / 'train_oof_scores.csv', index=False, encoding='utf-8-sig')
    weights, groups = build_weight_table(train, oof, 'hard')
    weights[weights['selected_hard']].to_csv(
        cfg.OOF_DIR / 'hard_examples_for_review.csv', index=False, encoding='utf-8-sig',
    )
    groups.to_csv(cfg.OOF_DIR / 'hard_group_candidates.csv', index=False, encoding='utf-8-sig')
    print('\nOOF评分完成。原始金标保持不变，已识别冲突不额外加权。')
    print('难例清单供复核；模型错判本身不能证明金标正确或错误。')
    print('评分文件：', cfg.OOF_DIR / 'train_oof_scores.csv')
    print('接下来运行 train_deberta_oof_hard.py；run_oof_all.py会自动接着运行。')


if __name__ == '__main__':
    main()
