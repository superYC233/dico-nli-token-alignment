import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.deberta_feature_fusion import config as cfg
from experiments.deberta_lexical.evaluation import (
    get_pair_logits, pair_aware, predict_logits, score_logits as original_score_logits,
)


def score_logits(logits, dev, output_dir):
    output_dir = Path(output_dir)
    summary = original_score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH)
    gold = torch.tensor([cfg.LABEL_TO_ID[label] for label in dev['label']])
    independent = logits.argmax(dim=1)
    paired = pair_aware(logits, dev)
    non_n = gold != 3
    threeway_correct = int(((logits[:, :3].argmax(dim=1) == gold) & non_n).sum())
    for method, predictions in [('independent', independent), ('pair_aware', paired)]:
        selected = summary['method'] == method
        summary.loc[selected, 'N_predicted'] = int((predictions == 3).sum())
        summary.loc[selected, 'non_N_to_N'] = int(((predictions == 3) & non_n).sum())
        summary.loc[selected, 'N_to_non_N'] = int(((predictions != 3) & ~non_n).sum())

    changed = independent != paired
    first_correct = independent == gold
    second_correct = paired == gold
    diagnostics = {
        'independent_FBE_threeway_correct': threeway_correct,
        'FBE_rows': int(non_n.sum()),
        'decode_changed': int(changed.sum()),
        'decode_fixed': int((~first_correct & second_correct).sum()),
        'decode_broken': int((first_correct & ~second_correct).sum()),
        'decode_changed_wrong': int((changed & ~first_correct & ~second_correct).sum()),
    }
    (output_dir / 'diagnostics.json').write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    count_columns = ['N_predicted', 'non_N_to_N', 'N_to_non_N']
    summary[count_columns] = summary[count_columns].astype(int)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print('N预测数量：')
    print(summary[['method'] + count_columns].to_string(index=False))
    print('非N行独立F/B/E排序（仅诊断）：', threeway_correct, '/', int(non_n.sum()))
    print('结构解码：修好', diagnostics['decode_fixed'], '；破坏', diagnostics['decode_broken'])
    return summary


def evaluate_without_changing_rng(model, loader, dev, device, output_dir):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    try:
        return score_logits(predict_logits(model, loader, device), dev, output_dir)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def compare_with_lexical(prediction_path, output_dir):
    # 预测完成后才读取历史结果，不进入训练、特征或解码。
    if not cfg.REFERENCE_PREDICTIONS.exists():
        print('未找到历史0.930569预测，跳过逐条比较：', cfg.REFERENCE_PREDICTIONS)
        return
    old = pd.read_csv(cfg.REFERENCE_PREDICTIONS, keep_default_na=False)
    current = pd.read_csv(prediction_path, keep_default_na=False)
    columns = ['instance_id', 'gold_label', 'prediction', 'pair_prediction']
    merged = current.merge(old[columns], on='instance_id', suffixes=('', '_old'), validate='one_to_one')
    if len(merged) != len(current) or not merged['gold_label'].eq(merged['gold_label_old']).all():
        raise ValueError('历史Lexical与本次Dev的样本或金标不同。')
    gold = merged['gold_label'].map(dict(zip(cfg.FULL_LABELS, cfg.LABELS)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        old_correct = merged[f'{column}_old'] == gold
        new_correct = merged[column] == gold
        fixed = merged[~old_correct & new_correct]
        broken = merged[old_correct & ~new_correct]
        fixed.to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        broken.to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
        print(f'相对0.930569的{method}：修好 {len(fixed)} 条，破坏 {len(broken)} 条。')
