"""右键运行：复用已有CSV，分开合并N门与F/B/E；不加载模型、不训练。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.deberta_lexical_mixed_pair import config_high_qwen_fusion as cfg
from experiments.deberta_lexical_mixed_pair.analyze_fusion import (
    load_members, fuse, decode, metrics, group_bootstrap,
)


RUN_NAME = 'fusion04_hierarchical_existing'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
BASELINE = 'relation_0938'
# 本次同时比较下面三个固定的等权方案；没有扫描权重/温度。
RECIPES = {
    'gate0938_dir_top2': ['relation_0938', 'scope_0935'],
    'gate0938_dir_scope_qwen': ['scope_0935', 'qwen_r16'],
    'gate0938_dir_top2_qwen': ['relation_0938', 'scope_0935', 'qwen_r16'],
}


def main():
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev['label'].map(dict(zip(cfg.FULL_LABELS, range(4)))).to_numpy()
    groups = [group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False)]
    pairs = np.array([group for group in groups if len(group) == 2])
    singletons = np.concatenate([group for group in groups if len(group) == 1])
    outputs, member_manifest = load_members(dev, cfg)
    baseline = decode(outputs[BASELINE]['log_probability'], pairs, 'pair_aware')
    all_logits = {BASELINE: outputs[BASELINE]['log_probability']}
    for name, directions in RECIPES.items():
        all_logits[name] = fuse(outputs, {
            'mode': 'hierarchical', 'gates': [BASELINE], 'directions': directions,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, changes, bootstrap = [], [], []
    for name, logits in all_logits.items():
        table = dev.copy()
        for index, label in enumerate(cfg.LABELS):
            table[f'logit_{label}'] = logits[:, index]
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            predicted = decode(logits, pairs, method)
            values = metrics(gold, predicted, pairs)
            table[column] = [cfg.LABELS[index] for index in predicted]
            records.append({'recipe': name, 'decoding': method, **values})
            score_dir = OUTPUT_DIR / 'official_checks' / name / method
            score_dir.mkdir(parents=True, exist_ok=True)
            score_path = score_dir / 'predictions.csv'
            pd.DataFrame({
                'instance_id': dev['instance_id'],
                'label': [cfg.FULL_LABELS[index] for index in predicted],
            }).to_csv(score_path, index=False)
            report = score_files(cfg.GOLD_PATH, score_path, output_dir=score_dir)
            assert abs(report.weighted_f1 - values['weighted_f1']) < 1e-12
            assert abs(report.soft_cons - values['soft_cons']) < 1e-12
            assert abs(report.hard_cons - values['hard_cons']) < 1e-12
            if name != BASELINE and method == 'pair_aware':
                bootstrap.append({'recipe': name, **group_bootstrap(gold, baseline, predicted, groups, cfg)})
                changes.append({
                    'recipe': name,
                    'fixed': int(((baseline != gold) & (predicted == gold)).sum()),
                    'broken': int(((baseline == gold) & (predicted != gold)).sum()),
                    'singleton_changes': int((predicted[singletons] != baseline[singletons]).sum()),
                })
                changed = table.loc[baseline != predicted].copy()
                changed['baseline_prediction'] = [cfg.LABELS[index] for index in baseline[baseline != predicted]]
                changed['candidate_prediction'] = [cfg.LABELS[index] for index in predicted[baseline != predicted]]
                changed.to_csv(OUTPUT_DIR / f'{name}_changes.csv', index=False, encoding='utf-8-sig')
        table.to_csv(OUTPUT_DIR / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(OUTPUT_DIR / 'fixed_broken.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(bootstrap).to_csv(OUTPUT_DIR / 'paired_bootstrap.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'gate': BASELINE, 'recipes': RECIPES, 'members': member_manifest,
        'formula': 'g=softmax(z_gate)[N]; q=mean(softmax(z_member[F,B,E])); p=[(1-g)*q,g]',
        'forced_singleton_N': False, 'models_loaded': False, 'training_performed': False,
        'development_selection': 'Three recipes compared on repeatedly reused Dev; exploratory results, not held-out evidence.',
        'bootstrap_scope': '5000 source-pair resamples conditional on selected fitted models; no seed/selection correction.',
    }
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['recipe', 'decoding', 'weighted_f1', 'N_correct', 'FBE_correct', 'hard_cons']].to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))
    print('输出：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
