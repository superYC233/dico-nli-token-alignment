"""右键运行：仅官方3042行模型，四类概率等权平均；读取已有CSV，不训练。"""
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.deberta_lexical_mixed_pair import config_fusion as cfg
from experiments.deberta_lexical_mixed_pair import config_high_fusion as high
from experiments.deberta_lexical_mixed_pair.analyze_hierarchical_gate import MEMBERS as gate_members
from experiments.deberta_lexical_mixed_pair.analyze_fusion import load_members, fuse, decode, metrics


RUN_NAME = 'fusion06_official3042_probability'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
MEMBERS = {name: cfg.MEMBERS[name] for name in [
    'lexical_kl', 'native_kl', 'pooler', 'oof', 'staged_lexical',
]}
MEMBERS['original_0924'] = high.MEMBERS['original_0924']
MEMBERS['reason_all'] = gate_members['reason_all']


def main():
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev['label'].map(dict(zip(cfg.FULL_LABELS, range(4)))).to_numpy()
    pairs = np.array([group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False) if len(group) == 2])
    outputs, member_manifest = load_members(dev, SimpleNamespace(MEMBERS=MEMBERS, LABELS=cfg.LABELS))
    recipes = {name: [name] for name in MEMBERS}
    for count in range(2, len(MEMBERS) + 1):
        for names in combinations(MEMBERS, count):
            recipes['+'.join(names)] = list(names)

    records = []
    for recipe, names in recipes.items():
        logits = fuse(outputs, {'mode': 'probability', 'members': names})
        for method in ['independent', 'pair_aware']:
            predicted = decode(logits, pairs, method)
            records.append({
                'recipe': recipe, 'model_count': len(names), 'decoding': method,
                **metrics(gold, predicted, pairs),
            })
    summary = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / 'all_recipes_summary.csv', index=False, encoding='utf-8-sig')

    selected = {}
    for method in ['independent', 'pair_aware']:
        candidates = summary.loc[(summary.decoding == method) & (summary.model_count >= 2)]
        best = candidates.sort_values(['weighted_f1', 'model_count', 'recipe'], ascending=[False, True, True]).iloc[0]
        selected[f'best_{method}'] = recipes[best['recipe']]
    selected_records = []
    for selection, names in selected.items():
        logits = fuse(outputs, {'mode': 'probability', 'members': names})
        table = dev.copy()
        for index, label in enumerate(cfg.LABELS):
            table[f'logit_{label}'] = logits[:, index]
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            predicted = decode(logits, pairs, method)
            values = metrics(gold, predicted, pairs)
            table[column] = [cfg.LABELS[index] for index in predicted]
            selected_records.append({'selection': selection, 'members': '+'.join(names), 'decoding': method, **values})
            directory = OUTPUT_DIR / 'official_checks' / selection / method
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'predictions.csv'
            pd.DataFrame({
                'instance_id': dev['instance_id'],
                'label': [cfg.FULL_LABELS[index] for index in predicted],
            }).to_csv(path, index=False)
            report = score_files(cfg.GOLD_PATH, path, output_dir=directory)
            assert abs(report.weighted_f1 - values['weighted_f1']) < 1e-12
            assert abs(report.soft_cons - values['soft_cons']) < 1e-12
            assert abs(report.hard_cons - values['hard_cons']) < 1e-12
        table.to_csv(OUTPUT_DIR / f'{selection}_predictions.csv', index=False, encoding='utf-8-sig')
    selected_summary = pd.DataFrame(selected_records)
    selected_summary.to_csv(OUTPUT_DIR / 'selected_summary.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'members': member_manifest, 'selected': selected,
        'training_rows_per_model': 3042,
        'data_policy': 'No added phrase rows; excludes scope/relation augmentation and Qwen C reversed-N rows. Reason model has auxiliary evidence annotations; OOF uses reweighting.',
        'formula': 'p=mean(softmax(z_model[F,B,E,N])); independent=argmax(p); pair-aware uses log(p) FB/BF/EE scores',
        'ensemble_candidates_compared': len(recipes) - len(MEMBERS),
        'models_loaded': False, 'training_performed': False, 'forced_singleton_N': False,
        'selection_scope': 'Best of 120 equal-weight subsets of these seven existing models, selected on reused Dev; optimistic, not held-out validation or a global optimum.',
        'checkpoint_scope': 'Historical CSV provenance retained; original_0924 checkpoint is missing at its configured path and was not reloaded.',
    }
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(selected_summary[['selection', 'members', 'decoding', 'weighted_f1', 'N_correct', 'FBE_correct']].to_string(index=False))
    print('输出：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
