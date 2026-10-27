"""右键运行：固定三个F/B/E专家，仅替换N概率来源；读取CSV，不训练。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.deberta_lexical_mixed_pair import config_high_qwen_fusion as high
from experiments.deberta_lexical_mixed_pair import config_fusion as clean
from experiments.deberta_lexical_mixed_pair.analyze_fusion import load_members, fuse, decode, metrics


RUN_NAME = 'fusion05_replace_gate'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
DIRECTIONS = ['relation_0938', 'scope_0935', 'qwen_r16']
GATES = ['relation_0938', 'qwen_r16', 'lexical_kl', 'native_kl', 'reason_all']
MEMBERS = {name: high.MEMBERS[name] for name in DIRECTIONS}
MEMBERS.update({name: clean.MEMBERS[name] for name in ['lexical_kl', 'native_kl']})
MEMBERS['reason_all'] = {
    'prediction': high.PROJECT_ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/run09_reason_all_seed42/epoch_7/predictions_with_scores.csv',
    'checkpoint': Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run09_reason_all_seed42/best_model.pth'),
    'training_data': 'official Train 3042; supported reason-evidence auxiliary; best independent epoch 7',
}


def main():
    dev = pd.read_csv(high.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev['label'].map(dict(zip(high.FULL_LABELS, range(4)))).to_numpy()
    pairs = np.array([group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False) if len(group) == 2])
    outputs, member_manifest = load_members(dev, SimpleNamespace(MEMBERS=MEMBERS, LABELS=high.LABELS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    pair_reference = None
    for gate in GATES:
        name = f'gate_{gate}'
        logits = fuse(outputs, {'mode': 'hierarchical', 'gates': [gate], 'directions': DIRECTIONS})
        table = dev.copy()
        for index, label in enumerate(high.LABELS):
            table[f'logit_{label}'] = logits[:, index]
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            predicted = decode(logits, pairs, method)
            values = metrics(gold, predicted, pairs)
            table[column] = [high.LABELS[index] for index in predicted]
            if method == 'pair_aware':
                if pair_reference is None:
                    pair_reference = predicted[pairs].copy()
                # 在三组合解码中，每行的(1-g)对所有候选相同，应完全抵消。
                assert np.array_equal(predicted[pairs], pair_reference)
            records.append({
                'gate': gate, 'model_count': len(set(DIRECTIONS + [gate])),
                'decoding': method, **values,
            })
            score_dir = OUTPUT_DIR / 'official_checks' / name / method
            score_dir.mkdir(parents=True, exist_ok=True)
            score_path = score_dir / 'predictions.csv'
            pd.DataFrame({
                'instance_id': dev['instance_id'],
                'label': [high.FULL_LABELS[index] for index in predicted],
            }).to_csv(score_path, index=False)
            report = score_files(high.GOLD_PATH, score_path, output_dir=score_dir)
            assert abs(report.weighted_f1 - values['weighted_f1']) < 1e-12
            assert abs(report.soft_cons - values['soft_cons']) < 1e-12
            assert abs(report.hard_cons - values['hard_cons']) < 1e-12
        table.to_csv(OUTPUT_DIR / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'direction_members': DIRECTIONS, 'gate_candidates': GATES,
        'members': member_manifest,
        'formula': 'g=p_gate(N); q=mean(softmax(z_expert[F,B,E])); p=[(1-g)*q,g]',
        'forced_singleton_N': False, 'per_row_gold_routing': False,
        'weights_or_temperatures_scanned': False,
        'models_loaded': False, 'training_performed': False,
        'selection_scope': 'Five gate candidates compared on reused Dev; exploratory, not held-out validation.',
        'note': 'External gates add a fourth model; original/Qwen gates reuse one of the three experts.',
    }
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary[['gate', 'model_count', 'decoding', 'weighted_f1', 'N_correct', 'FBE_correct', 'non_N_to_N']].to_string(index=False))
    print('输出：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
