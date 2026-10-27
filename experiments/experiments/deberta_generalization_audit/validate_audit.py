"""用官方 scorer 和逐条一致性检查审计结果；不会训练模型。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from evaluation_functions import score_files

OUT = Path(__file__).resolve().parent / 'outputs/row_ce_frozen_epoch8'
LABELS = ['F', 'B', 'E', 'N']
FULL = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']


def main():
    dev = pd.read_csv(OUT / 'dev_predictions.csv', keep_default_na=False)
    train = pd.read_csv(OUT / 'train_predictions.csv', keep_default_na=False)
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    for split, frame in [('train', train), ('dev', dev)]:
        assert frame['instance_id'].is_unique
        assert set(frame['prediction']) <= set(LABELS)
        assert np.isfinite(frame[[f'logit_{c}' for c in LABELS]].to_numpy()).all()
        np.testing.assert_allclose(frame[[f'prob_{c}' for c in LABELS]].sum(1), 1, atol=2e-7)
        assert int(frame.correct.sum()) == summary[split]['correct']
    assert not set(train.pair_id) & set(dev.pair_id)
    old = pd.read_csv(ROOT / 'experiments/deberta_row_ce_frozen/outputs/run01_freeze6/epoch_8/predictions_with_scores.csv')
    checked = dev.merge(old[['instance_id', 'prediction', 'pair_prediction']], on='instance_id',
                        suffixes=('', '_old'), validate='one_to_one')
    assert (checked.prediction == checked.prediction_old).all()
    assert (checked.pair_prediction == checked.pair_prediction_old).all()
    references = pd.read_csv(ROOT / 'data/dev/dico_nli_dev_track1_reference.csv', keep_default_na=False)
    checked = dev.merge(references[['instance_id', 'label']], on='instance_id', suffixes=('', '_ref'), validate='one_to_one')
    assert len(checked) == 660 and (checked.label == checked.label_ref).all()
    verified = {}
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        directory = OUT / 'official_validation' / method
        directory.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame({'instance_id': dev.instance_id, 'label': dev[column].map(dict(zip(LABELS, FULL)))})
        path = directory / 'predictions.csv'
        predictions.to_csv(path, index=False)
        report = score_files(ROOT / 'data/dev/dico_nli_dev_track1_reference.csv', path, output_dir=directory)
        f1 = report.classification.weighted_f1
        expected = summary['dev' if method == 'independent' else 'dev_pair_aware']['weighted_f1']
        assert abs(f1 - expected) < 1e-12
        verified[method] = {'weighted_f1': f1, 'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons}
    neighbors = pd.read_csv(OUT / 'dev_top5_train_neighbours.csv', keep_default_na=False)
    assert len(neighbors) == 660 * 5
    assert set(neighbors.train_instance_id) <= set(train.instance_id)
    notes = pd.read_csv(OUT / 'reviewed_error_groups.csv')
    assert len(notes) == 74 and notes.independent_errors.sum() == 101
    assert notes.pair_aware_errors.sum() == 73
    checks = summary['source_audit']['source_snapshot_check']
    assert all(v['only_in_all'] == v['only_in_domains'] == 0 for v in checks.values())
    assert summary['source_audit']['contexts_with_conflicting_token_positions'] == 0
    (OUT / 'validation.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
    print('逐条复现、概率、源文件快照、审核覆盖及官方评分检查通过。')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
