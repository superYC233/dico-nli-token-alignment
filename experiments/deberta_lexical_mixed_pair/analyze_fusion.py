"""只融合已有预测并做官方评分/分组bootstrap，不加载或训练模型。"""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.deberta_lexical_mixed_pair import config_fusion as cfg


def log_softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def safe_log(probabilities):
    return np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))


def load_members(dev, config=cfg):
    outputs = {}
    manifest = {}
    for name, member in config.MEMBERS.items():
        path = member['prediction']
        frame = pd.read_csv(path, keep_default_na=False)
        if not frame['instance_id'].is_unique or set(frame['instance_id']) != set(dev['instance_id']):
            raise ValueError(f'{name}：必须是同一Dev的660个唯一instance_id。')
        frame = frame.set_index('instance_id').loc[dev['instance_id']]
        gold_column = 'gold_label' if 'gold_label' in frame else 'label'
        if frame[gold_column].tolist() != dev['label'].tolist() or frame['pair_id'].tolist() != dev['pair_id'].tolist():
            raise ValueError(f'{name}：金标或pair_id与官方Dev不一致。')
        values = frame[[f'logit_{label}' for label in config.LABELS]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}：原始logits存在非有限数。')
        outputs[name] = {
            'log_probability': log_softmax(values),
            'probability': np.exp(log_softmax(values)),
            'direction_probability': np.exp(log_softmax(values[:, :3])),
        }
        prediction_column = 'prediction' if 'prediction' in frame else 'teacher_prediction'
        if prediction_column in frame:
            saved = frame[prediction_column].tolist()
            current = [config.LABELS[i] for i in values.argmax(1)]
            if saved != current:
                raise ValueError(f'{name}：缓存prediction不是四类分数argmax，需先核对推理口径。')
        manifest[name] = {
            'prediction_file': str(path), 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'checkpoint': str(member['checkpoint']), 'checkpoint_exists': member['checkpoint'].exists(),
            'training_data': member['training_data'],
            'checkpoint_content_reloaded': False,
        }
    return outputs, manifest


def fuse(outputs, recipe):
    if recipe['mode'] == 'probability':
        return safe_log(np.mean([outputs[name]['probability'] for name in recipe['members']], axis=0))
    if recipe['mode'] == 'log_probability':
        return log_softmax(np.mean([outputs[name]['log_probability'] for name in recipe['members']], axis=0))
    if recipe['mode'] == 'hierarchical':
        gate = np.mean([outputs[name]['probability'][:, 3] for name in recipe['gates']], axis=0)
        direction = np.mean([outputs[name]['direction_probability'] for name in recipe['directions']], axis=0)
        return safe_log(np.column_stack([(1 - gate[:, None]) * direction, gate]))
    raise ValueError(recipe['mode'])


def decode(logits, pair_indices, mode):
    prediction = logits.argmax(axis=1)
    if mode == 'independent':
        return prediction
    states = np.array([[0, 1], [1, 0], [2, 2]] + ([[3, 3]] if mode == 'twin' else []))
    first, second = pair_indices[:, 0], pair_indices[:, 1]
    scores = logits[first[:, None], states[None, :, 0]] + logits[second[:, None], states[None, :, 1]]
    chosen = states[scores.argmax(axis=1)]
    prediction[first], prediction[second] = chosen[:, 0], chosen[:, 1]
    return prediction


def confusion(gold, prediction, weights=None):
    return np.bincount(4 * gold + prediction, weights=weights, minlength=16).reshape(4, 4)


def weighted_f1(matrix):
    support = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    per_class = np.divide(2 * diagonal, support + predicted,
                          out=np.zeros_like(diagonal, dtype=float), where=support + predicted != 0)
    return (per_class * support).sum(axis=-1) / support.sum(axis=-1)


def metrics(gold, prediction, pairs):
    matrix = confusion(gold, prediction)
    n_predicted = matrix[:, 3].sum()
    soft = ((prediction[pairs] == np.array([0, 1])).all(1) |
            (prediction[pairs] == np.array([1, 0])).all(1) |
            (prediction[pairs] == np.array([2, 2])).all(1))
    hard = (prediction[pairs] == gold[pairs]).all(1)
    return {
        'weighted_f1': float(weighted_f1(matrix)), 'correct': int(matrix.trace()),
        'N_correct': int(matrix[3, 3]), 'non_N_to_N': int(matrix[:3, 3].sum()),
        'N_to_non_N': int(matrix[3, :3].sum()),
        'N_precision': float(matrix[3, 3] / n_predicted) if n_predicted else 0.0,
        'N_recall': float(matrix[3, 3] / matrix[3].sum()),
        'FBE_correct': int(matrix.diagonal()[:3].sum()),
        'F_correct': int(matrix[0, 0]), 'B_correct': int(matrix[1, 1]), 'E_correct': int(matrix[2, 2]),
        'soft_cons': float(soft.mean()), 'hard_cons': float(hard.mean()),
    }


def group_bootstrap(gold, baseline, candidate, groups, config=cfg):
    first = np.array([confusion(gold[g], baseline[g]) for g in groups])
    second = np.array([confusion(gold[g], candidate[g]) for g in groups])
    rng = np.random.default_rng(config.SEED)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=config.BOOTSTRAP_ROUNDS)
    a = (counts @ first.reshape(len(groups), 16)).reshape(-1, 4, 4)
    b = (counts @ second.reshape(len(groups), 16)).reshape(-1, 4, 4)
    delta = weighted_f1(b) - weighted_f1(a)
    return {
        'delta_f1_pp': float((weighted_f1(confusion(gold, candidate)) - weighted_f1(confusion(gold, baseline))) * 100),
        'ci95_low_pp': float(np.percentile(delta, 2.5) * 100),
        'ci95_high_pp': float(np.percentile(delta, 97.5) * 100),
        'bootstrap_positive_fraction': float((delta > 0).mean()),
    }


def main(config=cfg):
    dev = pd.read_csv(config.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev['label'].map(dict(zip(config.FULL_LABELS, range(4)))).to_numpy()
    groups = [group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False)]
    pairs = np.array([group for group in groups if len(group) == 2])
    outputs, manifest = load_members(dev, config)
    all_logits = {name: values['log_probability'] for name, values in outputs.items()}
    all_logits.update({name: fuse(outputs, recipe) for name, recipe in config.RECIPES.items()})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = {}
    for name, logits in all_logits.items():
        predictions[name] = {}
        table = dev[['instance_id', 'pair_id', 'text1', 'text2', 'label']].rename(columns={'label': 'gold_label'}).copy()
        for index, label in enumerate(config.LABELS):
            table[f'logit_{label}'] = logits[:, index]
        for mode, column in [('independent', 'prediction'), ('twin', 'twin_prediction'), ('pair_aware', 'pair_prediction')]:
            predicted = decode(logits, pairs, mode)
            predictions[name][mode] = predicted
            table[column] = [config.LABELS[i] for i in predicted]
            records.append({'recipe': name, 'decoding': mode, **metrics(gold, predicted, pairs)})
        table.to_csv(config.OUTPUT_DIR / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(records)
    summary.to_csv(config.OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')

    changes = []
    baseline = predictions[config.BASELINE]
    for name in config.RECIPES:
        for mode, predicted in predictions[name].items():
            was_correct = baseline[mode] == gold
            now_correct = predicted == gold
            changes.append({'recipe': name, 'decoding': mode,
                            'fixed': int((~was_correct & now_correct).sum()),
                            'broken': int((was_correct & ~now_correct).sum())})
    pd.DataFrame(changes).to_csv(config.OUTPUT_DIR / 'fixed_broken.csv', index=False, encoding='utf-8-sig')
    overlaps = []
    for first, second in combinations(outputs, 2):
        a, b = predictions[first]['independent'], predictions[second]['independent']
        overlaps.append({'first': first, 'second': second,
                         'both_wrong': int(((a != gold) & (b != gold)).sum()),
                         'first_only_correct': int(((a == gold) & (b != gold)).sum()),
                         'second_only_correct': int(((a != gold) & (b == gold)).sum()),
                         'disagree': int((a != b).sum())})
    pd.DataFrame(overlaps).to_csv(config.OUTPUT_DIR / 'error_overlap.csv', index=False, encoding='utf-8-sig')
    bootstrap = []
    for name in config.BOOTSTRAP_RECIPES:
        for mode in ['independent', 'pair_aware']:
            bootstrap.append({'recipe': name, 'decoding': mode, 'baseline': config.BASELINE,
                              **group_bootstrap(gold, baseline[mode], predictions[name][mode], groups, config)})
    pd.DataFrame(bootstrap).to_csv(config.OUTPUT_DIR / 'paired_bootstrap.csv', index=False, encoding='utf-8-sig')

    # 只对实际推荐的固定方案再调用官方评分器，核对离线公式。
    for name in config.BOOTSTRAP_RECIPES:
        for mode in ['independent', 'pair_aware']:
            directory = config.OUTPUT_DIR / 'official_checks' / name / mode
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'predictions.csv'
            pd.DataFrame({'instance_id': dev['instance_id'],
                          'label': [config.FULL_LABELS[i] for i in predictions[name][mode]]}).to_csv(path, index=False)
            report = score_files(config.GOLD_PATH, path, output_dir=directory)
            expected = summary.loc[(summary.recipe == name) & (summary.decoding == mode)].iloc[0]
            assert abs(report.weighted_f1 - expected['weighted_f1']) < 1e-12
            assert abs(report.soft_cons - expected['soft_cons']) < 1e-12
            assert abs(report.hard_cons - expected['hard_cons']) < 1e-12
    info = {
        'members': manifest, 'recipes': config.RECIPES,
        'dev_sha256': hashlib.sha256(config.DEV_PATH.read_bytes()).hexdigest(),
        'rows': len(dev), 'source_groups': len(groups), 'bootstrap_rounds': config.BOOTSTRAP_ROUNDS,
        'method': 'equal-weight global score fusion; no per-row gold routing; no forced singleton N',
        'development_selection': 'member checkpoints and recipe comparison use Dev; exploratory, optimistic',
        'bootstrap_scope': 'paired source-group resampling conditional on fitted/selected models; does not cover seed variance or selection bias',
        'models_reloaded': False, 'training_performed': False,
        'source_sha256': {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(config.__file__), Path(__file__)]
        },
    }
    (config.OUTPUT_DIR / 'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.loc[summary.decoding.isin(['independent', 'pair_aware'])].to_string(index=False))
    print('\n选定方案的分组bootstrap（差值单位：百分点）：')
    print(pd.DataFrame(bootstrap).to_string(index=False))
    print('\n输出：', config.OUTPUT_DIR)


if __name__ == '__main__':
    main()
