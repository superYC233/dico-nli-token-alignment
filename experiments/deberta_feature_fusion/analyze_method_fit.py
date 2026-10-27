"""只读已有预测并做CPU数值核验；不加载模型权重，不训练。"""
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.deberta_lexical.lexical_features import build_lexical_features


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / 'outputs/method_fit'
LABELS = ['F', 'B', 'E', 'N']
FULL_TO_SHORT = dict(zip([
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
], LABELS))
ORIGINAL_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
ORIGINAL_PREDICTIONS = ORIGINAL_DIR / 'epoch_2/predictions_with_scores.csv'
LEXICAL_DIR = ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42'
FUSION_DIR = HERE / 'outputs/run01_feature_fusion/feature_fusion_seed_42'
FAMILY_DIR = ROOT / 'experiments/deberta_family_aug/outputs'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'


def load_predictions(path):
    frame = pd.read_csv(path, keep_default_na=False)
    frame['gold'] = frame['gold_label'].map(FULL_TO_SHORT)
    return frame


def confusion(frame, column):
    result = np.zeros((4, 4), dtype=np.int64)
    for gold, prediction in zip(frame.gold, frame[column]):
        result[LABELS.index(gold), LABELS.index(prediction)] += 1
    return result


def weighted_f1(matrix):
    support = matrix.sum(axis=1)
    denominator = support + matrix.sum(axis=0)
    f1 = np.divide(2 * matrix.diagonal(), denominator,
                   out=np.zeros(4, dtype=float), where=denominator != 0)
    return float((support * f1).sum() / support.sum())


def summarize_predictions(name, frame):
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        matrix = confusion(frame, column)
        records.append({
            'model': name, 'method': method, 'weighted_f1': weighted_f1(matrix),
            'correct': int(matrix.trace()),
            'FBE_correct': int(matrix.diagonal()[:3].sum()),
            'N_correct': int(matrix[3, 3]),
            'non_N_to_N': int(matrix[:3, 3].sum()),
        })
    return records


def oracle_error_budget(frame):
    # 使用金标回顾性修正错误，只计算理想收益空间，绝不是部署预测规则。
    base = confusion(frame, 'pair_prediction')
    changes = []
    for pair_id, group in frame.groupby('pair_id', sort=False):
        wrong = group[group.gold != group.pair_prediction]
        boundary = ((wrong.gold == 'E') & wrong.pair_prediction.isin(['F', 'B'])) | (
            wrong.gold.isin(['F', 'B']) & (wrong.pair_prediction == 'E'))
        if len(group) == 2 and len(wrong) == 2 and boundary.all():
            delta = np.zeros((4, 4), dtype=np.int64)
            for row in wrong.itertuples(index=False):
                gold, predicted = LABELS.index(row.gold), LABELS.index(row.pair_prediction)
                delta[gold, predicted] -= 1
                delta[gold, gold] += 1
            changes.append((pair_id, delta))
    records = []
    for count in range(1, 6):
        scores = [weighted_f1(base + sum((changes[i][1] for i in indices),
                                         np.zeros((4, 4), dtype=np.int64)))
                  for indices in itertools.combinations(range(len(changes)), count)]
        records.append({
            'corrected_E_FB_groups': count, 'corrected_rows': 2 * count,
            'scenarios': len(scores), 'min_oracle_weighted_f1': min(scores),
            'max_oracle_weighted_f1': max(scores),
            'assumption': 'gold-oracle fixes only; all other predictions unchanged',
        })
    all_target = base + sum((delta for _, delta in changes), np.zeros((4, 4), dtype=np.int64))
    n_fixed = base.copy()
    n_fixed[3, :] = 0
    n_fixed[3, 3] = base[3].sum()
    return records, {
        'base_weighted_f1': weighted_f1(base), 'base_confusion': base.tolist(),
        'target_groups': len(changes), 'target_rows': 2 * len(changes),
        'all_E_FB_fixed_oracle_f1': weighted_f1(all_target),
        'only_five_N_errors_fixed_oracle_f1': weighted_f1(n_fixed),
        'limitation': 'Oracle accounting is not an estimate of method improvement.',
    }


def lexical_collision_audit(frame, split):
    groups = defaultdict(list)
    for row in frame.itertuples(index=False):
        vector = tuple(build_lexical_features(row.text1, row.text2))
        groups[vector].append(row)
    maximum_correct = sum(max(Counter(row.label for row in rows).values())
                          for rows in groups.values())
    ambiguous = [rows for rows in groups.values() if len({row.label for row in rows}) > 1]
    examples = []
    for group_id, rows in enumerate(sorted(ambiguous, key=len, reverse=True)[:12]):
        chosen = {}
        for row in rows:
            chosen.setdefault(row.label, row)
        for label, row in chosen.items():
            examples.append({
                'split': split, 'collision_id': group_id, 'collision_rows': len(rows),
                'instance_id': row.instance_id, 'text1': row.text1, 'text2': row.text2,
                'gold': FULL_TO_SHORT[label],
            })
    return {
        'split': split, 'rows': len(frame), 'distinct_exact_feature_vectors': len(groups),
        'mixed_label_feature_vectors': len(ambiguous),
        'rows_in_mixed_label_vectors': sum(map(len, ambiguous)),
        'feature_only_max_in_sample_correct': maximum_correct,
        'feature_only_max_in_sample_accuracy': maximum_correct / len(frame),
        'scope': '20 lexical features alone; not a bound on the combined DeBERTa model',
    }, examples


def numerical_examples():
    # 例一：两个任务读出分离后，更新共享表示仍可能使Row变差。
    states = []
    for value in [2.0, 0.5]:
        h = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        zero, one = h * 0, h * 0 + 1
        row_logits = torch.stack([h, zero, zero, one]).unsqueeze(0)
        pair_logits = torch.stack([-2 * h, zero, zero]).unsqueeze(0)
        row_loss = F.cross_entropy(row_logits, torch.tensor([0]))
        pair_loss = F.cross_entropy(pair_logits, torch.tensor([0]))
        row_grad = torch.autograd.grad(row_loss, h, retain_graph=True)[0]
        pair_grad = torch.autograd.grad(pair_loss, h)[0]
        states.append({
            'shared_h': value, 'row_logits': row_logits.detach().tolist()[0],
            'row_prediction': LABELS[int(row_logits.argmax())],
            'pair_logits': pair_logits.detach().tolist()[0],
            'row_ce': row_loss.item(), 'pair_ce': pair_loss.item(),
            'row_plus_half_pair_ce': row_loss.item() + 0.5 * pair_loss.item(),
            'd_row_d_h': row_grad.item(), 'd_pair_d_h': pair_grad.item(),
            'd_row_plus_half_pair_d_h': row_grad.item() + 0.5 * pair_grad.item(),
        })

    # 例二：相同三组合上的CRF条件NLL，与当前三组合CE是同一个式子。
    first = torch.tensor([2.0, 0.1, 0.5, 2.5], dtype=torch.float64)
    second = torch.tensor([0.2, 1.8, 0.4, 2.2], dtype=torch.float64)
    scores = torch.stack([first[0] + second[1], first[1] + second[0], first[2] + second[2]])
    ce3 = F.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
    crf3 = torch.logsumexp(scores, 0) - scores[0]
    all_scores = (first[:, None] + second[None, :]).reshape(1, 16)
    ce16 = F.cross_entropy(all_scores, torch.tensor([1]))
    row_sum = F.cross_entropy(first.unsqueeze(0), torch.tensor([0])) + F.cross_entropy(
        second.unsqueeze(0), torch.tensor([1]))
    assert torch.allclose(ce3, crf3, atol=1e-12)
    assert torch.allclose(ce16, row_sum, atol=1e-12)

    # 例三：相同词汇特征可以对应三种不同的普通字面关系。
    illustrative = [('automobiles', 'cars', 'E'), ('cars', 'vehicles', 'F'), ('cars', 'bananas', 'N')]
    features = [build_lexical_features(a, b) for a, b, _ in illustrative]
    assert features[0] == features[1] == features[2]

    # 例四：局部包含关系在no的向下蕴含语境里反向，不能直接查词典后投票。
    worlds = [set(values) for count in range(3)
              for values in itertools.combinations(['car', 'truck'], count)]
    no_car = {index for index, world in enumerate(worlds) if 'car' not in world}
    no_vehicle = {index for index, world in enumerate(worlds) if not world}
    return {
        'separate_heads_shared_representation_counterexample': states,
        'CRF_same_three_states_ce': ce3.item(), 'CRF_same_three_states_nll': crf3.item(),
        'all_16_additive_ce': ce16.item(), 'sum_of_row_ce': row_sum.item(),
        'identical_lexical_feature_examples': illustrative,
        'natural_logic_illustration': {
            'positive_cars_vs_vehicles': 'F', 'no_cars_vs_no_vehicles': 'B',
            'no_car_worlds': sorted(no_car), 'no_vehicle_worlds': sorted(no_vehicle),
            'scope': 'finite illustrative semantics, not a DiCo relabeling rule',
        },
    }


def semantic_group_budget(frame):
    review_path = ORIGINAL_DIR / 'error_analysis/error_groups_reviewed.csv'
    review = pd.read_csv(review_path, keep_default_na=False)
    target = review[
        ((review.gold_combination == 'EE') & review.current_combination.isin(['FB', 'BF'])) |
        (review.gold_combination.isin(['FB', 'BF']) & (review.current_combination == 'EE'))
    ].copy()
    base = confusion(frame, 'pair_prediction')
    records = []
    for name, group in target.groupby('primary_type', sort=False):
        selected = frame[frame.pair_id.isin(group.pair_id)]
        oracle = base.copy()
        for row in selected.itertuples(index=False):
            gold, prediction = LABELS.index(row.gold), LABELS.index(row.pair_prediction)
            oracle[gold, prediction] -= 1
            oracle[gold, gold] += 1
        records.append({
            'reviewed_type': name, 'groups': len(group), 'rows': len(selected),
            'never_correct_groups_in_ten_epochs': int((group.correct_epoch_count == 0).sum()),
            'sometimes_correct_groups_in_ten_epochs': int((group.correct_epoch_count > 0).sum()),
            'all_type_errors_fixed_oracle_f1': weighted_f1(oracle),
            'scope': 'Assistant-reviewed types; not independently validated causes or learnability bounds.',
        })
    target.to_csv(OUTPUT_DIR / 'original_E_FB_target_groups.csv', index=False, encoding='utf-8-sig')
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        'original_0p938': ORIGINAL_PREDICTIONS,
        'lexical_0p9306': LEXICAL_DIR / 'epoch_3/predictions_with_scores.csv',
        'feature_fusion': FUSION_DIR / 'epoch_10/predictions_with_scores.csv',
        'family768_manual': FAMILY_DIR / 'run01_family768_manual/epoch_1/predictions_with_scores.csv',
        'family768_copy': FAMILY_DIR / 'run01_family768_copy/epoch_1/predictions_with_scores.csv',
    }
    prediction_frames = {name: load_predictions(path) for name, path in paths.items()}
    metrics = pd.DataFrame([record for name, frame in prediction_frames.items()
                            for record in summarize_predictions(name, frame)])
    metrics.to_csv(OUTPUT_DIR / 'existing_model_metrics.csv', index=False, encoding='utf-8-sig')
    original = prediction_frames['original_0p938']
    oracle, oracle_facts = oracle_error_budget(original)
    pd.DataFrame(oracle).to_csv(OUTPUT_DIR / 'oracle_error_budget.csv', index=False, encoding='utf-8-sig')
    semantic = semantic_group_budget(original)
    pd.DataFrame(semantic).to_csv(OUTPUT_DIR / 'semantic_group_budget.csv', index=False, encoding='utf-8-sig')

    collision_stats, collision_examples = [], []
    for split, path in [('train', TRAIN_PATH), ('dev', DEV_PATH)]:
        stats, examples = lexical_collision_audit(pd.read_csv(path, keep_default_na=False), split)
        collision_stats.append(stats)
        collision_examples.extend(examples)
    pd.DataFrame(collision_stats).to_csv(OUTPUT_DIR / 'lexical_collision_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(collision_examples).to_csv(OUTPUT_DIR / 'lexical_collision_examples.csv', index=False, encoding='utf-8-sig')

    source = pd.read_csv(ROOT / 'experiments/deberta_family_aug/prepared/examples_source.csv')
    source.groupby(['split', 'family_type', 'contrast_type', 'label']).size().rename(
        'source_pairs').reset_index().to_csv(OUTPUT_DIR / 'existing_family_coverage.csv', index=False, encoding='utf-8-sig')
    probe_path = FAMILY_DIR / 'run01_family768_manual/probe_comparison_vs_0p938.csv'
    probe = pd.read_csv(probe_path)
    payload = {
        'training_performed': False, 'model_weights_loaded': False, 'device': 'cpu',
        'oracle_budget': oracle_facts, 'numerical_examples': numerical_examples(),
        'semantic_error_types': semantic,
        'lexical_collisions': collision_stats, 'existing_heldout_family_results': probe.to_dict('records'),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*paths.values(), TRAIN_PATH, DEV_PATH, probe_path]},
        'interpretation': 'Theoretical counterexamples, existing runs, and gold-oracle upper bounds. No estimate of future F1 or success probability.',
    }
    (OUTPUT_DIR / 'calculations.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))
    print('\nOracle only; not an expected training gain:')
    print(pd.DataFrame(oracle).drop(columns='assumption').to_string(index=False))
    print('\nLexical feature collisions:')
    print(pd.DataFrame(collision_stats).drop(columns='scope').to_string(index=False))
    print('\nReviewed semantic categories; oracle gains, not method predictions:')
    print(pd.DataFrame(semantic).drop(columns='scope').to_string(index=False))
    print('\nNumerical examples:')
    print(json.dumps(payload['numerical_examples'], ensure_ascii=False, indent=2))
    print('\n输出目录：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
