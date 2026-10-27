"""只复算已有Lexical预测、数据审计和解析例子；不加载模型，不训练。"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / 'outputs/independent_research_20261025'
LABELS = ['F', 'B', 'E', 'N']
FULL = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
TO_SHORT = dict(zip(FULL, LABELS))
PREDICTIONS = {
    'lexical_staged_ep3': ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv',
    'lexical_mixed_ep2': ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/run01_mixed_pair/epoch_2/predictions_with_scores.csv',
    'lexical_oof_hard_ep7': ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/run06_oof_hard/epoch_7/predictions_with_scores.csv',
}
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
SUBTYPE_PATH = ROOT / 'experiments/qwen_lora/outputs/sixclass_comparison/run01/all_comparison.csv'
TEACHER_PATH = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged/error_analysis/train_readout.csv'


def softmax(values, temperature=1.0):
    shifted = (values - values.max(axis=1, keepdims=True)) / temperature
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def confusion(gold, prediction):
    result = np.zeros((4, 4), dtype=np.int64)
    for first, second in zip(gold, prediction):
        result[LABELS.index(first), LABELS.index(second)] += 1
    return result


def weighted_f1(matrix):
    support = matrix.sum(axis=1)
    denominator = support + matrix.sum(axis=0)
    scores = np.divide(2 * matrix.diagonal(), denominator,
                       out=np.zeros(4, dtype=float), where=denominator != 0)
    return float((support * scores).sum() / support.sum())


def analyze_predictions(name, path, subtypes):
    frame = pd.read_csv(path, keep_default_na=False)
    frame['gold'] = frame.gold_label.map(TO_SHORT)
    logits = frame[[f'logit_{label}' for label in LABELS]].to_numpy()
    probability = softmax(logits)
    frame['prob_N'] = probability[:, 3]
    frame['max_probability'] = probability.max(axis=1)
    frame['n_subtype'] = frame.instance_id.map(subtypes).fillna('')
    wrong = frame.gold.ne(frame.prediction)
    n_boundary = wrong & (frame.gold.eq('N') | frame.prediction.eq('N'))
    e_fb = wrong & ((frame.gold.eq('E') & frame.prediction.isin(['F', 'B'])) |
                    (frame.gold.isin(['F', 'B']) & frame.prediction.eq('E')))
    matrix = confusion(frame.gold, frame.prediction)
    paired_matrix = confusion(frame.gold, frame.pair_prediction)
    was_correct = frame.gold.eq(frame.prediction)
    now_correct = frame.gold.eq(frame.pair_prediction)
    gold_n = frame.gold.eq('N')
    top_n = frame.prob_N.nlargest(90).index
    stats = {
        'model': name, 'independent_f1': weighted_f1(matrix),
        'pair_aware_f1': weighted_f1(paired_matrix),
        'independent_correct': int(matrix.trace()),
        'independent_FBE_correct': int(matrix.diagonal()[:3].sum()),
        'pair_FBE_correct': int(paired_matrix.diagonal()[:3].sum()),
        'N_correct': int(matrix[3, 3]), 'non_N_to_N': int(matrix[:3, 3].sum()),
        'N_to_non_N': int(matrix[3, :3].sum()),
        'errors': int(wrong.sum()), 'N_boundary_errors': int(n_boundary.sum()),
        'E_FB_errors': int(e_fb.sum()),
        'F_B_errors': int((wrong & frame.gold.isin(['F', 'B']) & frame.prediction.isin(['F', 'B'])).sum()),
        'N_ROC_AUC': float(roc_auc_score(gold_n, frame.prob_N)),
        'N_average_precision': float(average_precision_score(gold_n, frame.prob_N)),
        'N_TP_at_fixed_top90_by_probN': int(gold_n.loc[top_n].sum()),
        'decoder_fixed': int((~was_correct & now_correct).sum()),
        'decoder_broken': int((was_correct & ~now_correct).sum()),
        'checkpoint_note': ('epoch2 has saved predictions but no matching saved model weights'
                            if name == 'lexical_mixed_ep2' else 'historical selected run/checkpoint'),
    }
    group_rows = []
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        first, second = group.prediction.tolist()
        if first == second == 'N':
            kind = 'NN_consistent_but_wrong_on_true_FBE_pair'
        elif (first, second) in [('F', 'B'), ('B', 'F'), ('E', 'E')]:
            kind = 'FB_BF_EE'
        else:
            kind = 'swap_inconsistent'
        group_rows.append({'kind': kind, 'wrong_rows': int(group.gold.ne(group.prediction).sum())})
    group_summary = pd.DataFrame(group_rows).groupby('kind').agg(
        groups=('kind', 'size'), wrong_rows=('wrong_rows', 'sum'),
    ).reset_index()
    group_summary.insert(0, 'model', name)
    frame[wrong].to_csv(OUTPUT_DIR / f'{name}_independent_errors.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(OUTPUT_DIR / f'{name}_confusion.csv', encoding='utf-8-sig')
    subtype_summary = frame[gold_n].assign(correct=lambda data: data.prediction.eq('N')).groupby('n_subtype').agg(
        rows=('instance_id', 'size'), correct=('correct', 'sum'),
    ).reset_index()
    subtype_summary.insert(0, 'model', name)
    return stats, group_summary, subtype_summary, frame


def bias_only_oracle(frame):
    # 只分析固定模型的N轴可调范围。使用Dev金标，不是可提交预测或已验证调参结果。
    logits = frame[[f'logit_{label}' for label in LABELS]].to_numpy()
    gold = frame.gold.map(dict(zip(LABELS, range(4)))).to_numpy()
    non_n_best = logits[:, :3].argmax(axis=1)
    thresholds = logits[:, :3].max(axis=1) - logits[:, 3]
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(gold, non_n_best):
        matrix[target, prediction] += 1
    records = []
    order = np.argsort(thresholds, kind='stable')
    for count in range(len(frame) + 1):
        if count:
            index = order[count - 1]
            matrix[gold[index], non_n_best[index]] -= 1
            matrix[gold[index], 3] += 1
        # 同阈值的行必须一起变化；不使用无法由一个统一bias实现的中间状态。
        if count and count < len(frame) and thresholds[order[count - 1]] == thresholds[order[count]]:
            continue
        records.append({
            'N_predicted': count, 'weighted_f1': weighted_f1(matrix),
            'N_correct': int(matrix[3, 3]), 'non_N_to_N': int(matrix[:3, 3].sum()),
            'correct': int(matrix.trace()),
        })
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / 'mixed_ep2_bias_only_oracle.csv', index=False, encoding='utf-8-sig')
    return {
        'best_dev_oracle': result.loc[result.weighted_f1.idxmax()].to_dict(),
        'first_threshold_N_recall_at_least_96': result[result.N_correct.ge(96)].iloc[0].to_dict(),
        'scope': 'Retrospective Dev-gold oracle. Ranking of F/B/E never changes. Not a training result or proposed deployment rule.',
    }


def data_audit():
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    normalize = lambda value: ' '.join(value.casefold().split())
    unordered = lambda first, second: tuple(sorted((normalize(first), normalize(second))))
    train_pairs = defaultdict(set)
    for row in train.itertuples(index=False):
        train_pairs[unordered(row.text1, row.text2)].add(row.pair_id)
    duplicates = []
    for row in dev.itertuples(index=False):
        key = unordered(row.text1, row.text2)
        if key in train_pairs:
            duplicates.append({
                'instance_id': row.instance_id, 'pair_id': row.pair_id,
                'text1': row.text1, 'text2': row.text2, 'label': row.label,
                'train_pair_ids': '|'.join(sorted(train_pairs[key])),
            })
    duplicate_frame = pd.DataFrame(duplicates)
    duplicate_frame.to_csv(OUTPUT_DIR / 'official_train_dev_pair_overlap.csv', index=False, encoding='utf-8-sig')
    marked = train.assign(a=train.text1.map(normalize), b=train.text2.map(normalize))
    conflicts = marked[marked.groupby(['a', 'b']).label.transform('nunique').gt(1)]
    conflicts.to_csv(OUTPUT_DIR / 'official_train_conflicting_inputs.csv', index=False, encoding='utf-8-sig')
    return {
        'normalization': 'casefold and whitespace collapse only',
        'unique_unordered_input_pairs_shared': len({unordered(row.text1, row.text2) for row in duplicate_frame.itertuples()}),
        'dev_rows_with_shared_input_pair': len(duplicate_frame),
        'dev_pair_ids_with_shared_input_pair': int(duplicate_frame.pair_id.nunique()),
        'train_conflicting_ordered_inputs': len(conflicts[['a', 'b']].drop_duplicates()),
        'train_conflict_rows': len(conflicts), 'data_modified': False,
    }


def structured_teacher_diagnostic():
    teacher = pd.read_csv(TEACHER_PATH, keep_default_na=False)
    scores = []
    for _, group in teacher.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        first, second = group.iloc[0], group.iloc[1]
        scores.append([first.logit_F + second.logit_B, first.logit_B + second.logit_F,
                       first.logit_E + second.logit_E])
    scores = np.asarray(scores)
    records = []
    for temperature in [1, 2, 4]:
        probability = softmax(scores, temperature)
        records.append({
            'temperature': temperature, 'groups': len(scores),
            'mean_max_probability': float(probability.max(axis=1).mean()),
            'fraction_max_probability_above_0p98': float((probability.max(axis=1) > .98).mean()),
            'mean_entropy': float(-(probability * np.log(probability.clip(1e-15))).sum(axis=1).mean()),
        })
    pd.DataFrame(records).to_csv(OUTPUT_DIR / 'structured_teacher_train_distribution.csv', index=False, encoding='utf-8-sig')
    return records


def mathematical_examples():
    # 构造例用于反驳“N必须96才能达到0.87”，不是某个实际模型的预测。
    matrix = np.array([[180, 0, 0, 10], [0, 180, 0, 10], [12, 12, 150, 0], [0, 0, 25, 81]])
    row_gradient = np.array([2., 1., -3.])
    pair_gradient = np.array([-1., 2., 1.])
    fisher_mask = np.array([1., 0., 1.])
    return {
        'hypothetical_N81_confusion': matrix.tolist(),
        'hypothetical_N81_weighted_f1': weighted_f1(matrix),
        'fisher_mask_counterexample': {
            'row_gradient': row_gradient.tolist(), 'pair_gradient': pair_gradient.tolist(),
            'mask': fisher_mask.tolist(), 'unmasked_dot': float(row_gradient @ pair_gradient),
            'masked_dot': float(row_gradient @ (fisher_mask * pair_gradient)),
            'meaning': 'A Fisher coordinate mask does not by itself remove Row/Pair gradient conflict.',
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subtypes = pd.read_csv(SUBTYPE_PATH, keep_default_na=False).set_index('instance_id').n_relation
    stats, groups, subtype_tables, frames = [], [], [], {}
    for name, path in PREDICTIONS.items():
        summary, group_table, subtype_table, frame = analyze_predictions(name, path, subtypes)
        stats.append(summary)
        groups.append(group_table)
        subtype_tables.append(subtype_table)
        frames[name] = frame
    pd.DataFrame(stats).to_csv(OUTPUT_DIR / 'independent_error_summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(groups).to_csv(OUTPUT_DIR / 'swap_group_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.concat(subtype_tables).to_csv(OUTPUT_DIR / 'N_subtype_diagnostics.csv', index=False, encoding='utf-8-sig')
    report = {
        'training_performed': False, 'model_inference_performed': False,
        'errors': stats, 'data_audit': data_audit(),
        'bias_oracle': bias_only_oracle(frames['lexical_mixed_ep2']),
        'structured_teacher': structured_teacher_diagnostic(),
        'math': mathematical_examples(),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*PREDICTIONS.values(), TRAIN_PATH, DEV_PATH, SUBTYPE_PATH, TEACHER_PATH]},
    }
    (OUTPUT_DIR / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(stats)[['model', 'independent_f1', 'pair_aware_f1', 'errors', 'N_boundary_errors', 'E_FB_errors']].to_string(index=False))
    print('data audit:', report['data_audit'])
    print('bias-only Dev oracle:', report['bias_oracle'])
    print('输出：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
