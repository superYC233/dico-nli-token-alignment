"""只读已有预测，比较范围增强与原C+r16；不加载模型、不训练。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path(__file__).resolve().parent / 'outputs/run02_c_scope'
BASELINE_PATH = ROOT / 'experiments/qwen_lora/outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
             'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}


def weighted_f1_from_matrices(matrices):
    support = matrices.sum(axis=-1)
    predicted = matrices.sum(axis=-2)
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    per_class = np.divide(2 * tp, support + predicted,
                          out=np.zeros_like(tp, dtype=float), where=(support + predicted) > 0)
    return (per_class * support).sum(axis=-1) / support.sum(axis=-1)


def paired_bootstrap(frame, gold, before, after, iterations=20000, seed=42):
    groups = pd.factorize(frame.pair_id, sort=False)[0]
    count = int(groups.max() + 1)
    gold_ids = pd.Categorical(gold, categories=LABELS).codes
    tensors = []
    for predictions in [before, after]:
        pred_ids = pd.Categorical(predictions, categories=LABELS).codes
        values = np.zeros((count, 4, 4), dtype=np.int64)
        np.add.at(values, (groups, gold_ids, pred_ids), 1)
        tensors.append(values.reshape(count, 16))
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, iterations, 500):
        size = min(500, iterations - start)
        # 同一抽样权重用于两个模型；每个pair_id内的所有行始终一起出现。
        weights = rng.multinomial(count, np.full(count, 1 / count), size=size)
        old = (weights @ tensors[0]).reshape(size, 4, 4)
        new = (weights @ tensors[1]).reshape(size, 4, 4)
        deltas.append(100 * (weighted_f1_from_matrices(new) - weighted_f1_from_matrices(old)))
    samples = np.concatenate(deltas)
    old_matrix = tensors[0].sum(axis=0).reshape(4, 4)
    new_matrix = tensors[1].sum(axis=0).reshape(4, 4)
    point = float(100 * (weighted_f1_from_matrices(new_matrix) - weighted_f1_from_matrices(old_matrix)))
    sklearn_delta = 100 * (f1_score(gold, after, labels=LABELS, average='weighted') -
                           f1_score(gold, before, labels=LABELS, average='weighted'))
    assert abs(point - sklearn_delta) < 1e-10
    return {'delta_f1_pp': point, 'ci95_pp': np.quantile(samples, [0.025, 0.975]).tolist(),
            'pair_groups': count, 'iterations': iterations, 'seed': seed,
            'limitation': 'Fixed selected models and reused Dev only; excludes seed variance, selection and multiple-testing correction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--baseline', type=Path, default=BASELINE_PATH)
    args = parser.parse_args()
    history = pd.read_csv(args.results_dir / 'epoch_summary.csv')
    structured = history[history.method == 'pair_aware'].sort_values('epoch')
    best = structured.loc[structured.weighted_f1.idxmax()]
    epoch = int(best.epoch)
    current_path = args.results_dir / f'epoch_{epoch}/predictions_with_scores.csv'
    current = pd.read_csv(current_path, keep_default_na=False)
    old = pd.read_csv(args.baseline, keep_default_na=False)
    assert old.instance_id.is_unique and current.instance_id.is_unique
    assert set(old.instance_id) == set(current.instance_id)
    old = old.set_index('instance_id').loc[current.instance_id].reset_index()
    assert old.gold_label.equals(current.gold_label) and old.pair_id.equals(current.pair_id)
    gold = current.gold_label.map(LABEL_MAP)
    output = args.results_dir / 'comparison_with_c_r16'
    output.mkdir(parents=True, exist_ok=True)
    changes, counts, bootstrap = [], [], {}
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before, after = old[column], current[column]
        old_correct, new_correct = before == gold, after == gold
        changed = before != after
        fixed, broken = ~old_correct & new_correct, old_correct & ~new_correct
        changed_wrong = changed & ~old_correct & ~new_correct
        changes.append({'method': method, 'changed': int(changed.sum()), 'fixed': int(fixed.sum()),
                        'broken': int(broken.sum()), 'changed_wrong': int(changed_wrong.sum())})
        table = current[['instance_id', 'pair_id', 'text1', 'text2', 'gold_label']].copy()
        table['before_prediction'], table['after_prediction'] = before, after
        for name, mask in [('fixed', fixed), ('broken', broken), ('changed_wrong', changed_wrong)]:
            table.loc[mask].to_csv(output / f'{method}_{name}.csv', index=False, encoding='utf-8-sig')
        for name, predictions in [('baseline', before), ('scope', after)]:
            matrix = confusion_matrix(gold, predictions, labels=LABELS)
            pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
                output / f'{method}_{name}_confusion.csv', encoding='utf-8-sig')
            record = {'method': method, 'model': name,
                      'weighted_f1': f1_score(gold, predictions, labels=LABELS, average='weighted'),
                      'correct': int((predictions == gold).sum()),
                      'E_to_FB': int(((gold == 'E') & predictions.isin(['F', 'B'])).sum()),
                      'FB_to_E': int((gold.isin(['F', 'B']) & (predictions == 'E')).sum()),
                      'F_B_reversed': int((((gold == 'F') & (predictions == 'B')) |
                                           ((gold == 'B') & (predictions == 'F'))).sum()),
                      'nonN_to_N': int(((gold != 'N') & (predictions == 'N')).sum()),
                      'N_to_FB': int(((gold == 'N') & predictions.isin(['F', 'B'])).sum()),
                      'N_to_E': int(((gold == 'N') & (predictions == 'E')).sum())}
            record.update({f'{label}_correct': int(((gold == label) & (predictions == gold)).sum())
                           for label in LABELS})
            counts.append(record)
        bootstrap[method] = paired_bootstrap(current, gold, before, after)
    comparison = pd.DataFrame(counts)
    comparison.to_csv(output / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(output / 'changes.csv', index=False, encoding='utf-8-sig')
    (output / 'bootstrap.json').write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding='utf-8')
    confidence_rows = []
    for number in sorted(history.epoch.unique()):
        frame = pd.read_csv(args.results_dir / f'epoch_{int(number)}/predictions_with_scores.csv')
        values = frame[[f'logit_{label}' for label in LABELS]].to_numpy()
        weights = np.exp(values - values.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        errors = frame.prediction != frame.gold_label.map(LABEL_MAP)
        confidence_rows.append({
            'epoch': int(number), 'raw_errors': int(errors.sum()),
            'mean_max_probability_4labels': probabilities.max(axis=1).mean(),
            'mean_max_probability_on_raw_errors': probabilities[errors].max(axis=1).mean() if errors.any() else None,
            'predicted_E_pair': int((frame.pair_prediction == 'E').sum()),
        })
    pd.DataFrame(confidence_rows).to_csv(output / 'epoch_confidence.csv', index=False, encoding='utf-8-sig')
    config = json.loads((args.results_dir / 'run_config.json').read_text(encoding='utf-8'))
    baseline = comparison[(comparison.method == 'pair_aware') & (comparison.model == 'baseline')].iloc[0]
    scope = comparison[(comparison.method == 'pair_aware') & (comparison.model == 'scope')].iloc[0]
    raw = comparison[(comparison.method == 'independent') & (comparison.model == 'scope')].iloc[0]
    ci = bootstrap['pair_aware']['ci95_pp']
    report = [
        '# Qwen C+r16 + 200条范围增强：当前Dev结果', '',
        f'- 导出的best_adapter对应第{epoch}轮，pair-aware W-F1={scope.weighted_f1:.9f}。',
        f'- 原C+r16：{baseline.weighted_f1:.9f}；差值{100*(scope.weighted_f1-baseline.weighted_f1):+.4f}个百分点。',
        f'- 两者答对行数：{int(baseline.correct)} → {int(scope.correct)}。',
        f'- 当前最佳结构模型的独立W-F1：{raw.weighted_f1:.9f}。',
        f'- E正确：{int(baseline.E_correct)} → {int(scope.E_correct)}；F/B各正确：'
        f'{int(baseline.F_correct)}/{int(baseline.B_correct)} → {int(scope.F_correct)}/{int(scope.B_correct)}；'
        f'N正确：{int(baseline.N_correct)} → {int(scope.N_correct)}。',
        f'- E→F/B：{int(baseline.E_to_FB)} → {int(scope.E_to_FB)}；'
        f'F/B→E：{int(baseline.FB_to_E)} → {int(scope.FB_to_E)}；'
        f'N→E：{int(baseline.N_to_E)} → {int(scope.N_to_E)}。',
        f'- 固定所选模型，383个pair_id整组bootstrap 20000次：差值95%区间[{ci[0]:+.3f}, {ci[1]:+.3f}]个百分点。', '',
        '当前结果没有证明整体提升。E类观察到改善，同时F/B与N有损失；不能把个别样本变化直接当作机制证据。',
        '最佳独立轮和最佳结构轮可能不同，不能拼成同一模型的成绩。',
        'epoch_confidence.csv是四标签内归一化的条件分布统计，不是校准后的正确概率；各轮错误集合也不同。',
        'bootstrap没有覆盖训练种子、Dev反复使用、checkpoint选择或多重比较，不是独立测试。',
        f'数据：{config["data"]["c_rows"]}条C + {config["data"]["scope_rows"]}条增强；'
        '5轮训练步数由1100增至1165，缺少等预算复制对照。',
        '本实验没有训练期Pair CE；当前独立/结构分差不能归因于Pair CE更新。', '',
        '逐条修好/破坏、两种解码混淆矩阵和原始计数见本目录CSV。',
    ]
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    source_paths = [args.baseline, current_path, args.results_dir / 'run_config.json']
    (output / 'sources.json').write_text(json.dumps(
        [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
         for path in source_paths], ensure_ascii=False, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print(json.dumps(bootstrap, ensure_ascii=False, indent=2))
    print('报告：', output / 'REPORT.md')


if __name__ == '__main__':
    main()
