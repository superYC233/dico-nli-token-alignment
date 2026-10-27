"""只读取已有结果，不加载模型、不重新预测。"""
import json

import pandas as pd

import config as cfg


def summarize():
    metrics, frames = [], {}
    for method in cfg.METHODS:
        directory = cfg.RESULTS_DIR / method
        if not (directory / 'metrics.json').exists():
            raise FileNotFoundError(f'{method} 尚未完成，请先运行 evaluate_retrieval.py。')
        metrics.extend(json.loads((directory / 'metrics.json').read_text(encoding='utf-8')))
        frames[method] = pd.read_csv(directory / 'predictions_with_scores.csv', keep_default_na=False).set_index('instance_id')
    summary = pd.DataFrame(metrics)
    summary.to_csv(cfg.OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')
    print('\n===== 三组比较：同一个原 C+r16 Adapter =====')
    columns = ['method', 'decoding', 'weighted_f1', 'N_correct', 'FBE_correct',
               'N_f1', 'illegal_generated', 'mean_prompt_tokens']
    print(summary[columns].to_string(index=False))

    comparisons = []
    for new, old in [('random', 'baseline'), ('retrieval', 'baseline'), ('retrieval', 'random')]:
        before, after = frames[old], frames[new]
        if not before.index.is_unique or set(before.index) != set(after.index):
            raise ValueError('结果的 instance_id 不一致，不能逐条比较。')
        after = after.loc[before.index]
        if not before['gold_label'].equals(after['gold_label']):
            raise ValueError('结果的金标不一致。')
        gold = before['gold_label'].map(cfg.LABEL_MAP)
        for decoding, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            was_correct = before[column] == gold
            now_correct = after[column] == gold
            changed = before[column] != after[column]
            old_f1 = summary.loc[(summary['method'] == old) & (summary['decoding'] == decoding), 'weighted_f1'].iloc[0]
            new_f1 = summary.loc[(summary['method'] == new) & (summary['decoding'] == decoding), 'weighted_f1'].iloc[0]
            comparisons.append({
                'comparison': f'{new} minus {old}', 'decoding': decoding,
                'delta_f1_percentage_points': 100 * (new_f1 - old_f1),
                'changed': int(changed.sum()),
                'fixed': int((~was_correct & now_correct).sum()),
                'broken': int((was_correct & ~now_correct).sum()),
                'changed_wrong': int((changed & ~was_correct & ~now_correct).sum()),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(cfg.OUTPUT_DIR / 'comparisons.csv', index=False, encoding='utf-8-sig')
    print('\n===== 逐条改动 =====')
    print(comparison_frame.to_string(index=False))
    demos = pd.read_csv(cfg.PREPARED_DIR / 'demonstrations.csv', keep_default_na=False)
    distribution = demos.groupby(['method', 'source_short_label']).size().rename('count').reset_index()
    distribution.to_csv(cfg.OUTPUT_DIR / 'example_label_distribution.csv', index=False, encoding='utf-8-sig')
    print('\n汇总文件：', cfg.OUTPUT_DIR / 'summary.csv')


if __name__ == '__main__':
    summarize()

