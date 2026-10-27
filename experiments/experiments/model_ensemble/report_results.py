"""复核已搜索组合，保存官方评分、成员清单及中文报告；不运行模型。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.model_ensemble.analyze_all_models import OUTPUT, DEV_PATH, FULL, LABELS, EnsembleSearch
from experiments.deberta_lexical_mixed_pair.analyze_fusion import decode, metrics, group_bootstrap
from experiments.deberta_lexical_mixed_pair import config_fusion as bootstrap_cfg


def main():
    members = json.loads((OUTPUT / 'members.json').read_text(encoding='utf-8'))
    lp = np.load(OUTPUT / 'aligned_scores.npz')['log_probability']
    for member, values in zip(members, lp):
        member['log_probability'] = values
    dev = pd.read_csv(DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev.label.map(dict(zip(FULL, range(4)))).to_numpy()
    groups = [group.index.to_numpy() for _, group in dev.groupby('pair_id', sort=False)]
    pairs = np.array([group for group in groups if len(group) == 2])
    search = EnsembleSearch(members, gold, pairs)
    by_id = {member['id']: (index, member) for index, member in enumerate(members)}
    best = {}
    large = [m['id'] for m in members if 'large' in m['model_name'].lower()]
    qwen = [m['id'] for m in members if 'qwen' in m['model_name'].lower()]
    modern = [m['id'] for m in members if 'modernbert' in m['model_name'].lower()]
    for table in pd.read_csv(OUTPUT / 'all_combinations.csv', chunksize=50000, keep_default_na=False):
        table = table[table.model_count.ge(2) & table.all_checkpoints_found]
        for mode in table['mode'].unique():
            subset = table[table['mode'].eq(mode)]
            conditions = {
                'all': np.ones(len(subset), dtype=bool),
                'both_targets': subset.independent_f1.ge(.87) & subset.pair_f1.ge(.94),
                'official3042': subset.official3042_only,
                'no_large': ~subset.members.str.contains('|'.join(large)),
                'with_qwen': subset.members.str.contains('|'.join(qwen)),
                'with_modern': subset.members.str.contains('|'.join(modern)),
                **{f'size{k}': subset.model_count.eq(k) for k in range(2, 7)},
            }
            for category, condition in conditions.items():
                candidates = subset[condition]
                if candidates.empty:
                    continue
                for metric in ['independent_f1', 'pair_f1']:
                    row = candidates.loc[candidates[metric].idxmax()].to_dict()
                    key = (mode, category, metric)
                    if key not in best or (row[metric], -row['model_count']) > (best[key][metric], -best[key]['model_count']):
                        best[key] = row

    choices = [
        ('普通平均：最高结构分', 'probability', 'all', 'pair_f1'),
        ('普通平均：最高独立分', 'probability', 'all', 'independent_f1'),
        ('普通平均：独立≥0.87且结构≥0.94', 'probability', 'both_targets', 'pair_f1'),
        ('普通平均：最佳双模型结构分', 'probability', 'size2', 'pair_f1'),
        ('普通平均：最佳三模型结构分', 'probability', 'size3', 'pair_f1'),
        ('普通平均：官方3042行最高结构分', 'probability', 'official3042', 'pair_f1'),
        ('普通平均：官方3042行最高独立分', 'probability', 'official3042', 'independent_f1'),
        ('普通平均：不用large的最高独立分', 'probability', 'no_large', 'independent_f1'),
        ('普通平均：包含Qwen的最高独立分', 'probability', 'with_qwen', 'independent_f1'),
        ('普通平均：包含ModernBERT的最高独立分', 'probability', 'with_modern', 'independent_f1'),
        ('几何平均：最高结构分', 'log_probability', 'all', 'pair_f1'),
        ('几何平均：两项目标同时达到', 'log_probability', 'both_targets', 'pair_f1'),
        ('几何平均：官方3042行最高结构分', 'log_probability', 'official3042', 'pair_f1'),
        ('分层融合：0.938作为N门', 'hier_0938', 'all', 'pair_f1'),
        ('分层融合：0.9306作为N门', 'hier_0930', 'all', 'pair_f1'),
    ]
    selected, recipe_manifest, bootstrap = [], {}, []
    baseline = decode(search.lp[search.gates['hier_0938']], pairs, 'pair_aware')
    for index, (title, mode, category, metric) in enumerate(choices, 1):
        row = best[(mode, category, metric)]
        code = f'S{index:02d}'
        indices = tuple(by_id[member][0] for member in row['members'].split('+'))
        logits = search.combine(np.array([indices]), mode)[0]
        predictions = {}
        table = dev.copy()
        for i, label in enumerate(LABELS):
            table[f'logit_{label}'] = logits[:, i]
        records = []
        for decoding, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            predicted = decode(logits, pairs, decoding)
            predictions[decoding] = predicted
            values = metrics(gold, predicted, pairs)
            expected = row['independent_f1'] if decoding == 'independent' else row['pair_f1']
            assert abs(values['weighted_f1'] - expected) < 1e-12
            table[column] = np.array(LABELS)[predicted]
            directory = OUTPUT / 'official_checks' / code / decoding
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'predictions.csv'
            pd.DataFrame({'instance_id': dev.instance_id, 'label': np.array(FULL)[predicted]}).to_csv(path, index=False)
            report = score_files(ROOT / 'data/dev/dico_nli_dev_track1_reference.csv', path, output_dir=directory)
            assert abs(report.weighted_f1 - expected) < 1e-12
            assert abs(report.soft_cons - values['soft_cons']) < 1e-12
            assert abs(report.hard_cons - values['hard_cons']) < 1e-12
            records.append({'method': decoding, **values})
        table.to_csv(OUTPUT / f'{code}_predictions.csv', index=False, encoding='utf-8-sig')
        changed = predictions['pair_aware'] != baseline
        changes = table.loc[changed].copy()
        changes['baseline_prediction'] = np.array(LABELS)[baseline[changed]]
        changes.to_csv(OUTPUT / f'{code}_changes_vs_0938.csv', index=False, encoding='utf-8-sig')
        full_indices = sorted(set(indices) | ({search.gates[mode]} if mode.startswith('hier_') else set()))
        row.update({'code': code, 'title': title,
                    'member_names': ' | '.join(members[i]['name'] for i in full_indices),
                    'fixed_vs_0938': int(((baseline != gold) & (predictions['pair_aware'] == gold)).sum()),
                    'broken_vs_0938': int(((baseline == gold) & (predictions['pair_aware'] != gold)).sum())})
        selected.append(row)
        recipe_manifest[code] = {
            'title': title, 'mode': mode, 'direction_members': row['members'].split('+'), 'gate': row['gate'],
            'members': [{k: v for k, v in members[i].items() if k != 'log_probability'} for i in full_indices],
            'same_scores_metrics': records, 'forced_singleton_N': False,
            'gold_used_for_selection': True, 'gold_used_for_per_row_prediction_routing': False,
        }
        bootstrap.append({'code': code, 'baseline': 'original0.938', **group_bootstrap(gold, baseline, predictions['pair_aware'], groups, bootstrap_cfg)})

    result = pd.DataFrame(selected)
    result.to_csv(OUTPUT / 'recommended_combinations.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'selected_recipes.json').write_text(json.dumps(recipe_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(bootstrap).to_csv(OUTPUT / 'conditional_group_bootstrap.csv', index=False, encoding='utf-8-sig')
    curves = []
    for mode in ['probability', 'log_probability', 'hier_0938', 'hier_0930']:
        for size in range(2, 7):
            for objective in ['independent_f1', 'pair_f1']:
                r = best[(mode, f'size{size}', objective)].copy()
                r.update({'objective': objective})
                curves.append(r)
    pd.DataFrame(curves).to_csv(OUTPUT / 'best_per_model_count.csv', index=False, encoding='utf-8-sig')

    text = ['# 现有模型多模型预测比较', '',
            '全部结果来自已有Dev CSV的离线计算；没有训练、加载真实模型或强制单例判N。',
            f'共有{len(members)}份去重后的、与官方660行逐一对齐的四类分数。每个运行只使用匹配其best_summary的一个版本。',
            '1—3个成员穷举；4—6个方向成员采用beam扩展，不保证全局最优。分层方式额外计入N门，实际成员数见表。',
            '模型路径存在只证明文件存在，本次没有加载权重复核。当前ModernBERT结果是盘点时已保存版本。',
            '', '| 编号 | 方案 | 模型数 | independent F1 | pair-aware F1 | N正确 | FBE正确 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in selected:
        text.append(f"| {row['code']} | {row['title']} | {row['model_count']} | {row['independent_f1']:.6f} | {row['pair_f1']:.6f} | {row['pair_N']} | {row['pair_FBE']} |")
    text += ['', '## 各组合的成员', '']
    for row in selected:
        text += [f"### {row['code']} {row['title']}", '', f"模式：{row['mode']}；相对0.938修好{row['fixed_vs_0938']}行、破坏{row['broken_vs_0938']}行。", '']
        for member in recipe_manifest[row['code']]['members']:
            role = 'N门' if member['id'] == row['gate'] else '成员'
            if member['id'] == row['gate'] and member['id'] in row['members'].split('+'):
                role = '方向成员及N门'
            text.append(f"- {member['id']}（{role}）：{member['name']}；训练行数={member['train_rows']}。")
            text.append(f"  预测：`{member['prediction_file']}`")
            text.append(f"  找到的权重：`{' ; '.join(member['checkpoint_candidates_existing'])}`")
        text.append('')
    text += ['## 解释边界', '',
             '普通平均是每个模型四类softmax后等权平均；几何平均是log-prob等权平均。分层融合将N概率和F/B/E条件概率分别组合。',
             '每行表格的独立分和结构分来自同一固定配方、同一份融合分数，没有拼接不同方案的最佳指标。',
             'official3042表示每个成员声明使用官方3042行、未新增训练文本；部分含证据辅助标注、OOF权重或其他训练改动，不等于只用原始四分类监督。',
             '这次进行了大量组合搜索，Dev分数有明显选择偏差，不是新测试集或交叉验证结果。条件bootstrap没有修正搜索偏差，也不包含随机种子变化。',
             '详细成员与权重路径见selected_recipes.json，所有计算过的组合见all_combinations.csv。']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    # 当前run01的详细人工核对说明在重新评分后也保留，不退回只有目录简称的报告。
    from experiments.model_ensemble.expand_report_details import OUTPUT as details_output, expand_report
    if OUTPUT == details_output:
        expand_report()
    print(result[['code', 'title', 'model_count', 'independent_f1', 'pair_f1', 'pair_N', 'pair_FBE']].to_string(index=False))
    print('全部选定结果已通过官方评分器；报告：', OUTPUT / 'RESULTS.md')


if __name__ == '__main__':
    main()
