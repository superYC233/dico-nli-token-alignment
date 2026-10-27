"""对齐0.935参考与120行修订版的曲线和既有探针；不加载模型、不训练。"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_manual_aug.review_events import bootstrap_difference


HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'
CURRENT = HERE / 'outputs/run04_boundary_120'
OUTPUT_DIR = CURRENT / 'measurement_review'
SHORT = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
         'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
LABELS = ['F', 'B', 'E', 'N']


def pair_key(first, second):
    normalize = lambda text: ' '.join(re.findall(r'[^\W_]+', str(text).casefold()))
    return tuple(sorted([normalize(first), normalize(second)]))


def read_run(folder):
    history = pd.read_csv(folder / 'epoch_summary.csv')
    paired = history[history.method.eq('pair_aware')].sort_values('epoch')
    best = paired.sort_values(['weighted_f1', 'epoch'], ascending=[False, True]).iloc[0]
    epoch = int(best.epoch)
    predictions = pd.read_csv(folder / f'epoch_{epoch}/predictions_with_scores.csv', keep_default_na=False)
    assert len(predictions) == 660 and predictions.instance_id.is_unique
    predictions['gold'] = predictions.gold_label.map(SHORT)
    train = pd.read_csv(folder / 'training_data.csv', keep_default_na=False)
    probes = pd.read_csv(folder / 'preserved_diagnostics/independent/predictions.csv', keep_default_na=False)
    return history, paired, predictions.set_index('instance_id'), train, probes.set_index('instance_id'), epoch


def make_figure(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    panels = [('pa_f1', 'Dev pair-aware W-F1'), ('independent_f1', 'Dev independent W-F1'),
              ('N_correct', 'Dev N correct / 106'), ('E_correct', 'Dev E correct / 174')]
    for axis, (metric, title) in zip(axes.flat, panels):
        for prefix, label, color in [('reference', 'Scope 200 reference', '#3768aa'),
                                     ('boundary', 'Reference + boundary 120', '#cf742c')]:
            values = curves[f'{prefix}_{metric}']
            axis.plot(curves.epoch, values, marker='o', markersize=4, color=color, label=label)
        axis.set_title(title)
        axis.grid(alpha=.25)
        axis.set_xticks(range(1, 11))
    axes[0, 0].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel('Epoch')
    figure.suptitle('Same seed and recipe; different data\nEpochs form training trajectories, not independent trials.', fontsize=11)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / 'curves.png', dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_history, old_pa, old, old_train, old_probes, old_epoch = read_run(REFERENCE)
    new_history, new_pa, new, new_train, new_probes, new_epoch = read_run(CURRENT)
    new = new.loc[old.index]
    assert old.gold.equals(new.gold) and old.text1.equals(new.text1) and old.text2.equals(new.text2)
    new_probes = new_probes.loc[old_probes.index]
    for column in ['pair_id', 'text1', 'text2', 'kind', 'family_id', 'proposed_label', 'review_status', 'review_label']:
        assert old_probes[column].equals(new_probes[column]), f'探针输入或建议标签不同：{column}'

    curves = pd.DataFrame({'epoch': old_pa.epoch.to_numpy()})
    descriptive = []
    for name, history, paired in [('reference', old_history, old_pa), ('boundary', new_history, new_pa)]:
        independent = history[history.method.eq('independent')].sort_values('epoch')
        assert paired.epoch.tolist() == independent.epoch.tolist() == list(range(1, 11))
        curves[f'{name}_pa_f1'] = paired.weighted_f1.to_numpy()
        curves[f'{name}_independent_f1'] = independent.weighted_f1.to_numpy()
        for column in ['F_correct', 'B_correct', 'E_correct', 'N_correct', 'FBE_correct']:
            curves[f'{name}_{column}'] = paired[column].to_numpy()
        descriptive.append({
            'run': name, 'min_pa_f1': float(paired.weighted_f1.min()),
            'max_pa_f1': float(paired.weighted_f1.max()),
            'range_percentage_points': float((paired.weighted_f1.max() - paired.weighted_f1.min()) * 100),
            'mean_pa_f1': float(paired.weighted_f1.mean()), 'median_pa_f1': float(paired.weighted_f1.median()),
            'mean_E_correct': float(paired.E_correct.mean()), 'mean_N_correct': float(paired.N_correct.mean()),
            'scope': 'Descriptive trajectory statistics only; epochs are not independent training replicates.',
        })
    curves.to_csv(OUTPUT_DIR / 'curves.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(descriptive).to_csv(OUTPUT_DIR / 'curve_statistics.csv', index=False, encoding='utf-8-sig')

    probe_comparison = old_probes[['pair_id', 'text1', 'text2', 'kind', 'family_id', 'proposed_label', 'review_status']].copy()
    probe_comparison['reference_prediction'] = old_probes.prediction
    probe_comparison['boundary_prediction'] = new_probes.prediction
    probe_comparison['reference_match'] = old_probes.prediction.eq(old_probes.proposed_label)
    probe_comparison['boundary_match'] = new_probes.prediction.eq(new_probes.proposed_label)
    probe_comparison['fixed_proposal'] = ~probe_comparison.reference_match & probe_comparison.boundary_match
    probe_comparison['broken_proposal'] = probe_comparison.reference_match & ~probe_comparison.boundary_match
    probe_comparison.reset_index().to_csv(OUTPUT_DIR / 'probe_comparison.csv', index=False, encoding='utf-8-sig')
    probe_summary = probe_comparison.groupby('kind', sort=False).agg(
        rows=('kind', 'size'), reference_matches=('reference_match', 'sum'),
        boundary_matches=('boundary_match', 'sum'), fixed=('fixed_proposal', 'sum'), broken=('broken_proposal', 'sum'),
    ).reset_index()
    probe_summary['reference_proposal_match_rate'] = probe_summary.reference_matches / probe_summary.rows
    probe_summary['boundary_proposal_match_rate'] = probe_summary.boundary_matches / probe_summary.rows
    probe_summary.to_csv(OUTPUT_DIR / 'probe_summary.csv', index=False, encoding='utf-8-sig')

    comparison = old[['pair_id', 'text1', 'text2', 'gold']].copy()
    comparison['reference_prediction'] = old.pair_prediction
    comparison['event_prediction'] = new.pair_prediction  # 复用已经检查过的配对bootstrap函数。
    before, after = old.pair_prediction.eq(old.gold), new.pair_prediction.eq(new.gold)
    comparison['fixed'] = ~before & after
    comparison['broken'] = before & ~after
    repairs = comparison.groupby('gold')[['fixed', 'broken']].sum().reindex(LABELS)
    repairs.to_csv(OUTPUT_DIR / 'repairs_by_class.csv', encoding='utf-8-sig')
    for flag in ['fixed', 'broken']:
        comparison[comparison[flag]].reset_index().to_csv(OUTPUT_DIR / f'{flag}.csv', index=False, encoding='utf-8-sig')
    bootstrap = bootstrap_difference(comparison)

    for name, frame in [('reference', old), ('boundary', new)]:
        matrix = pd.crosstab(frame.gold, frame.pair_prediction).reindex(index=LABELS, columns=LABELS, fill_value=0)
        matrix.to_csv(OUTPUT_DIR / f'{name}_confusion.csv', encoding='utf-8-sig')

    source_path = CURRENT / 'boundary_examples_source.csv'
    source = pd.read_csv(source_path, keep_default_na=False)
    config = json.loads((CURRENT / 'run_config.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == config['augmentation']['manual_source_sha256']
    assert hashlib.sha256(new_train.to_csv(index=False).encode('utf-8')).hexdigest() == config['augmentation']['effective_train_sha256']
    source_row = source[source.example_id.eq('b_f20')].iloc[0]
    actual_row = new_train[new_train.instance_id.eq('aug_manual_b_f20__0')].iloc[0]
    assert source_row.label == 'B' and actual_row.label == 'BACKWARD_ENTAILMENT'
    source[source.example_id.eq('b_f20')].to_csv(OUTPUT_DIR / 'actual_b_f20.csv', index=False, encoding='utf-8-sig')

    examples = [('of wooden debris', 'of used wooden building materials'),
                ('In Bhutto Death', 'over Bhutto murder'), ('along a city path', 'on the street'),
                ('atop a motorcycle', 'on a motorcycle'), ('Officer', 'Indian policeman')]
    phrase_checks = []
    for split in ['train', 'dev']:
        frame = pd.read_csv(ROOT / f'data/{split}/dico_nli_{split}_track1_participant_labeled.csv', keep_default_na=False)
        keys = [pair_key(a, b) for a, b in zip(frame.text1, frame.text2)]
        for first, second in examples:
            selected = frame[[key == pair_key(first, second) for key in keys]]
            phrase_checks.append({'text1': first, 'text2': second, 'split': split,
                                  'matching_rows': len(selected), 'pair_ids': '|'.join(selected.pair_id.unique())})
    pd.DataFrame(phrase_checks).to_csv(OUTPUT_DIR / 'example_split_check.csv', index=False, encoding='utf-8-sig')

    probe_train_overlap = {}
    for name, train in [('reference', old_train), ('boundary', new_train)]:
        keys = {pair_key(a, b) for a, b in zip(train.text1, train.text2)}
        probe_train_overlap[name] = sum(pair_key(a, b) in keys for a, b in zip(old_probes.text1, old_probes.text2))
    source_pp = source[source.contrast_type.isin(['preposition_restriction', 'preposition_equivalence', 'spatial_difference'])]
    pp_rows = int(source_pp.label.map({'F': 2, 'B': 2, 'E': 2, 'N': 1}).sum())
    priors = {name: float(train.label.eq('EQUIVALENCE').mean()) for name, train in [('reference', old_train), ('boundary', new_train)]}
    facts = {
        'reference_best_epoch': old_epoch, 'boundary_best_epoch': new_epoch,
        'curves': descriptive, 'actual_b_f20_label': source_row.label, 'actual_b_f20_training_label': actual_row.label,
        'E_proportions': priors, 'E_change_percentage_points': (priors['boundary'] - priors['reference']) * 100,
        'preposition_source_contrasts': len(source_pp), 'preposition_expanded_rows': pp_rows,
        'probe_rows': len(probe_comparison), 'probe_train_exact_pair_overlap': probe_train_overlap,
        'reference_probe_proposal_matches': int(probe_comparison.reference_match.sum()),
        'boundary_probe_proposal_matches': int(probe_comparison.boundary_match.sum()),
        'reference_approved_probes': int(old_probes.review_status.eq('approved').sum()),
        'boundary_approved_probes': int(new_probes.review_status.eq('approved').sum()),
        'reference_PA_probe_proposal_matches': int(old_probes.pair_prediction.eq(old_probes.proposed_label).sum()),
        'boundary_PA_probe_proposal_matches': int(new_probes.pair_prediction.eq(new_probes.proposed_label).sum()),
        'reference_PA_E_predictions': int(old.pair_prediction.eq('E').sum()),
        'boundary_PA_E_predictions': int(new.pair_prediction.eq('E').sum()),
        'dev_fixed': int(comparison.fixed.sum()), 'dev_broken': int(comparison.broken.sum()), 'bootstrap': bootstrap,
        'scope': 'Read existing saved predictions only. No model loaded; no training or data edits.',
        'files': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            REFERENCE / 'epoch_summary.csv', CURRENT / 'epoch_summary.csv', source_path,
            REFERENCE / 'preserved_diagnostics/independent/predictions.csv',
            CURRENT / 'preserved_diagnostics/independent/predictions.csv', Path(__file__).resolve(),
        ]},
    }
    (OUTPUT_DIR / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    make_figure(curves)

    lines = [
        '# 修订120行：测量核对与下一步', '',
        '只读取已有曲线、训练快照和探针预测。没有加载模型、重跑推理、修改标签或训练。', '',
        '## 两条完整曲线', '',
        '| epoch | 原scope200 pair-aware | 新120 pair-aware | 原N正确 | 新N正确 | 原E正确 | 新E正确 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in curves.itertuples(index=False):
        lines.append(f'| {row.epoch} | {row.reference_pa_f1:.6f} | {row.boundary_pa_f1:.6f} | {row.reference_N_correct} | {row.boundary_N_correct} | {row.reference_E_correct} | {row.boundary_E_correct} |')
    lines += [
        '', '原版不是未知曲线，也不是稳定在0.93：范围0.827040—0.935493，极差10.8453个百分点。',
        '新120范围0.801611—0.907866，极差10.6255个百分点。',
        '原版跨epoch均值0.875417，新版0.879768；这只是描述训练轨迹，不能把10个epoch当10次独立训练来检验优劣。',
        '两个范围重叠不能证明新版不差；按同一选模协议，新版最佳比原最佳低2.7627个百分点。',
        f'对383组作20000次配对bootstrap，固定这两个已选模型的95%区间为[{bootstrap["ci95_percentage_points"][0]:.4f}, {bootstrap["ci95_percentage_points"][1]:.4f}]个百分点。',
        '区间不含0说明这两个固定模型在当前Dev上的差有证据；不包含训练种子、选模及反复使用Dev的不确定性，不能据此确定增强的普遍因果效果。', '',
        '## 同一200行探针的既有对照', '',
        '已逐条核对ID、文本、组、题型和建议标签相同；两模型训练集均没有这些探针的完整文本对。不是120行训练样本的直接测验。',
        '原probe文件和新probe文件均存在，不需要再加载0.935权重推理。', '',
        '| 类型 | 行数 | 原版独立匹配建议标签 | 新版独立匹配建议标签 | 修好建议 | 破坏建议 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in probe_summary.itertuples(index=False):
        lines.append(f'| {row.kind} | {row.rows} | {row.reference_proposal_match_rate:.2f} | {row.boundary_proposal_match_rate:.2f} | {row.fixed} | {row.broken} |')
    lines += [
        '', '总体独立匹配由152/200到174/200；modifier_scope从18/40到20/40，但其中修好20、破坏18，不能把净+2理解为已经稳定掌握范围。',
        '两边独立审核通过的新探针均为0，因此这些是建议标签匹配率，不是经独立确认的准确率。',
        'pair-aware匹配为原198/200、新200/200。约束只保证合法组合，不保证金标正确；若金标FB而EE联合分数最高，约束仍会答错两行。',
        '这些探针来自Train词汇家族，另有训练/合成背景；完整文本对零交集不使它们成为独立随机测试。', '',
        '## 对所给建议的事实纠正', '',
        '1. b_f20的当次源快照为B，实际训练展开第0行为BACKWARD_ENTAILMENT、第1行为FORWARD_ENTAILMENT，方向正确；源文件哈希与run_config一致。不要误改。',
        '2. 新增介词对照不是0：3个源对照、展开5行，确实偏少。也不是全部名词短语，包含油炸/烹饪、辅导/教学、角色交换等动词结构。风格仍偏规整、部分职业/几何定义与新闻片段距离较大，这是合理的风险假设。',
        f'3. 新增E占40%，但比较应使用完整训练集：882/3242={priors["reference"]:.4%}到930/3362={priors["boundary"]:.4%}，只增加{facts["E_change_percentage_points"]:.4f}个百分点。',
        '4. 该变化不能直接证明“推E导致掉分”。最佳模型的pair-aware预测E数量反而188→177；F/B→E仍为20行，E→F/B由12→24行。需要区分类别比例、预测偏置和语义边界。',
        '5. 正确参考是0.935版F178/B178/E162/N99；新版F178/B178/E150/N93。原建议的177/177/158/98不是这个参考checkpoint。',
        '6. 掉分集中E/N不能排除训练轨迹影响。原版N在53—99、E在128—162间变化；原版F/B则各175—183，波动本来就有类别差异。',
        '7. Officer/Indian policeman在当前官方Train，不在Dev；atop a motorcycle/on a motorcycle这整对未见于当前Train或Dev。其余三组示例在Dev，见example_split_check.csv。', '',
        '## 数据批评中值得保留的部分', '',
        '新模型确实破坏了9个原来正确的E组，含In Bhutto Death/over Bhutto murder、of wooden debris/of used wooden building materials、on the street/along a city path，以及动作改写与新闻事件称呼。',
        '其新增120行对新闻省略、介词角色与同一事件的指称改写覆盖不足。减少长度捷径，并不自动带来同域迁移。',
        '但带介词不代表同一种语义问题：of/in/over的句法角色、中心词近义、同一新闻事件指称和上下文缺失要分开；不能把所有差异归因于介词，也不能把换介词自动标成N。',
        '原版也会错定义改写和词义包含，因此不能说百科式关系完全与Dev无关；是否改善取决于具体子类型和迁移证据。', '',
        '## 现在的执行顺序', '',
        '先停止生成下一批手工数据；保留run04为当前协议下未提升的结果，不删除或改标。',
        '如继续投入一次训练，先做已经准备好的120行复制对照：仍运行train_boundary_manual.py，换新RUN_NAME并设AUGMENT_MODE="copy"。',
        '它匹配3362行、各类新增数量、FB/BF/EE新增组数、211/89步骤预算；用来区别新增内容与更新预算/批次序列的影响。单次对照仍不能证明跨种子稳定。',
        '如果复制同样出现低峰和大幅类别漂移，优先回到训练机制；若复制优于手工，再从Train标签较一致的自然片段出发补少量同域对照。',
        '下一批不应直接全部改成介词：应把自然片段中的范围限制、同义/定义、事件身份、角色变化分别配对，并保留相同骨架下的F/E/N反例。',
        '严格“连功能词也不与Dev重合”的要求仍有效：of/in/on/to等正是禁用词，不能既新增这些词的直接训练又声称词汇零交集。本轮没有擅自放宽。',
        '所有训练仍由你运行。本脚本仅汇总已存在的文件。', '',
        '## 输出', '',
        '`curves.png`为两条曲线及独立F1、N/E正确数；`curves.csv`、`probe_summary.csv`、`probe_comparison.csv`为原始对照。',
        '`fixed.csv`、`broken.csv`、混淆矩阵、`facts.json`记录固定模型差异及输入哈希。',
    ]
    (OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(pd.DataFrame(descriptive).to_string(index=False))
    print(probe_summary.to_string(index=False))
    print('实际b_f20：', source_row.label, '/', actual_row.label)
    print('完整训练E占比变化（百分点）：', facts['E_change_percentage_points'])
    print('报告：', OUTPUT_DIR / 'REPORT.md')
    print('没有加载模型、推理或训练。')


if __name__ == '__main__':
    main()
