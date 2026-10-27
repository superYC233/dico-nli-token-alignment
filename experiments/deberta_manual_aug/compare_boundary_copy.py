"""合看原scope版、人工120和复制120；只读已保存结果，不训练或生成样本。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import f1_score

from experiments.deberta_manual_aug.analyze_boundary_measurements import pair_key, read_run
from experiments.deberta_manual_aug.analyze_current_error_types import NOTES, error_family
from experiments.deberta_manual_aug.review_events import bootstrap_difference


HERE = Path(__file__).resolve().parent
RUNS = {
    'reference': ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope',
    'manual': HERE / 'outputs/run04_boundary_120',
    'copy': HERE / 'outputs/run05_boundary_copy120',
}
NAMES = {'reference': '原scope200', 'manual': '原版+人工120', 'copy': '原版+复制120'}
OUTPUT_DIR = RUNS['copy'] / 'measurement_review'
LABELS = ['F', 'B', 'E', 'N']


def save_table(frame, name, index=False):
    frame.to_csv(OUTPUT_DIR / name, index=index, encoding='utf-8-sig')


def make_figure(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [('pa_f1', 'Dev pair-aware W-F1'), ('independent_f1', 'Dev independent W-F1'),
              ('E_correct', 'E correct / 174'), ('N_correct', 'N correct / 106')]
    for axis, (metric, title) in zip(axes.flat, panels):
        for name, label, color in [('reference', 'Scope 200 reference', '#3768aa'),
                                   ('manual', 'Reference + manual 120', '#cf742c'),
                                   ('copy', 'Reference + copied 120', '#33956b')]:
            axis.plot(curves.epoch, curves[f'{name}_{metric}'], marker='o', markersize=4,
                      color=color, label=label)
        axis.set_title(title)
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel('Epoch')
    figure.suptitle('Saved training trajectories; epochs are not independent trials.', fontsize=11)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / 'three_run_curves.png', dpi=160)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {name: read_run(path) for name, path in RUNS.items()}
    configs = {name: json.loads((path / 'run_config.json').read_text(encoding='utf-8'))
               for name, path in RUNS.items()}
    reference = loaded['reference'][2]
    reference_train = loaded['reference'][3]
    reference_probes = loaded['reference'][4]
    rows = reference[['pair_id', 'text1', 'text2', 'gold']].copy()
    probes = reference_probes[['pair_id', 'text1', 'text2', 'kind', 'family_id',
                               'proposed_label', 'review_status', 'review_label']].copy()
    probe_input_columns = probes.columns.tolist()
    curves = pd.DataFrame({'epoch': range(1, 11)})
    metrics, curve_stats, checks = [], [], {}

    for name, (history, paired, predictions, train, run_probes, epoch) in loaded.items():
        predictions = predictions.loc[reference.index]
        run_probes = run_probes.loc[reference_probes.index]
        for column in ['pair_id', 'text1', 'text2', 'gold']:
            assert predictions[column].equals(reference[column]), f'{name} Dev未对齐：{column}'
        for column in probe_input_columns:
            assert run_probes[column].equals(reference_probes[column]), f'{name} 探针不同：{column}'
        assert train.iloc[:len(reference_train)].reset_index(drop=True).equals(reference_train)
        canonical_hash = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
        assert canonical_hash == configs[name]['augmentation']['effective_train_sha256']

        rows[f'{name}_prediction'] = predictions.pair_prediction
        rows[f'{name}_wrong'] = predictions.pair_prediction.ne(predictions.gold)
        rows[f'{name}_error_family'] = [error_family(g, p) for g, p in
                                       zip(predictions.gold, predictions.pair_prediction)]
        probes[f'{name}_prediction'] = run_probes.prediction
        probes[f'{name}_match'] = run_probes.prediction.eq(run_probes.proposed_label)
        independent = history[history.method.eq('independent')].sort_values('epoch')
        assert paired.epoch.tolist() == independent.epoch.tolist() == list(range(1, 11))
        curves[f'{name}_pa_f1'] = paired.weighted_f1.to_numpy()
        curves[f'{name}_independent_f1'] = independent.weighted_f1.to_numpy()
        for label in LABELS:
            curves[f'{name}_{label}_correct'] = paired[f'{label}_correct'].to_numpy()

        selected = paired[paired.epoch.eq(epoch)].iloc[0]
        selected_ind = independent[independent.epoch.eq(epoch)].iloc[0]
        pa_f1 = f1_score(predictions.gold, predictions.pair_prediction, average='weighted')
        ind_f1 = f1_score(predictions.gold, predictions.prediction, average='weighted')
        assert abs(pa_f1 - selected.weighted_f1) < 1e-12
        assert abs(ind_f1 - selected_ind.weighted_f1) < 1e-12
        record = {'run': name, 'best_epoch': epoch, 'train_rows': len(train),
                  'independent_f1': ind_f1, 'pair_aware_f1': pa_f1,
                  'error_rows': int(rows[f'{name}_wrong'].sum()),
                  'error_groups': rows.loc[rows[f'{name}_wrong'], 'pair_id'].nunique()}
        for label in LABELS:
            correct = int((predictions.gold.eq(label) & predictions.pair_prediction.eq(label)).sum())
            assert correct == selected[f'{label}_correct']
            record[f'{label}_correct'] = correct
        metrics.append(record)
        curve_stats.append({'run': name, 'mean_pa_f1': paired.weighted_f1.mean(),
                            'median_pa_f1': paired.weighted_f1.median(),
                            'min_pa_f1': paired.weighted_f1.min(), 'max_pa_f1': paired.weighted_f1.max(),
                            'range_percentage_points': (paired.weighted_f1.max() - paired.weighted_f1.min()) * 100})
        train_pairs = {pair_key(a, b) for a, b in zip(train.text1, train.text2)}
        checks[name] = {'training_canonical_sha256': canonical_hash,
                        'reference_prefix_equal': True,
                        'probe_exact_unordered_train_pair_overlaps': sum(
                            pair_key(a, b) in train_pairs for a, b in zip(run_probes.text1, run_probes.text2)),
                        'probe_proposal_matches': int(probes[f'{name}_match'].sum()),
                        'approved_probe_rows': int(run_probes.review_status.eq('approved').sum())}
        matrix = pd.crosstab(predictions.gold, predictions.pair_prediction).reindex(
            index=LABELS, columns=LABELS, fill_value=0)
        save_table(matrix, f'{name}_confusion.csv', index=True)

    # 两个120行方案的差异应限于数据内容及其来源记录，而非训练参数。
    allowed_differences = {'run_name', 'augmentation', 'source_sha256'}
    config_differences = [key for key in configs['manual'] if configs['manual'][key] != configs['copy'][key]]
    assert set(config_differences) <= allowed_differences
    extra_manual = pd.read_csv(RUNS['manual'] / 'extra_train_rows.csv', keep_default_na=False)
    extra_copy = pd.read_csv(RUNS['copy'] / 'extra_train_rows.csv', keep_default_na=False)
    assert len(extra_manual) == len(extra_copy) == 120
    assert extra_manual.label.tolist() == extra_copy.label.tolist()
    pair_shapes = lambda frame: [tuple(group.label) for _, group in frame.groupby('pair_id', sort=False)]
    assert pair_shapes(extra_manual) == pair_shapes(extra_copy)
    original = pd.read_csv(ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv', keep_default_na=False)
    original_triples = set(zip(original.text1, original.text2, original.label))
    assert all(triple in original_triples for triple in zip(extra_copy.text1, extra_copy.text2, extra_copy.label))
    assert loaded['copy'][3].iloc[-120:].reset_index(drop=True).equals(extra_copy)
    assert loaded['manual'][3].iloc[-120:].reset_index(drop=True).equals(extra_manual)

    # 沿用此前逐组复核。主类互斥；PP是可以重叠的句法框架标记，不能解释为已证实原因。
    metadata = pd.DataFrame.from_dict({f'dico_phrasis_{key}': value for key, value in NOTES.items()}, orient='index')
    union_mask = rows[[f'{name}_wrong' for name in RUNS]].any(axis=1)
    errors = rows[union_mask].join(metadata, on='pair_id')
    assert errors.primary_type.notna().all(), '出现尚未逐组复核的新错误。'
    type_summary = pd.DataFrame({name: errors.groupby('primary_type')[f'{name}_wrong'].sum()
                                 for name in RUNS}).sort_values(['copy', 'manual'], ascending=False)
    type_summary['manual_minus_copy_errors'] = type_summary.manual - type_summary['copy']
    family_summary = pd.DataFrame({name: rows.loc[rows[f'{name}_wrong'], f'{name}_error_family'].value_counts()
                                   for name in RUNS}).fillna(0).astype(int)
    pp_counts = {}
    for name in RUNS:
        wrong = errors[errors[f'{name}_wrong']]
        pp_counts[name] = {'rows': int(wrong.has_PP_frame.sum()),
                            'groups': wrong.loc[wrong.has_PP_frame, 'pair_id'].nunique()}
        save_table(wrong.reset_index(), f'{name}_error_rows.csv')
    summary_columns = ['pair_id', 'text1', 'text2', 'gold', 'primary_type', 'has_PP_frame',
                       'context_review_needed', 'possible_label_or_direction_issue',
                       'relation_to_review', 'augmentation_direction']
    error_groups = errors.reset_index().groupby('pair_id', sort=False)[
        [column for column in summary_columns if column != 'pair_id'] +
        [f'{name}_prediction' for name in RUNS]].first()
    for name in RUNS:
        error_groups[f'{name}_error_rows'] = errors.groupby('pair_id')[f'{name}_wrong'].sum()
    save_table(error_groups, 'error_groups_all_three.csv', index=True)
    save_table(error_groups[error_groups.copy_error_rows.gt(0)], 'copy_error_groups.csv', index=True)
    save_table(type_summary, 'error_type_summary.csv', index=True)
    save_table(family_summary, 'label_confusion_summary.csv', index=True)

    probe_summary = probes.groupby('kind', sort=False).agg(
        rows=('kind', 'size'), reference=('reference_match', 'sum'),
        manual=('manual_match', 'sum'), copy=('copy_match', 'sum'))
    save_table(probe_summary, 'provisional_probe_summary.csv', index=True)
    save_table(probes.reset_index(), 'provisional_probe_predictions.csv')
    save_table(curves, 'three_run_curves.csv')
    save_table(pd.DataFrame(curve_stats), 'curve_statistics.csv')
    save_table(pd.DataFrame(metrics), 'best_models.csv')

    comparisons = {}
    for before, after in [('reference', 'manual'), ('reference', 'copy'), ('manual', 'copy')]:
        name = f'{after}_minus_{before}'
        frame = rows[['pair_id', 'text1', 'text2', 'gold']].copy()
        frame['reference_prediction'] = rows[f'{before}_prediction']
        frame['event_prediction'] = rows[f'{after}_prediction']
        frame['fixed'] = rows[f'{before}_wrong'] & ~rows[f'{after}_wrong']
        frame['broken'] = ~rows[f'{before}_wrong'] & rows[f'{after}_wrong']
        frame['changed'] = frame.reference_prediction.ne(frame.event_prediction)
        frame['changed_wrong'] = frame.changed & rows[f'{before}_wrong'] & rows[f'{after}_wrong']
        assert frame.changed.sum() == frame[['fixed', 'broken', 'changed_wrong']].sum().sum()
        changed = frame[frame.changed].join(metadata, on='pair_id')
        save_table(changed.reset_index(), f'{name}_changed.csv')
        save_table(changed.groupby('primary_type')[['fixed', 'broken', 'changed_wrong']].sum(),
                   f'{name}_changes_by_type.csv', index=True)
        comparisons[name] = {key: int(frame[key].sum()) for key in ['changed', 'fixed', 'broken', 'changed_wrong']}
        comparisons[name]['bootstrap'] = bootstrap_difference(frame)

    sources = [Path(__file__).resolve()]
    for path in RUNS.values():
        sources += [path / 'run_config.json', path / 'training_data.csv', path / 'epoch_summary.csv',
                    path / 'preserved_diagnostics/independent/predictions.csv']
        name = next(key for key, value in RUNS.items() if value == path)
        sources.append(path / f'epoch_{loaded[name][5]}/predictions_with_scores.csv')
    facts = {'metrics': metrics, 'curve_statistics': curve_stats, 'PP_frame_errors': pp_counts,
             'comparisons': comparisons, 'checks': checks,
             'manual_copy_config_differences': config_differences,
             'manual_copy_same_labels_per_row': True, 'manual_copy_same_pair_label_sequences': True,
             'copy_extra_all_text_and_labels_from_official_train': True,
             'copy_above_manual_epochs': int((curves.copy_pa_f1 > curves.manual_pa_f1).sum()),
             'scope': 'Saved predictions only; no training, inference, annotation or augmentation changes.',
             'review_limit': 'Semantic types are assistant judgement, not gold causal findings. Probe labels are provisional.',
             'file_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (OUTPUT_DIR / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    make_figure(curves)

    lines = ['# 原scope、人工120、复制120：合并分析', '',
             '只读三个运行的训练快照、已有Dev预测和诊断预测；没有训练、修改原结果或生成新增样本。', '',
             '## 结论', '',
             '本次人工120未优于同预算复制对照；保留0.935原scope版作为当前最终分数参照。',
             '不沿用“下一批全造介词短语”的推断。主要问题是E与单向蕴含的边界，带介词只是一个重叠的表面特征。', '',
             '| 运行 | 最佳epoch | 独立W-F1 | pair-aware W-F1 | F/B/E/N正确 | 错误行/组 |',
             '|---|---:|---:|---:|---|---|']
    for item in metrics:
        counts = '/'.join(str(item[f'{label}_correct']) for label in LABELS)
        lines.append(f'| {NAMES[item["run"]]} | {item["best_epoch"]} | {item["independent_f1"]:.6f} | '
                     f'{item["pair_aware_f1"]:.6f} | {counts} | {item["error_rows"]}/{item["error_groups"]} |')
    lines += ['', '## 复制对照说明了什么', '',
              '人工与复制两版使用同一3242行前缀，总行数3362；新增顺序逐行标签相同，配对形状相同，均为F24/B24/E48/N24。',
              '两版核心训练配置一致，均为每轮211个Row步和89个Pair步。复制120的文本与标签全部来自官方Train。',
              '人工版相对复制版低1.5321个百分点；复制版相对人工版修好12行、破坏2行，E多对8行，N多对2行。',
              '所以不能单凭“E占新增40%”或“步数变多”解释二者差距；样本内容、样式、选取的原样本权重及梯度路径仍可能有影响。',
              '这是当前seed和选模协议下的结果，不是人工增强普遍无效的证明。', '',
              '## 最多的错误', '',
              '以下主类为助手逐组归类，互斥；是待核验的关系分析，不是独立人工金标或已证明的错误原因。', '',
              '| 主类 | 原scope | 人工120 | 复制120 |',
              '|---|---:|---:|---:|']
    for kind, row in type_summary.iterrows():
        lines.append(f'| {kind} | {row.reference} | {row.manual} | {row["copy"]} |')
    lines += ['', '| 标签混淆 | 原scope | 人工120 | 复制120 |', '|---|---:|---:|---:|']
    for kind, row in family_summary.iterrows():
        lines.append(f'| {kind} | {row.reference} | {row.manual} | {row["copy"]} |')
    lines += ['', '原scope的E↔F/B为32/43行，人工版44/61行，复制版36/51行。',
              '带PP框架的错误分别为21、33、25行；这不是纯介词语义错误数量，不能与上面的主类相加。',
              'of/from/runway/airport同时涉及中心词和关系；over police crackdown/over police violence的介词相同。',
              'from a bird feeder/from the feeder、at the station/at the train station都是同一介词下的限定范围问题。',
              'May Housing/Housing in May是修饰结构改写；convicted/get jail time在这两个120版本均已答对。', '',
              '## 人工相对复制的具体得失', '',
              '人工版额外答错：car/vehicle、bribing/pay off、slide up/zipping up、crackdown/violence、debris/building materials等五个配对组，以及shore/beach、into/at camera两个N单例。',
              '人工版额外答对：Venezuela Government, Opposition Head/Venezuelan opposition leader这一组。',
              '这些是差异清单，不能从短语表面直接确定每一条的根因或构造通用金标。', '',
              '## 曲线与探针', '',
              '三个运行的PA极差都约10个百分点。复制版8/10个epoch高于人工版；epoch相互相关，不能当作8次独立胜出。',
              '跨epoch平均PA：原scope0.875417，人工0.879768，复制0.893590；只是轨迹描述。',
              '相同200条探针的独立建议标签匹配数：原152、人工174、复制188。',
              'modifier_scope分别为18/40、20/40、29/40；因此不能将人工版的探针提升直接归于新增语义知识。',
              '探针均未独立人工批准，且源自Train家族；这些是建议标签匹配率，不是独立测试准确率。', '',
              '## 固定已选模型的配对bootstrap', '',
              '按383个pair_id整组重采样20000次；以下单位为W-F1百分点。',
              '| 差值 | 观测差 | 95%区间 |', '|---|---:|---|']
    for name, result in comparisons.items():
        boot = result['bootstrap']
        lo, hi = boot['ci95_percentage_points']
        lines.append(f'| {name} | {boot["delta_percentage_points"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', '此区间不包含训练种子、最佳epoch选择和反复使用Dev带来的不确定性。', '',
              '## 是否再造数据', '',
              '暂不扩充当前百科式120，也不把所有含介词的错误改写成新训练样本。',
              '若做下一批，只做小批自然短语试点，优先两种可清楚标注的关系：同一关系中的真正限定F/B，以及不增信息的结构改写E；以角色/目标变化N作反例。',
              '具体模板和标签应先从Train中的同类用法确认；需要恢复新闻上下文的错误先查来源，不强制编出普遍E/N规则。',
              '新造词形、词干、短语与Dev隔离的要求继续保留，包括介词；不能为了贴近Dev而偷偷放开in/on/of等功能词。',
              '若该限制使Train中的自然模板无法成立，就先停止该模板，不以生硬的同义词替换补齐数量。',
              '研究方向来自Dev错误分析须披露；文本零重合不代表独立于Dev开发。',
              '本次只分析，没有创建下一批训练数据，也没有改动训练RUN_NAME或AUGMENT_MODE。', '']
    (OUTPUT_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print('\n按关系复核的错误行数：\n', type_summary.to_string())
    print('\n配对bootstrap：\n', json.dumps(comparisons, ensure_ascii=False, indent=2))
    print('\n分析报告：', OUTPUT_DIR / 'REPORT.md')


if __name__ == '__main__':
    main()
