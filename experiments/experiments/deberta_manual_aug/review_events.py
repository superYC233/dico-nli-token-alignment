"""检查事件增强的覆盖与捷径，只读已有数据和预测，另存分析报告；不训练。"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / 'event_review'
REFERENCE_DIR = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'
EVENT_DIR = HERE / 'outputs/run03_event_200'
SHORT = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
         'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
LABELS = ['F', 'B', 'E', 'N']

# 这是对判别模式的人工分析，不是重新给官方样本打标签。
# “部分”表示只练习了某个子操作，不表示该Dev例子一定能由它修好。
# “未覆盖”也不表示不会发生跨模式迁移。
COVERAGE = {
    '0000034': ('未覆盖', '实体职务与新闻片段', '没有实体指称、职务和省略成分对照；先核对上下文。'),
    '0000198': ('未覆盖', '名词词汇包含', '宾语删除不是两个不同中心名词之间的包含；需要独立的新词汇层级对照。'),
    '0000375': ('部分', '不同谓词的动作粒度', '新增仅练习V+宾语到V，缺少具体谓词到一般谓词的同词数对照。'),
    '0000432': ('待核验', '修饰范围与方向疑点', '不能从black top和necklace推出big necklace；不据此编造相同方向规则。'),
    '0000497': ('未覆盖', '地点、部分整体与介词', '动作宾语和事件名词不覆盖of/from的关系；需要源上下文。'),
    '0000499': ('未覆盖', '外观改写与异常片段', '没有外观属性、对象称呼和非标准短语对照；不能仅按词数解释。'),
    '0000616': ('部分', '修饰限制、对象类型与片段', 'event_scope只练习去掉事件限定；没有TV/screen一类对象类型及of片段。'),
    '0000729': ('未覆盖', '定义性等价', '副词换位和两个状态词的替换都不覆盖短词到多词定义的等价。'),
    '0000817': ('未覆盖', '场景地点的指称等价', '没有场景改写；字面普遍等价不明确，需核对源上下文。'),
    '0000948': ('未覆盖', '同一新闻事件的语境等价', '两个不同事件谓词的关系没有训练；不能将crackdown与violence普遍等同。'),
    '0000979': ('未覆盖', '介词角色与源对齐', '没有after/to之类角色变化；金标的语境依据需要核验。'),
    '0001103': ('未覆盖', '颜色近义与指称约定', '没有颜色、近义精度和类别称谓对照；不把所有近似都标为E。'),
    '0001156': ('未覆盖', '属性描述与场景名词', '没有属性描述与地点名词之间的改写；普遍等价并不明确。'),
    '0001456': ('部分', '运动与一般活动', 'V+宾语删除不能直接教会具体运动谓词到一般活动谓词；playing的词义依赖语境。'),
    '0001551': ('待核验', '集合方向疑点', '通常集合解释与官方B方向有疑问；保留官方标签，不能由它反推一般规则。'),
    '0001567': ('未覆盖', '事件、统计与新闻省略', '同一事件的计划/完成不覆盖事故与伤亡统计的关系；需原新闻语境。'),
    '0001781': ('部分', '地点类别的限定', '去掉限定有相似子操作，但没有PP框架、站点词义和不同长度的E反例。'),
    '0001940': ('部分', '活动粒度与事件表达', '删除宾语有部分关联；没有两种不同动作表达的包含，且需固定事件/体貌解释。'),
    '0002242': ('未覆盖', '多义谓词与开始状态', '计划/完成的固定替换没有解决Opens的词义与论元缺失；多义本身不等于N。'),
    '0002463': ('未覆盖', '名词化与动词、相关不等价', '新增两侧均为事件名词的状态对照，没有名词与动词表达的关系。'),
    '0002492': ('未覆盖', '等级阈值与最高级', '没有程度、比较集合和最高级对照。'),
    '0002616': ('未覆盖', '路径与终点介词', '没有路径、接触/到达和目标角色对照；片段仍需语境核验。'),
    '0002738': ('部分', '不同事件的后续关系', '计划/完成是同一事件的状态；缺少两个不同事件之间有关但不能必然互推的对照。'),
    '0002900': ('待核验', '职务上下位与标签约定', '通常senator属于lawmaker；不能为了修一条Dev而否认一般包含关系。'),
    '0002933': ('待核验', '地域、机构职务与称谓', '需要核对地域表达、MP词义和源标注依据；没有相应增强。'),
    # 本轮相对0.935版新增的12个错误组，也纳入覆盖复核。
    '0000018': ('未覆盖', '新闻事件、名词与介词', '没有新闻短语省略与事件名词关系；无法从副词换位外推。'),
    '0000125': ('未覆盖', '地点属性与介词', '没有地点属性与路径/位置角色对照；需语境核验。'),
    '0000221': ('未覆盖', '地点名词与场景指称', '没有field/yard式地点名词关系；不能直接照金标推广词义。'),
    '0000808': ('未覆盖', '定义性等价、长而不增信息', '所有新增F/B都与长度方向一致；缺少这种不同长度的定义性E。'),
    '0000825': ('未覆盖', '介词短语与场景改写', '没有路径表达与复合形容词的改写；副词换位不足以覆盖。'),
    '0000939': ('未覆盖', '施事片段与事件省略', '新增角色N交换两个人，没有教会事件短语与施事片段何时可等价。'),
    '0000980': ('未覆盖', '同一新闻事件的不同称呼', '同一对象/事件的指称等价依赖语境；不是计划与完成的区别。'),
    '0000994': ('未覆盖', '时间修饰与名词/PP改写', '副词前后换位没有覆盖名词修饰与介词结构之间的改写。'),
    '0001136': ('未覆盖', '地点名词与路径角色', '没有street/path及on/along的场景与角色对照。'),
    '0002244': ('未覆盖', '数量与中心对象缺失', '有数字不等于保留了对象类型；当前没有数量、单位与中心名词的对照。'),
    '0002251': ('未覆盖', '人、职务与组织类型', '两个人互换角色不是人与组织的类型差异；需要实体类型对照。'),
    '0003283': ('未覆盖', '词性、片段与指称范围', '没有名词短语与修饰性片段的关系；不能仅用长短决定包含。'),
}

SOURCE_REVIEW = {
    'action_specificity': ('可保留少量', '标签在同一动作、通常字面义下较清楚；但20组全是删宾语，长度即可判断方向。',
                           '补不同谓词、相同词数的包含；同一宾语也要出现在E/N对照里。'),
    'event_scope': ('优先重写', '完成/计划+事件名词的解释依赖同一事件范围；有些搭配生硬，不能当无条件逻辑模板。',
                    '逐条固定可数/有界事件解释；减少状态前缀，保留自然、范围清楚的少量对照。'),
    'manner_equivalence': ('减少重复', '通常标签可成立，但只是副词移位；40行均为等长、同词集合的E，未练习真正的词义改写。',
                           '换成不同词汇或句法的语义等价；必须同时出现不同长度的E。'),
    'event_state_equivalence': ('减少重复并核验搭配', '40行只重复两组状态词；完成状态需要有界事件，部分名词组合像模板拼接。',
                                '保留少量自然表达；大部分预算用于定义改写和其他动作同义表达。'),
    'action_object_change': ('可保留', '不同对象通常不构成互相蕴含，N不要求二者不能同时发生；仍需逐条检查对象包含和词义。',
                            '补同对象、换谓词的N，以及词面更像但类型不同的N，防止只看对象词。'),
    'participant_roles': ('可保留但补反例', '角色交换通常不保证互推；不能把可同时成立误当标签错误，也不能把所有词序变化都当N。',
                          '保留非对称谓词的明确施受事；另补角色不变但句法变化的E，检查consult的互惠读法。'),
    'planned_vs_completed': ('可保留少量', '计划/排期与完成通常不能必然互推，但20行实际只有两组状态替换和固定方向。',
                             '补不同事件谓词间的相关非蕴含；同时审核有真实蕴含的阶段关系，避免见状态差异就答N。'),
}


def words(text):
    return re.findall(r'[^\W_]+', str(text).casefold())


def surface_rule(first, second):
    """刻意构造的捷径证明；不训练、不作为正式预测方法。"""
    left, right = words(first), words(second)
    if len(left) > len(right):
        return 'F'
    if len(left) < len(right):
        return 'B'
    if Counter(left) == Counter(right):
        return 'E' if len(left) <= 3 else 'N'
    if {left[0], right[0]} in [{'completed', 'finished'}, {'scheduled', 'timetabled'}]:
        return 'E'
    return 'N'


def read_prediction(path):
    frame = pd.read_csv(path, keep_default_na=False)
    assert frame.instance_id.is_unique and len(frame) == 660
    frame['gold'] = frame.gold_label.map(SHORT)
    return frame


def best_run(folder):
    summary = pd.read_csv(folder / 'epoch_summary.csv')
    best = summary[summary.method.eq('pair_aware')].sort_values(
        ['weighted_f1', 'epoch'], ascending=[False, True],
    ).iloc[0]
    epoch = int(best.epoch)
    path = folder / f'epoch_{epoch}/predictions_with_scores.csv'
    return read_prediction(path), epoch, int(summary.epoch.max()), path


def error_family(gold, prediction):
    if gold == prediction:
        return 'correct'
    if gold == 'N':
        return 'N_to_E' if prediction == 'E' else 'N_to_FB'
    if gold == 'E':
        return 'E_to_FB'
    if prediction == 'E':
        return 'FB_to_E'
    return 'F_B_reversal'


def bootstrap_difference(frame):
    # 同一pair_id整体重采样；区间仅针对这两个已选定模型和当前Dev。
    groups = list(frame.groupby('pair_id', sort=False))
    matrices = []
    for column in ['reference_prediction', 'event_prediction']:
        counts = np.zeros((len(groups), 16), dtype=np.int64)
        for index, (_, group) in enumerate(groups):
            for gold, prediction in zip(group.gold, group[column]):
                counts[index, LABELS.index(gold) * 4 + LABELS.index(prediction)] += 1
        matrices.append(counts)

    def weighted_f1(counts):
        counts = counts.reshape(-1, 4, 4)
        support = counts.sum(axis=2)
        predicted = counts.sum(axis=1)
        numerator = 2 * counts.diagonal(axis1=1, axis2=2)
        denominator = support + predicted
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
        return (support * f1).sum(axis=1) / support.sum(axis=1)

    rng = np.random.default_rng(20261022)
    differences = []
    for _ in range(20):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=1000)
        differences.extend((weighted_f1(weights @ matrices[1]) - weighted_f1(weights @ matrices[0])) * 100)
    interval = np.quantile(differences, [.025, .975])
    return {'resamples': 20000, 'groups': len(groups), 'seed': 20261022,
            'delta_percentage_points': (weighted_f1(matrices[1].sum(axis=0))[0]
                                        - weighted_f1(matrices[0].sum(axis=0))[0]) * 100,
            'ci95_percentage_points': interval.tolist(),
            'scope': 'Fixed selected models/current Dev only; excludes training-seed and checkpoint-selection uncertainty.'}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = HERE / 'event_examples.csv'
    notes_path = HERE / 'prepared_events/manual_label_notes.csv'
    source = pd.read_csv(source_path, keep_default_na=False)
    notes = pd.read_csv(notes_path, keep_default_na=False)
    original = pd.read_csv(ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv', keep_default_na=False)
    dev = pd.read_csv(ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv', keep_default_na=False)
    full = pd.read_csv(HERE / 'prepared_events/training_data_manual.csv', keep_default_na=False)
    reference_path = REFERENCE_DIR / 'epoch_3/predictions_with_scores.csv'
    reference = read_prediction(reference_path).set_index('instance_id')
    current, epoch, observed_epochs, event_path = best_run(EVENT_DIR)
    current = current.set_index('instance_id').loc[reference.index]
    assert current.gold.equals(reference.gold)
    assert current.text1.equals(reference.text1) and current.text2.equals(reference.text2)
    saved_data = pd.read_csv(EVENT_DIR / 'extra_train_rows.csv', keep_default_na=False)
    prepared_data = pd.read_csv(HERE / 'prepared_events/manual_rows.csv', keep_default_na=False)
    assert saved_data.equals(prepared_data), '本报告的200行必须与该训练的实际快照一致。'

    comparison = reference[['pair_id', 'text1', 'text2', 'gold']].copy()
    comparison['reference_prediction'] = reference.pair_prediction
    comparison['event_prediction'] = current.pair_prediction
    before = comparison.reference_prediction.eq(comparison.gold)
    after = comparison.event_prediction.eq(comparison.gold)
    comparison['fixed'] = ~before & after
    comparison['broken'] = before & ~after
    comparison['changed'] = comparison.reference_prediction.ne(comparison.event_prediction)
    comparison['changed_wrong'] = comparison.changed & ~before & ~after
    comparison['reference_error_family'] = [error_family(g, p) for g, p in zip(comparison.gold, comparison.reference_prediction)]
    comparison['event_error_family'] = [error_family(g, p) for g, p in zip(comparison.gold, comparison.event_prediction)]
    comparison.reset_index().to_csv(OUTPUT_DIR / 'prediction_comparison.csv', index=False, encoding='utf-8-sig')
    for flag in ['fixed', 'broken']:
        comparison[comparison[flag]].reset_index().to_csv(OUTPUT_DIR / f'{flag}.csv', index=False, encoding='utf-8-sig')

    # 覆盖所有旧错误及本次新增错误，不能只挑修好的例子。
    coverage_rows = []
    for pair_id, group in comparison.groupby('pair_id', sort=False):
        old_errors = group.reference_prediction.ne(group.gold)
        new_errors = group.event_prediction.ne(group.gold)
        if not (old_errors.any() or new_errors.any()):
            continue
        coverage, mechanism, issue = COVERAGE[pair_id.rsplit('_', 1)[-1]]
        row = group.iloc[0]
        coverage_rows.append({
            'pair_id': pair_id, 'text1': row.text1, 'text2': row.text2,
            'gold': row.gold, 'reference_prediction': row.reference_prediction,
            'event_prediction': row.event_prediction,
            'reference_wrong_rows': int(old_errors.sum()), 'event_wrong_rows': int(new_errors.sum()),
            'fixed_rows': int(group.fixed.sum()), 'broken_rows': int(group.broken.sum()),
            'coverage_assessment': coverage, 'mechanism_to_check': mechanism,
            'coverage_gap_or_review_issue': issue,
            'assessment_status': 'assistant hypothesis; not independent gold adjudication',
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT_DIR / 'error_coverage.csv', index=False, encoding='utf-8-sig')

    review_rows = []
    for row in source.itertuples(index=False):
        decision, issue, direction = SOURCE_REVIEW[row.contrast_type]
        extra = ''
        if row.family in {'consume_food', 'drink_beverages', 'assist_people', 'supervise_workers', 'mentor_professionals'} and row.contrast_type in {'event_scope', 'event_state_equivalence', 'planned_vs_completed'}:
            extra = '此处的consumption/assistance/supervision/mentoring等尤其需要明确一次事件的边界和完成标准。'
        if row.family == 'consult_specialists' and row.contrast_type == 'participant_roles':
            extra = 'consult可能被理解为寻求意见或相互商议；优先换成施受事方向更明确的谓词。'
        review_rows.append({
            'example_id': row.example_id, 'family': row.family, 'contrast_type': row.contrast_type,
            'text1': row.text1, 'text2': row.text2, 'label': row.label,
            'review_decision': decision, 'issue': issue + extra, 'revision_direction': direction,
            'review_status': 'template-level audit with family notes; not independent adjudication',
        })
    pd.DataFrame(review_rows).to_csv(OUTPUT_DIR / 'source_review.csv', index=False, encoding='utf-8-sig')

    # 检查相同有序文本对出现不同标签；区分既有问题与新增引入的问题。
    full['normalized_text1'] = full.text1.map(lambda text: ' '.join(words(text)))
    full['normalized_text2'] = full.text2.map(lambda text: ' '.join(words(text)))
    keys = ['normalized_text1', 'normalized_text2']
    conflict = full.groupby(keys).label.transform('nunique').gt(1)
    full[conflict].to_csv(OUTPUT_DIR / 'existing_label_conflicts.csv', index=False, encoding='utf-8-sig')

    vocabulary = lambda frame: {word for column in ['text1', 'text2'] for text in frame[column] for word in words(text)}
    new_words, train_words = vocabulary(notes), vocabulary(original)
    tokens = [word for column in ['text1', 'text2'] for text in notes[column] for word in words(text)]
    shortcut = {}
    for name, frame, gold in [('new_200', notes, notes.target), ('official_train', original, original.label.map(SHORT)),
                              ('dev', dev, dev.label.map(SHORT))]:
        predictions = [surface_rule(a, b) for a, b in zip(frame.text1, frame.text2)]
        shortcut[name] = {'rows': len(frame), 'correct': int((np.asarray(predictions) == gold.to_numpy()).sum()),
                          'accuracy': accuracy_score(gold, predictions),
                          'weighted_f1': f1_score(gold, predictions, average='weighted')}

    metrics = []
    for name, folder in [('reference_scope_200', REFERENCE_DIR),
                         ('manual_80', HERE / 'outputs/run01_original_manual'),
                         ('copy_80', HERE / 'outputs/run02_original_copy'), ('event_200', EVENT_DIR)]:
        frame, best_epoch, observed, path = best_run(folder)
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            correct = frame[column].eq(frame.gold)
            metrics.append({'run': name, 'best_pair_epoch': best_epoch, 'observed_epochs': observed,
                            'method': method, 'weighted_f1': f1_score(frame.gold, frame[column], average='weighted'),
                            'accuracy': correct.mean(), 'N_correct': int((correct & frame.gold.eq('N')).sum()),
                            'FBE_correct': int((correct & frame.gold.ne('N')).sum()),
                            'selection': 'same checkpoint selected by Dev pair-aware W-F1',
                            'prediction_source': str(path.resolve())})
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / 'run_metrics.csv', index=False, encoding='utf-8-sig')

    old_coverage = coverage[coverage.reference_wrong_rows.gt(0)]
    counts = old_coverage.groupby('coverage_assessment').agg(groups=('pair_id', 'size'), rows=('reference_wrong_rows', 'sum'))
    bootstrap = bootstrap_difference(comparison)
    states = {'planned', 'scheduled', 'timetabled', 'completed', 'finished'}
    statistics = {
        'new_rows': len(notes), 'source_contrasts': len(source), 'lexical_families': int(source.family.nunique()),
        'template_types': int(source.contrast_type.nunique()),
        'rows_by_type': notes.groupby('contrast_type').size().to_dict(),
        'state_word_rows': sum(bool(states & set(words(a) + words(b))) for a, b in zip(notes.text1, notes.text2)),
        'new_unique_words': len(new_words), 'words_not_in_official_train': len(new_words - train_words),
        'token_fraction_not_in_official_train': sum(word not in train_words for word in tokens) / len(tokens),
        'phrase_word_counts': dict(Counter(len(words(text)) for column in ['text1', 'text2'] for text in notes[column])),
        'surface_rule': shortcut,
        'best_event_epoch': epoch, 'observed_event_epochs': observed_epochs,
        'fixed': int(comparison.fixed.sum()), 'broken': int(comparison.broken.sum()),
        'changed': int(comparison.changed.sum()), 'changed_wrong': int(comparison.changed_wrong.sum()),
        'old_error_groups': len(old_coverage), 'old_error_rows': int(old_coverage.reference_wrong_rows.sum()),
        'event_error_groups': int(coverage.event_wrong_rows.gt(0).sum()),
        'event_error_rows': int(coverage.event_wrong_rows.sum()),
        'coverage_of_old_errors': counts.to_dict('index'),
        'class_repairs': comparison.groupby('gold')[['fixed', 'broken']].sum().to_dict('index'),
        'bootstrap': bootstrap,
        'normalized_ordered_label_conflict_groups': len(full.loc[conflict, keys].drop_duplicates()),
        'conflict_rows_from_new_200': int(full.loc[conflict, 'instance_id'].isin(notes.instance_id).sum()),
        'analysis_only': 'No model loaded; no training; no training data/config modified.',
        'assessment_limit': 'Coverage/fluency judgements are assistant hypotheses; shortcut existence does not prove model use.',
        'files': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [source_path, notes_path, reference_path, event_path, Path(__file__).resolve()]},
    }
    (OUTPUT_DIR / 'statistics.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding='utf-8')
    print('源对照/训练行：', len(source), '/', len(notes))
    print('只用表面规则：', shortcut['new_200'])
    print('最佳事件轮次：', epoch, '；已记录轮次：', observed_epochs)
    print('修好/破坏：', statistics['fixed'], '/', statistics['broken'])
    print('旧错误覆盖（人工判别模式分析）：\n', counts.to_string())
    print('已有标签冲突组：', statistics['normalized_ordered_label_conflict_groups'], '；新增涉及行：', statistics['conflict_rows_from_new_200'])
    print('配对bootstrap：', bootstrap)
    print('报告数据目录：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
