"""逐组审计0.938145的41行错误：分数、跨轮次、源语境与实际Train读出。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from experiments.deberta_manual_aug.prepare_manual_data import normalize, INVERSE
from experiments.deberta_manual_aug.analyze_current_error_types import NOTES, SHORT, error_family
from experiments.deberta_manual_aug.review_events import bootstrap_difference


HERE = Path(__file__).resolve().parent
RUN = HERE / 'outputs/run06_relation48_staged'
REFERENCE = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope'
MIXED = HERE / 'outputs/run08_relation48_mixed_g8'
SOURCES = ROOT / 'experiments/deberta_generalization_audit/outputs/row_ce_frozen_epoch8'
OUTPUT = RUN / 'error_analysis'
LABELS = ['F', 'B', 'E', 'N']
COMBINATIONS = [('F', 'B'), ('B', 'F'), ('E', 'E')]

# 新复核只描述待判断的关系，不替换官方金标，不据此宣布不可学习或噪声上限。
# 具体可迁移对照仍需先核对Train约定及文本读法。
REVIEW = {
    '0000018': ('先查切块与语境', '数量范围、事件称呼和残缺of短语混合。近似源对是of Rage rallies，没有Dev里的all，不能当精确溯源。', '先确认源版本及all的来源；不把Rage与anger替换当作完整增强方案。'),
    '0000034': ('先查切块与语境', '源标题是Venezuela Government, Opposition Head for Showdown；Head for Showdown更合理地读作动词。不能先验把Head当领导人名词。', '这不是单纯的职务同义词；需要判断原切块是否混入动词，不能硬造Government+Opposition Head蕴含leader的通用规则。'),
    '0000198': ('可审核的判据迁移', '按数据采用的糖果分类义，chocolate bar比candy bar窄。预测EE把包含当等价。词义存在地域约定，先固定食物义。', '同修饰、不同中心词的上下位，与真正同义改写配对；不简单增加词数。'),
    '0000375': ('先固定事件读法', '源部分上下文包含White House与1:15 PM，支持口头声明读法；孤立deliver statement也可能是递交书面声明。', '先固定口头表达场景，再练习具体言语行为与一般说话；不能给所有deliver statement无条件贴F。'),
    '0000432': ('先核验标注口径', 'black top与necklace没有提供big属性；按孤立字面，两侧也未必可互推。源FORW已存在，不能归咎于本次F/B翻转代码。', '保留官方标签，但不要据此教增加无关修饰就必然F。先核对原图片及原对齐。'),
    '0000497': ('先查切块与语境', 'Dev为of some runway/from the airport；公开近似源是of runway/from an airport，并非完全匹配。地点部分整体和介词角色同时变化。', '不把runway属于airport直接等同两个残缺PP的蕴含；先核对源版本和起飞事件角色。'),
    '0000499': ('先查切块与语境', '源文本本身含cow-schemed与inside and black and white等不自然表达，模型可能同时处理外观改写、对象类别与切块异常。', '先核对文本质量与对象名称；不要为了覆盖这一条继续制造不自然短语。'),
    '0000514': ('可审核的判据迁移', '官方视bird为feeder的类别限制；a/the还引入指称差异。单行F正确，配对后被改成EE，合法组合仅领先金标0.049。', '练习类别限制与定义展开的对照，同时固定指称；不能把这一次近失断言为纯随机噪声。'),
    '0000641': ('先查切块与语境', 'death与murder不是通用同义词；源EQUI可能依赖同一Bhutto新闻事件。配对选择与EE接近。', '区分词义包含和同一事件的不同描述；不把所有death/murder训练成E。'),
    '0000948': ('先查切块与语境', '两侧都是over。源部分上下文均涉及土耳其工会因警方行动而罢工，源EQUI依赖具体事件指称。', '真正要核验的是是否同一事件，不是添加介词释义；crackdown与violence不能普遍互换。'),
    '0000954': ('先查切块与语境', 'school和club作为组织类型并非无条件等价；源EQUI可能是同一批机构的称呼，恢复的上下文仍有空缺。', '核对机构指称和当地用法，不用普通school/club概念硬造E。'),
    '0000980': ('先查切块与语境', 'collapse在一般类别义下比disaster具体；源语境是同一Bangladesh事故，金标按共指为E。模型F符合一般类别包含读法。', '训练判据必须区分同一事件指称与一般类别包含；不能从F错误直接推出模型不懂词义。'),
    '0001103': ('可审核的判据迁移', '在猫毛色描述中ginger与orange被源EQUI合并；模型仍判单向包含。A orange的冠词异常也在源文件中。', '审核对象限定下的颜色近义用法；不要把所有相近颜色或不同材料都当E。'),
    '0001136': ('先查切块与语境', 'street/path与on/along同时变化。未找到完全匹配的公开源记录，不能声称已验证它们指同一路段。', '先查来源；不把所有city path与street无条件合并。'),
    '0001174': ('先查切块与语境', '源部分上下文为Pile of wooden debris with chair和A pile of used wooden building materials with a chair；同一堆物体的共指有解释力。', '两侧of相同；重点是材料、废料和状态在具体场景中的对应，不是介词替换。'),
    '0001551': ('先核验标注口径', '按普通集合义Saudi Women比Saudis窄，模型FB有字面依据；源BACK来自Saudi Women Allowed与Saudis to permit women的不同句法角色。', '这不是新增程序把F/B写反；应核对源对齐约定，不训练反常的人群集合方向。'),
    '0001781': ('可审核的判据迁移', '数据把train当station的类别限制。完整源句都谈列车，同一指称也可能诱导EE。单行具体侧F正确，配对被改错。', '在同一位置关系下练习真正类型限制与同一对象改写；先统一该数据的判据。'),
    '0001940': ('可审核的判据迁移', '按任务约定having a meal属于eat的一种活动粒度，一般eat不保证构成一餐。模型十轮均EE且合法组合内分数很集中。', '需要不同谓词/活动粒度的对照；新增48主要是时间、地点和对象限制，不等于已经覆盖这一判据。'),
    '0002242': ('先核验标注口径', '源REL对应China自由贸易区Opens与Shanghai Free Trade Zone begins operation。官方将REL映成N；开业/开始运作的普通语义可支持E。', '不能继续解释为模型不知道Opens的词义；此例需要核对源REL与当前四分类约定。'),
    '0002616': ('先查切块与语境', '源SIMI的一侧是rooftop跳到另一rooftop，另一侧是balcony跳到另一balcony；关键参照对象不在another片段中。', '需要补回参照对象才好判断；不能直接教所有onto/to不蕴含，也不能只靠片段词典关系确定金标。'),
    '0002900': ('先核验标注口径', '普通职务义下senator通常是lawmaker的一种，F有依据；公开源标注确为SIMI并映为N。', '保留官方金标并核对口径；不要为了修N而否认通常的职务包含。'),
    '0002933': ('先核验标注口径', '源部分上下文均在讲受贿相关的Nigeria议员，源SIMI映N。两种国别/议员称呼的近义程度需核对机构语义和原标注。', '不能把这条N直接当作缺少词义知识的硬负例，更不能凭模型答案改金标。'),
    '0003283': ('先核验标注口径', '源部分上下文两侧都描述蓝色汽车的特写。这个语境使Close up更像镜头/视图名词，不宜主要归因于动词多义。源SIMI映N。', '核对源相似与等价的约定；不编造close-up view与close up永远不等价的规则。'),
}


def save(frame, filename, index=False):
    frame.to_csv(OUTPUT / filename, index=index, encoding='utf-8-sig')


def read_prediction(path):
    frame = pd.read_csv(path, keep_default_na=False).set_index('instance_id')
    frame['gold'] = frame.gold_label.map(SHORT)
    assert len(frame) == 660 and frame.index.is_unique and frame.gold.notna().all()
    return frame


def softmax(values):
    probabilities = np.exp(values - values.max())
    return probabilities / probabilities.sum()


def make_figures(groups, fit_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    ordered = groups.sort_values(['correct_epoch_count', 'pair_id'])
    columns = [f'epoch_{epoch}' for epoch in range(1, 11)]
    correct = ordered[columns].eq(ordered.gold_combination, axis=0).to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(correct, cmap=ListedColormap(['#eeeeee', '#7cabc7']), vmin=0, vmax=1, aspect='auto')
    for row, values in enumerate(ordered[columns].to_numpy()):
        for column, value in enumerate(values):
            axis.text(column, row, value, ha='center', va='center', fontsize=7)
    names = [f'{item.pair_id.rsplit("_", 1)[1]}: {item.text1[:31]} / {item.text2[:31]}'
             for item in ordered.itertuples(index=False)]
    axis.set_yticks(range(len(ordered)), names, fontsize=7)
    axis.set_xticks(range(10), range(1, 11))
    axis.set_xlabel('Epoch')
    axis.set_title('The 23 error groups at selected epoch 2\nLabels show the predicted pair combination (or singleton class).', fontsize=11)
    axis.add_patch(Rectangle((.5, -.5), 1, len(ordered), fill=False, edgecolor='#bf6f21', linewidth=2))
    axis.legend(handles=[Patch(facecolor='#7cabc7', label='Matches official gold'),
                         Patch(facecolor='#eeeeee', label='Wrong')], loc='upper left', bbox_to_anchor=(0, -.08), ncol=2)
    figure.subplots_adjust(left=.49, right=.98, top=.92, bottom=.12)
    figure.savefig(OUTPUT / 'error_epoch_heatmap.png', dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    supports = {'train_all': [933, 933, 894, 530], 'dev': [190, 190, 174, 106]}
    x = np.arange(4)
    for axis, split in zip(axes, ['train_all', 'dev']):
        for offset, method, color in [(-.18, 'independent', '#bb704c'), (.18, 'pair_aware', '#4e87ad')]:
            item = fit_summary[(fit_summary.split == split) & fit_summary.method.eq(method)].iloc[0]
            recalls = np.array([item[f'{label}_correct'] for label in LABELS]) / supports[split]
            axis.bar(x + offset, recalls, width=.36, color=color, label=method)
        axis.set_xticks(x, LABELS)
        axis.set_ylim(0, 1.08)
        axis.set_title('Actual Train 3290' if split == 'train_all' else 'Dev 660')
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Recall')
    axes[1].legend(fontsize=8)
    figure.suptitle('Same selected epoch-2 model: row readout versus constrained decoding', fontsize=10)
    figure.tight_layout()
    figure.savefig(OUTPUT / 'train_dev_class_recall.png', dpi=170)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    current = read_prediction(RUN / 'epoch_2/predictions_with_scores.csv')
    old = read_prediction(REFERENCE / 'epoch_3/predictions_with_scores.csv').loc[current.index]
    mixed = read_prediction(MIXED / 'epoch_7/predictions_with_scores.csv').loc[current.index]
    for other in [old, mixed]:
        assert current[['pair_id', 'text1', 'text2', 'gold']].equals(other[['pair_id', 'text1', 'text2', 'gold']])
    assert abs(f1_score(current.gold, current.pair_prediction, average='weighted') - .9381451572973859) < 1e-12
    history = pd.read_csv(RUN / 'epoch_summary.csv')
    assert int(history[history.method.eq('pair_aware')].sort_values('weighted_f1', ascending=False).iloc[0].epoch) == 2
    all_epochs = {epoch: read_prediction(RUN / f'epoch_{epoch}/predictions_with_scores.csv').loc[current.index]
                  for epoch in range(1, 11)}
    errors = current[current.pair_prediction.ne(current.gold)].copy()
    assert len(errors) == 41 and errors.pair_id.nunique() == len(REVIEW) == 23
    sources = pd.read_csv(SOURCES / 'source_matches_with_context.csv', keep_default_na=False)
    source_records, records = [], []
    for pair_id, group in errors.groupby('pair_id', sort=False):
        full = current[current.pair_id.eq(pair_id)]
        key = pair_id.replace('dico_phrasis_', '')
        priority, explanation, direction = REVIEW[key]
        first = full.iloc[0]
        if len(full) == 2:
            second = full.iloc[1]
            scores = np.array([first.logit_F + second.logit_B, first.logit_B + second.logit_F,
                               first.logit_E + second.logit_E], dtype=float)
            gold_index = COMBINATIONS.index(tuple(full.gold))
            prediction_index = int(scores.argmax())
            assert tuple(full.pair_prediction) == COMBINATIONS[prediction_index]
            score_space = 'three legal combinations only; conditional, not calibrated semantic probability'
        else:
            scores = first[[f'logit_{label}' for label in LABELS]].to_numpy(dtype=float)
            gold_index = LABELS.index(first.gold)
            prediction_index = int(scores.argmax())
            score_space = 'four row classes; not calibrated semantic probability'
        epoch_labels = {f'epoch_{epoch}': ''.join(frame.loc[full.index].pair_prediction)
                        for epoch, frame in all_epochs.items()}
        gold_string = ''.join(full.gold)
        correct_epochs = [epoch for epoch in all_epochs if epoch_labels[f'epoch_{epoch}'] == gold_string]
        exact_sources = sources[sources.instance_id.eq(full.index[0])].copy()
        # 只复用旧审计的文本溯源字段，不复用旧冻结模型的预测或正确率。
        for _, hit in exact_sources.iterrows():
            assert normalize(hit.text1) == normalize(first.text1) and normalize(hit.text2) == normalize(first.text2)
            source_records.append({column: hit[column] for column in [
                'instance_id', 'source_file', 'source_line', 'context_id', 'source_score',
                'matched_swapped', 'oriented_source_relation', 'context1', 'context2', 'context_has_conflicting_slots']})
        contexts = ['A: ' + row.context1 + '\nB: ' + row.context2 for row in
                    exact_sources.drop_duplicates(['context1', 'context2']).itertuples(index=False)]
        record = {'pair_id': pair_id, 'text1': first.text1, 'text2': first.text2,
                  'gold_combination': gold_string, 'current_combination': ''.join(full.pair_prediction),
                  'raw_combination': ''.join(full.prediction),
                  'reference_combination': ''.join(old.loc[full.index].pair_prediction),
                  'mixed_combination': ''.join(mixed.loc[full.index].pair_prediction),
                  'error_rows': len(group), 'primary_type': NOTES[key]['primary_type'],
                  'has_PP_frame': NOTES[key]['has_PP_frame'], 'review_priority': priority,
                  'assessment': explanation, 'suggested_check': direction,
                  'score_over_gold': float(scores[prediction_index] - scores[gold_index]),
                  'top_probability_in_declared_space': float(softmax(scores)[prediction_index]),
                  'score_space': score_space, 'correct_epochs': '|'.join(map(str, correct_epochs)),
                  'correct_epoch_count': len(correct_epochs),
                  'source_relations': '|'.join(exact_sources.oriented_source_relation.unique()),
                  'exact_source_hits': len(exact_sources), 'source_partial_contexts': '\n---\n'.join(contexts),
                  'status': 'assistant review; official labels preserved; not independently adjudicated',
                  **epoch_labels}
        records.append(record)
    groups = pd.DataFrame(records)
    save(groups, 'error_groups_reviewed.csv')
    save(groups[['pair_id', 'text1', 'text2', 'gold_combination', 'correct_epoch_count'] +
                [f'epoch_{epoch}' for epoch in all_epochs]], 'error_epoch_stability.csv')
    save(pd.DataFrame(source_records), 'error_exact_source_contexts.csv')
    contexts = pd.read_csv(SOURCES / 'source_reconstructed_contexts.csv', keep_default_na=False)
    candidates = []
    for key, filename, line_number in [('0000018', 'PhrasIS.test.headlines.all.txt', 1150),
                                       ('0000497', 'PhrasIS.train.images.all.txt', 1961)]:
        path = ROOT / 'experiments/deberta_generalization_audit/sources' / filename
        raw = path.read_text(encoding='utf-8').splitlines()[line_number - 1]
        fields = raw.split('\t')
        identifier = f'{filename}:{fields[6].strip()}'
        context = contexts[contexts.context_id.eq(identifier)].set_index('side')
        case = groups[groups.pair_id.eq('dico_phrasis_' + key)].iloc[0]
        candidates.append({'pair_id': case.pair_id, 'dev_text1': case.text1, 'dev_text2': case.text2,
                           'source_file': filename, 'source_line': line_number,
                           'candidate_text1': fields[3].strip(), 'candidate_text2': fields[2].strip(),
                           'source_relation_unoriented': fields[1], 'candidate_is_swapped': True,
                           'candidate_context1': context.loc[1].reconstructed_context,
                           'candidate_context2': context.loc[0].reconstructed_context,
                           'status': 'near-source candidate, not an exact match; wording changes must be resolved',
                           'source_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save(pd.DataFrame(candidates), 'near_source_candidates_not_exact.csv')
    errors['error_family'] = [error_family(g, p) for g, p in zip(errors.gold, errors.pair_prediction)]
    save(errors.reset_index(), 'error_rows.csv')
    confusion = pd.crosstab(current.gold, current.pair_prediction).reindex(index=LABELS, columns=LABELS, fill_value=0)
    save(confusion, 'dev_pair_aware_confusion.csv', index=True)
    category_counts = groups.groupby('primary_type').agg(groups=('pair_id', 'size'), rows=('error_rows', 'sum'))
    save(category_counts.sort_values('rows', ascending=False), 'semantic_error_types.csv', index=True)
    labels = errors.groupby('error_family').agg(rows=('gold', 'size'), groups=('pair_id', 'nunique'))
    save(labels, 'label_error_types.csv', index=True)
    # 当前最佳模型真实Train读出，不能拿另一个旧模型的Train表现替代。
    train = pd.read_csv(OUTPUT / 'train_readout.csv', keep_default_na=False)
    manifest = json.loads((OUTPUT / 'inference_manifest.json').read_text(encoding='utf-8'))
    assert manifest['epoch'] == 2 and not manifest['training_performed']
    assert all(value == 0 for value in manifest['dev_prediction_differences'].values())
    assert train.instance_id.tolist() == pd.read_csv(RUN / 'training_data.csv').instance_id.tolist()
    fit_summary = pd.read_csv(OUTPUT / 'train_dev_readout_summary.csv')
    new_errors = train[train.instance_id.str.startswith('aug_manual_r_') & ~train.correct].copy()
    save(new_errors, 'train_new48_independent_errors.csv')
    save(train[~train.pair_correct], 'train_pair_aware_errors.csv')

    # 仅按实际文本建立Train近邻；标注原始模型的Train正确性，避免声称“有这种样本就已学会”。
    official = train[train.instance_id.str.startswith('dico_')].reset_index(drop=True)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=True)
    values = vectorizer.fit_transform(pd.concat([official.text1, official.text2], ignore_index=True))
    first_values, second_values = values[:len(official)], values[len(official):]
    first_query = vectorizer.transform(groups.text1)
    second_query = vectorizer.transform(groups.text2)
    similarity = ((first_query @ first_values.T) + (second_query @ second_values.T)).toarray() / 2
    neighbours = []
    for index, query in groups.iterrows():
        seen = set()
        for candidate in np.argsort(-similarity[index], kind='stable'):
            hit = official.iloc[candidate]
            if hit.pair_id in seen:
                continue
            seen.add(hit.pair_id)
            neighbours.append({'dev_pair_id': query.pair_id, 'rank': len(seen),
                               'dev_text1': query.text1, 'dev_text2': query.text2,
                               'train_instance_id': hit.instance_id, 'train_pair_id': hit.pair_id,
                               'train_text1': hit.text1, 'train_text2': hit.text2, 'train_gold': hit.gold,
                               'train_independent': hit.prediction, 'train_pair_aware': hit.pair_prediction,
                               'train_independent_correct': hit.correct, 'train_pair_correct': hit.pair_correct,
                               'char_tfidf_similarity': float(similarity[index, candidate]),
                               'limit': 'Surface neighbour only; not proof of semantic analogy or training generalization.'})
            if len(seen) == 3:
                break
    save(pd.DataFrame(neighbours), 'official_train_neighbours.csv')

    comparison = current[['pair_id', 'text1', 'text2', 'gold']].copy()
    comparison['reference_prediction'] = old.pair_prediction
    comparison['event_prediction'] = current.pair_prediction
    comparison['fixed'] = old.pair_prediction.ne(current.gold) & current.pair_prediction.eq(current.gold)
    comparison['broken'] = old.pair_prediction.eq(current.gold) & current.pair_prediction.ne(current.gold)
    save(comparison[comparison.fixed].reset_index(), 'fixed_vs_0p935.csv')
    save(comparison[comparison.broken].reset_index(), 'broken_vs_0p935.csv')
    bootstrap = bootstrap_difference(comparison)
    hurt = current[current.prediction.eq(current.gold) & current.pair_prediction.ne(current.gold)]
    save(hurt.reset_index(), 'correct_rows_changed_wrong_by_decoding.csv')

    # 核对精确/规范化文本重复，而非把语义近邻的标签不同叫“冲突”。
    normalized = train.copy()
    normalized['key'] = [tuple(sorted([normalize(a), normalize(b)])) for a, b in zip(train.text1, train.text2)]
    normalized['aligned_gold'] = [gold if normalize(a) <= normalize(b) else INVERSE[gold]
                                  for a, b, gold in zip(train.text1, train.text2, train.gold)]
    conflict_keys = set(normalized.groupby('key').aligned_gold.nunique().loc[lambda s: s.gt(1)].index)
    conflicts = normalized[normalized.key.isin(conflict_keys)]
    save(conflicts[['instance_id', 'pair_id', 'text1', 'text2', 'gold', 'aligned_gold']], 'train_exact_label_conflicts.csv')
    train_pairs = set(normalized.key)
    current_error_keys = {tuple(sorted([normalize(a), normalize(b)])) for a, b in zip(errors.text1, errors.text2)}
    exact_error_overlap = len(current_error_keys & train_pairs)

    facts = {
        'best_epoch': 2, 'weighted_f1': f1_score(current.gold, current.pair_prediction, average='weighted'),
        'error_rows': len(errors), 'error_groups': len(groups), 'paired_error_groups': int(groups.error_rows.eq(2).sum()),
        'singleton_error_groups': int(groups.error_rows.eq(1).sum()),
        'never_correct_error_groups_in_ten_epochs': int(groups.correct_epoch_count.eq(0).sum()),
        'never_correct_error_rows_in_ten_epochs': int(groups.loc[groups.correct_epoch_count.eq(0), 'error_rows'].sum()),
        'sometimes_correct_error_rows': int(groups.loc[groups.correct_epoch_count.gt(0), 'error_rows'].sum()),
        'exact_source_error_groups': int(groups.exact_source_hits.gt(0).sum()),
        'no_exact_source_error_ids': groups.loc[groups.exact_source_hits.eq(0), 'pair_id'].tolist(),
        'PP_frame_error_rows': int(groups.loc[groups.has_PP_frame, 'error_rows'].sum()),
        'error_pair_exact_overlap_in_training': exact_error_overlap,
        'train_conflicting_normalized_unordered_pairs': len(conflict_keys),
        'decoding_fixed': int((current.prediction.ne(current.gold) & current.pair_prediction.eq(current.gold)).sum()),
        'decoding_broken': len(hurt), 'reference_fixed': int(comparison.fixed.sum()),
        'reference_broken': int(comparison.broken.sum()), 'reference_bootstrap': bootstrap,
        'training_performed': False,
        'source_limit': 'Contexts reconstructed from aligned source spans, with GAPs; not complete original sentences or model inputs. Near-source records are not counted as exact matches.',
        'review_limit': 'Reading hypotheses, not corrected gold or a ceiling on learnability. Epochs are one trajectory, not independent trials.',
        'input_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            RUN / 'epoch_2/predictions_with_scores.csv', REFERENCE / 'epoch_3/predictions_with_scores.csv',
            MIXED / 'epoch_7/predictions_with_scores.csv', RUN / 'epoch_summary.csv',
            SOURCES / 'source_matches_with_context.csv', OUTPUT / 'train_readout.csv', Path(__file__).resolve()]},
    }
    (OUTPUT / 'error_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# 0.938145版本：41行、23组错误审计', '',
        '对象为run06_relation48_staged最佳epoch2。官方Dev独立预测和pair-aware均逐条复现，差异0。',
        '新增操作只有读取最佳权重并做Train/Dev推理、分析已有预测；没有优化器、反向传播、训练或改金标。', '',
        '## 最重要的新发现：Train的四分类读出也失衡', '',
        '| 数据 | 独立W-F1 | 独立正确 | pair-aware正确 | 独立非N→N |', '|---|---:|---:|---:|---:|',
    ]
    for split in ['train_all', 'train_official', 'train_old_scope200', 'train_new48', 'dev']:
        independent = fit_summary[(fit_summary.split == split) & fit_summary.method.eq('independent')].iloc[0]
        paired = fit_summary[(fit_summary.split == split) & fit_summary.method.eq('pair_aware')].iloc[0]
        lines.append(f'| {split} | {independent.weighted_f1:.6f} | {int(independent.correct)}/{int(independent.rows)} | '
                     f'{int(paired.correct)}/{int(paired.rows)} | {int(independent.nonN_to_N)} |')
    lines += [
        '', '新48条独立F/B/E/N分别答对9/1/12/12；旧scope200的B为0/40，官方Train的B为120/881。',
        '因此不能把当前问题仅说成“Train全会、Dev不会”。最佳checkpoint连训练行的四分类读出也未保住。',
        '同时解码后Train 3280/3290、Dev619/660，确有结构条件下的泛化差距；两种问题并存。',
        '当前没有同一epoch在Row阶段结束、Pair阶段开始前的Train读出，不能由最终checkpoint单独证明全部漂移都发生在Pair阶段。', '',
        '## Dev混淆', '', '| 金标→预测 | 错误行 | 组数 |', '|---|---:|---:|',
    ]
    for kind, item in labels.iterrows():
        lines.append(f'| {kind} | {item.rows} | {item.groups} |')
    lines += [
        '', 'E与F/B的混淆共30/41行，方向反转6行、N错误5行。E错误率14/174=8.05%，F/B各11/190=5.79%，N为5/106=4.72%。',
        '这些是结构解码后的错误分布。独立预测中还有297条非N→N，N precision只有0.2538，不能称为N分类整体已解决。', '',
        '## 逐组关系复核', '',
        'FB表示表中text1的F与反向行的B；BF含义相反。分差为所选合法组合减金标组合，N单例则是所选类减N。',
        '两种分数空间不同；合法组合内softmax只说明条件排序，不能当校准过的语义置信度。', '',
        '| ID | text1 / text2 | 金标→预测 | 领先金标分差 | 十轮中答对组的轮数 | 审核重点 |',
        '|---|---|---|---:|---:|---|',
    ]
    for item in groups.itertuples(index=False):
        lines.append(f'| {item.pair_id.rsplit("_", 1)[1]} | {item.text1} / {item.text2} | '
                     f'{item.gold_combination}→{item.current_combination} | {item.score_over_gold:.3f} | '
                     f'{item.correct_epoch_count}/10 | {item.assessment} |')
    lines += [
        '', '## 哪些反复错，哪些曾经答对', '',
        '7组13行在这10个epoch全部错误：deliver statement/speak、black top+necklace/big necklace、runway/airport、Saudi Women/Saudis、station/train station、eat/having a meal、Nigeria lawmaker/Nigerian MP。',
        '其中混有可迁移的活动/限定判断、源切块和口径问题，不能统一归为缺少词义。',
        '其余16组28行至少在一轮答对过，说明当前错集随epoch变化。一次正确可能偶然猜中，不能证明输入语境充分或规则已稳定掌握。',
        '不能按每组最好的epoch拼出新系统；这里只检查错误是否随训练轨迹变化。', '',
        '## 解码也破坏了两条本来正确的行', '',
        'from a bird feeder→from the feeder这一行单独预测F正确，合法组合却选EE。',
        '第一行F-E约+1.321，另一行B-E约-1.369；加和后FB比EE低约0.049，因此两行一起被判E。',
        'at the train station→at the station同样是单行F正确、配对后被改E，金标组合落后约0.912。',
        '0.938模型的结构解码总体修好272行、破坏2行；规则不是只改错行，也不保证金标组合。',
        '小分差值得做稳定性检查，但不证明这些问题只是噪声或无需补判据。', '',
        '## 源语境与任务口径', '',
        '23组中20组有公开源标注精确匹配；上下文来自对齐片段重建，含GAP，不是完整原句。',
        '0000034源标题里的Head for Showdown通常是动词。这纠正了此前按职务名词理解Head的分析。',
        '0001551源上下文是Saudi Women Allowed...与Saudis to permit women...，词在两侧句子承担不同角色。官方BACK不是本次程序误交换F/B的产物，但孤立集合读法与它有张力。',
        '0000432的big只在一侧出现，原图可能提供信息，两个孤立片段不能直接证明官方F方向。源文件里确为FORW。',
        '5个N错误的源关系：Opens/begins operation为REL；其余4个为SIMI。不能把SIMI/REL映成N后的一切错误都解释成模型缺少普遍语义知识。',
        'Opens源部分上下文明确是自由贸易区开业，因此“裸词Opens多义”不是主要证据；close-up源两侧都描述蓝色汽车镜头，主要按动词多义解释也不充分。',
        '0000018和0000497只有近似源候选，Dev分别多了all、改了冠词。0001136没有找到完全匹配源。来源有差异，不能强行套候选上下文。', '',
        '## 什么值得继续补，什么先审核', '',
        '优先可审核的关系：bird feeder/feeder、train station/station的类型限制，chocolate/candy的类别包含，eat/meal的活动粒度，猫毛色ginger/orange的领域近义。',
        '这5组是研究优先级，不是宣称它们一定能修或其余不可能修。新样本仍须从Train约定出发，不抄Dev内容。',
        'Bhutto death/murder、Bangladesh collapse/disaster、school/club、wooden debris/building materials等先核对同一事件/场景的指称，不造无条件等价规则。',
        'N的senator/lawmaker、Opens/operation、close-up等先核对任务口径，不直接批量做反语义硬负例。', '',
        '## Train近邻与冲突', '',
        f'当前23组错误与实际训练数据的规范化无序文本对精确交集为{exact_error_overlap}组。',
        f'实际Train有{len(conflict_keys)}组规范化无序文本对标签冲突；不能据此说当前23组都因冲突而错。',
        'official_train_neighbours.csv列出每组3个官方Train的字符相似近邻，并给出本次模型在该Train行的真实预测。表面近邻不是语义同型证据。',
        'station的近邻特别值得核对：Train里in a station/at a train station是E，in a station/at the city train station是B，inside a train station/at the train station是N。',
        '这三对不是精确文本冲突，介词和限定都不同；但说明不能只凭出现train station就套同一种标签。',
        '本次模型连Train中in a station/at a train station的E也没有拟合，pair-aware仍判BF。',
        'train_new48_independent_errors.csv列出新48条中仍被单独判错的14条；全部是F/B→N。', '',
        '## 与0.935493的差别', '',
        '修好17行、破坏15行，净多对2行。F/B/E总正确仍为518，F/B各多1、E少2；N从99到101。',
        f'固定两个已选模型的383组bootstrap差值{bootstrap["delta_percentage_points"]:+.4f}个百分点，95%区间'
        f'[{bootstrap["ci95_percentage_points"][0]:+.4f}, {bootstrap["ci95_percentage_points"][1]:+.4f}]，不含训练种子与选模不确定性。', '',
        '## 独立读出为什么仍可能失衡', '',
        '三组合Pair CE看不到N分数；全局F加a、B减a时，FB/BF/EE三个组合分数也完全不变。',
        '因此配对排序好不意味着N与非N、F与B的单行偏置受到同样约束。Train和Dev都表现出B召回很低，与这一盲区相容。',
        '没有N的直接Pair梯度，也不等于N输出或参数不变：共享编码器会更新；沿用同一个AdamW时，历史动量和权重衰减也可能更新当前梯度为零的参数分量。',
        '这只是允许失衡的数学机制，不是对全部错误的因果证明。单纯增大Pair权重不能补上它对N与全局F/B偏置的盲区。',
        '下一步应分开验证：固定现有模型的读出偏置诊断，以及上述少数清楚关系的泛化。不要把所有问题继续归为需要更多人工样本。',
        '本报告没有拟合Dev阈值、修改官方标签、挑选逐样本epoch或新增训练数据。', '',
        '附图：train_dev_class_recall.png显示同一模型在Train/Dev各类召回，error_epoch_heatmap.png显示当前23组在十轮中的变化。', '',
    ]
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    make_figures(groups, fit_summary)
    print('当前错误类型：\n', labels.to_string())
    print('关系主类：\n', category_counts.to_string())
    print('从未答对组/行：', facts['never_correct_error_groups_in_ten_epochs'], facts['never_correct_error_rows_in_ten_epochs'])
    print('源精确匹配组：', facts['exact_source_error_groups'])
    print('训练精确冲突组：', len(conflict_keys), '；当前错误与Train精确重合组：', exact_error_overlap)
    print('报告：', OUTPUT / 'REPORT.md')


if __name__ == '__main__':
    main()
