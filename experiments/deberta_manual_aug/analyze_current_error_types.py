"""逐组复核run04剩余错误及新增错误；不生成增强，不读取copy运行，不加载模型。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import f1_score


HERE = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / 'experiments/deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv'
CURRENT_PATH = HERE / 'outputs/run04_boundary_120/epoch_1/predictions_with_scores.csv'
OUTPUT_DIR = HERE / 'outputs/run04_boundary_120/measurement_review'
SHORT = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
         'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}

# 主类互斥，只表示本次复核认为最需要判断的关系；不是已证明的错误成因。
# pp标记可与各主类重叠；不能把所有带介词的错误都计为“介词知识错误”。
LEXICAL = '名词词义与实体类别'
STRUCTURE = '修饰范围与短语结构'
PREDICATE = '动作粒度与谓词改写'
CONTEXT = '新闻场景与事件指称'
PREPOSITION = '介词关系与路径角色'
DEGREE = '程度级别与比较'
DIRECTION = '标签方向疑点'


def note(kind, pp, context, label_question, issue, design):
    return {'primary_type': kind, 'has_PP_frame': pp, 'context_review_needed': context,
            'possible_label_or_direction_issue': label_question,
            'relation_to_review': issue, 'augmentation_direction': design}


NOTES = {
    '0000018': note(CONTEXT, True, True, False, 'Rage rallies与march of anger涉及事件称呼、数量及省略，不只是of的差异。', '先核对源事件指称；不把示威和愤怒行进普遍标成固定关系。'),
    '0000034': note(CONTEXT, False, True, False, '政府、反对派、Head与leader的新闻实体及省略关系。', '从Train中明确的职务/实体类型出发；不复刻这个新闻标题。'),
    '0000198': note(LEXICAL, False, False, False, 'chocolate bar与candy bar的词汇范围；没有介词差异。', '按普通零食类别义审核上下位；配不同长度但等价的反例。'),
    '0000375': note(PREDICATE, False, True, False, 'to是两侧不定式标记；deliver statement是否口头表达取决于事件义。', '补明确对象、明确同一事件解释的具体动作/一般动作对照，避免把书面声明也强制归为说话。'),
    '0000432': note(STRUCTURE, True, True, True, 'black top和necklace不直接保证big necklace；修饰归属与官方F方向需核验。', '可以练习修饰绑定，但不能照此例推广长侧一定更具体或F方向。'),
    '0000449': note(LEXICAL, False, False, False, 'car属于vehicle；black保留，关键是中心名词的词义包含。', '同修饰、换中心词的上下位，和同义/不包含的对应反例。'),
    '0000497': note(CONTEXT, True, True, False, 'runway/airport不是同一中心词，of/from也改变关系；单独片段不足以唯一恢复源对齐。', '检查源句；不自动把部分/整体或两个地点相关关系当蕴含。'),
    '0000499': note(STRUCTURE, True, True, False, '外观属性、dome/enclosure称呼及异常语法混合；inside本身两侧相同。', '优先自然且范围清楚的属性/对象改写；不复制异常片段拼接。'),
    '0000514': note(STRUCTURE, True, False, False, 'from两侧相同；bird限制feeder的类型，另有冠词变化。', '在同一自然介词框架中保持/删除真正缩小范围的限定，与定义性展开E相配。'),
    '0000616': note(STRUCTURE, True, True, False, 'TV、screen、wall-mounted及of片段混合，需要辨别对象与片段角色。', '属性限制与对象名称展开分开，不能只删词就赋F。'),
    '0000641': note(CONTEXT, True, True, False, 'death/murder不是无条件同义；金标E可能指向同一Bhutto事件。', '需要源事件或明确指称条件；仅更换in/over不能生成可靠E。'),
    '0000711': note(PREDICATE, True, True, False, 'of为介词、to为不定式标记；bribing/pay off关键在pay off的贿赂义与偿债义。', '使用对象可消歧的自然动作改写；不要把pay off在所有语境都等同bribe。'),
    '0000729': note(LEXICAL, False, False, False, '按人类指称的通常读法，短名称与多词定义的等价。', '保留定义性等价这一目标，但使用可审核的角色定义；不宣称所有adult female都不需语境。'),
    '0000817': note(CONTEXT, True, True, False, '水边岩石不必是海岸，金标E可能依赖图像/源指称。', '核对上下文；不把所有相近场景一律教为E。'),
    '0000943': note(PREDICATE, False, True, False, 'to是动作不定式；slide up/zipping up需要知道正在操作的对象。', '同一明确对象下不同表达的动作改写，不把任意上滑都等价于拉拉链。'),
    '0000948': note(CONTEXT, True, True, False, '两侧介词都是over；crackdown/violence是否指同一事件才是关键。', '同事件的称呼改写需清楚指称；不据此把两个词普遍等同。'),
    '0000950': note(PREDICATE, False, True, False, '两侧to均为不定式；set out有出发、陈述等多义，announce不与所有义等价。', '使用显式宾语/论元固定动词义，并与只有相关性的N相配。'),
    '0000954': note(LEXICAL, False, True, False, 'school与club的机构义可能在当前语境指同一组织，通常不无条件等价。', '审核机构类别与同一组织的称呼；不把school/club一律标E。'),
    '0000979': note(CONTEXT, True, True, False, 'after/to的时间/目标角色不同；源片段被标E需先检查原对齐。', '不以换介词自动给E或N；先确定关系角色和指称。'),
    '0000980': note(CONTEXT, False, True, False, 'collapse与disaster通常是具体/一般事件词，当前E可能依赖同一新闻事故。', '区分类别包含与同一事件指称；不能用一般类别关系冒充语境等价。'),
    '0000994': note(STRUCTURE, True, False, False, '月份前置修饰与in May结构的改写；不是增加一个新的地点/时间。', '在时间修饰清楚时补名词修饰与介词结构的等价，并配真正增加时间限制的F/B。'),
    '0001103': note(LEXICAL, False, True, False, 'ginger/orange是颜色近义与动物毛色用法；无介词。', '先核验颜色/物体义下的近义精度，不把所有近似颜色都合并为E。'),
    '0001136': note(CONTEXT, True, True, False, 'street/path及on/along同时改变，是否同一路段需场景参照。', '先固定对象和关系再比较PP；不能仅因都表示地点就标E。'),
    '0001156': note(CONTEXT, True, True, False, 'on两侧相同；grass/lawn是描述与场景指称，而非换介词。', '审核场景指称；当前run04已经修好，不再把它当本轮必修错误。'),
    '0001174': note(STRUCTURE, True, True, False, '两侧都是of；debris与used building materials及wooden/used范围更重要，通用字面E不明确。', '区分定义展开与真正增加来源/状态限制；先核对源场景，不能机械删used/building后标E。'),
    '0001456': note(PREDICATE, False, True, False, 'skijoring与playing的运动/活动义选择；不是词数决定方向。', '当前run04已修好；如补动作粒度，要固定运动义而不机械套任意动词。'),
    '0001551': note(DIRECTION, False, True, True, '通常Saudi Women比Saudis更窄，与官方B方向有疑问。', '保留官方标签并核验来源；不造反常集合规则来追此单例。'),
    '0001567': note(CONTEXT, True, True, False, '事故、死亡统计和新闻省略成分，不能仅按as/after区分。', '核对新闻源；若缺少必要语境，单独报告而不硬编同类金标。'),
    '0001781': note(STRUCTURE, True, False, False, '两侧同为at；train限定station类型，不是介词含义不同。', '保持PP关系的自然限制对照，配长而不增信息的等价表达。'),
    '0001940': note(PREDICATE, False, True, False, 'eat/having a meal涉及活动粒度与事件时段解释。', '补明确动作具体化和同体貌对照；不把任何进餐阶段都机械等同正在吃。'),
    '0002242': note(PREDICATE, False, True, False, 'Opens无对象，可能开门、开业等；多义本身不自动意味着N。', '先找Train里相同约定；使用明确论元区分词义，不直接用裸多义词制造N。'),
    '0002244': note(STRUCTURE, False, True, False, 'bare 2与two people的信息类型/中心对象不同，需核对任务对片段省略的约定。', '先查Train同结构标注；不能因删除中心词就统一F，也不能把所有数量表达都标N。'),
    '0002251': note(LEXICAL, False, False, False, 'Supreme Guide所指人员与group所指组织的实体类型不同。', '同领域、共享词但实体类型不同的自然短语N，和完整角色定义E相配。'),
    '0002463': note(PREDICATE, False, True, False, '名词化integration与join的动作/角色表达不保证普遍等价。', '当前run04已修好；不在未经审核情况下把相关名词化一律标N。'),
    '0002487': note(PREPOSITION, True, True, False, '事件名称NP与in+事件PP的关系角色不同，另有Southern范围限制。', '核验省略中心词；可考虑事件实体与参与/处于事件的角色差异，不只换地名。'),
    '0002492': note(DEGREE, False, False, False, '等级类别与比较集合中的最高级不同；不是普通名词上下位。', '只使用定义明确的等级和比较范围，不把所有最高级与程度形容词一律标N。'),
    '0002613': note(PREPOSITION, True, True, False, 'on/ beside有角色差异，shore/beach也不同，on beside片段本身异常。', '先恢复合理句法和参照，不直接把异常字符串作为通用介词规则。'),
    '0002616': note(PREPOSITION, True, True, False, 'onto/to可能包含接触或到达差异，但another省略了对象。', '用关系与参照清楚的路径/目标对照；不把所有to/onto都视作固定标签。'),
    '0002671': note(PREDICATE, False, True, False, 'performs/to do的词义、体貌和论元省略；to不是此处的地点介词。', '先查任务约定；不能推广为perform与do永远不蕴含。'),
    '0002738': note(PREDICATE, False, False, False, '定罪与入狱是相关事件而非必然互推。', '当前run04已修好；事件阶段不再是当前最优先的大批量增强方向。'),
    '0002844': note(PREPOSITION, True, True, False, 'Northwest China为地点NP，in China为位置关系，不能只按地区包含处理。', '核验实体与位置关系的片段约定，保留角色信息。'),
    '0002900': note(LEXICAL, False, True, True, '通常senator属于lawmaker，与官方N有字面冲突可能。', '查源标注；不为了此例教模型否认通常职务包含。'),
    '0002933': note(LEXICAL, False, True, True, 'Nigeria/Nigerian地域表达及lawmaker/MP称谓需要来源和机构背景。', '单独核验地域、职务词义和源约定，不按词面近似自动E。'),
    '0003245': note(PREPOSITION, True, True, False, 'into/at camera的关系还依赖looking等被省略谓词；不只是两个介词词典释义。', '补显式动作和相同参照的方向/目标对照；不把所有look into/at camera都标N。'),
    '0003283': note(STRUCTURE, False, True, False, 'close-up可作名词/修饰语，close up亦可能是动词；缺少词性和论元。', '先查Train及原句；用清楚的词性/中心对象变化对照，不强制所有长短表达N。'),
}


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(REFERENCE_PATH, keep_default_na=False).set_index('instance_id')
    new = pd.read_csv(CURRENT_PATH, keep_default_na=False).set_index('instance_id').loc[old.index]
    assert len(old) == len(new) == 660 and old.index.is_unique and new.index.is_unique
    assert old.gold_label.equals(new.gold_label) and old.text1.equals(new.text1) and old.text2.equals(new.text2)
    rows = new[['pair_id', 'text1', 'text2']].copy()
    rows['gold'] = old.gold_label.map(SHORT)
    rows['reference_prediction'] = old.pair_prediction
    rows['current_prediction'] = new.pair_prediction
    rows['old_wrong'] = rows.reference_prediction.ne(rows.gold)
    rows['current_wrong'] = rows.current_prediction.ne(rows.gold)
    rows['fixed'] = rows.old_wrong & ~rows.current_wrong
    rows['broken'] = ~rows.old_wrong & rows.current_wrong
    rows['current_error_family'] = [error_family(g, p) for g, p in zip(rows.gold, rows.current_prediction)]
    metadata = pd.DataFrame.from_dict({f'dico_phrasis_{key}': value for key, value in NOTES.items()}, orient='index')
    selected = rows[rows.old_wrong | rows.current_wrong].join(metadata, on='pair_id')
    assert selected.primary_type.notna().all(), '有新错误未逐组人工复核。'
    selected['assessment_status'] = 'assistant relation-level review; not independent gold or causal adjudication'
    current = selected[selected.current_wrong].copy()
    current.reset_index().to_csv(OUTPUT_DIR / 'current_error_rows.csv', index=False, encoding='utf-8-sig')
    grouped = selected.groupby('pair_id', sort=False).agg(
        text1=('text1', 'first'), text2=('text2', 'first'), gold=('gold', 'first'),
        reference_prediction=('reference_prediction', 'first'), current_prediction=('current_prediction', 'first'),
        old_error_rows=('old_wrong', 'sum'), current_error_rows=('current_wrong', 'sum'),
        fixed_rows=('fixed', 'sum'), broken_rows=('broken', 'sum'), primary_type=('primary_type', 'first'),
        has_PP_frame=('has_PP_frame', 'first'), context_review_needed=('context_review_needed', 'first'),
        possible_label_or_direction_issue=('possible_label_or_direction_issue', 'first'),
        relation_to_review=('relation_to_review', 'first'), augmentation_direction=('augmentation_direction', 'first'),
    ).reset_index()
    grouped.to_csv(OUTPUT_DIR / 'error_type_groups.csv', index=False, encoding='utf-8-sig')
    grouped[grouped.current_error_rows.gt(0)].to_csv(OUTPUT_DIR / 'current_error_groups.csv', index=False, encoding='utf-8-sig')
    summary = grouped.groupby('primary_type').agg(
        old_error_rows=('old_error_rows', 'sum'), current_error_rows=('current_error_rows', 'sum'),
        fixed_rows=('fixed_rows', 'sum'), broken_rows=('broken_rows', 'sum'),
    )
    summary['current_groups'] = grouped[grouped.current_error_rows.gt(0)].groupby('primary_type').size()
    summary['net_added_errors'] = summary.current_error_rows - summary.old_error_rows
    summary = summary.sort_values('current_error_rows', ascending=False)
    summary.to_csv(OUTPUT_DIR / 'error_type_summary.csv', encoding='utf-8-sig')
    confusion_types = current.groupby('current_error_family').agg(rows=('gold', 'size'), groups=('pair_id', 'nunique'))
    confusion_types.to_csv(OUTPUT_DIR / 'current_error_label_counts.csv', encoding='utf-8-sig')
    old_pp = int((selected.old_wrong & selected.has_PP_frame).sum())
    new_pp = int(current.has_PP_frame.sum())
    pp_broken = int((selected.broken & selected.has_PP_frame).sum())
    pp_fixed = int((selected.fixed & selected.has_PP_frame).sum())
    facts = {
        'reference_f1': f1_score(rows.gold, rows.reference_prediction, average='weighted'),
        'current_f1': f1_score(rows.gold, rows.current_prediction, average='weighted'),
        'old_errors': int(rows.old_wrong.sum()), 'current_errors': int(rows.current_wrong.sum()),
        'old_error_groups': int(grouped.old_error_rows.gt(0).sum()), 'current_error_groups': int(grouped.current_error_rows.gt(0).sum()),
        'old_PP_frame_error_rows': old_pp, 'current_PP_frame_error_rows': new_pp,
        'current_PP_frame_error_groups': int(current[current.has_PP_frame].pair_id.nunique()),
        'current_PP_frame_error_row_fraction': new_pp / len(current),
        'PP_frame_fixed': pp_fixed, 'PP_frame_broken': pp_broken,
        'scope': 'Selected run04 epoch1 vs reference epoch3. Copy run not read; no model loading, data generation or training.',
        'classification_limit': 'Primary types are mutually exclusive assistant judgement; PP-frame tags overlap them and do not establish causes or irreducible errors.',
        'files': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [REFERENCE_PATH, CURRENT_PATH, Path(__file__).resolve()]},
    }
    (OUTPUT_DIR / 'error_type_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# 当前错误子类型：先等copy，不新增训练样本', '',
        '对象：run04_boundary_120最佳epoch1（PA W-F1=0.907866）；参考为0.935493的scope版epoch3。',
        '只读取这两组已有预测，不读取正在运行的copy结果，不训练，也不生成下一批样本。', '',
        '## 当前最多的标签混淆', '',
        '剩余61行错误、37组；E→F/B为24行，F/B→E为20行，F↔B为4行，N错误13行。',
        'E与F/B混淆合计44/61=72.1%，比笼统的“介词不同”更明确。', '',
        '## 逐组关系复核', '',
        '主类按最需要判断的信息分组、互不重复；带主观性，不是已确认的错误成因。context标记只是核验提醒，不能据此宣布不可学习。', '',
        '| 主类 | 原0.935错误行 | 当前错误行 | 当前组数 | 本次修好 | 本次破坏 | 净增错误 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, row in summary.iterrows():
        lines.append(f'| {name} | {row.old_error_rows} | {row.current_error_rows} | {int(row.current_groups)} | {row.fixed_rows} | {row.broken_rows} | {row.net_added_errors:+d} |')
    lines += [
        '', '这里“新闻场景”不是说模型只缺新闻词汇；其中常需要同一事件指称、被省略谓词或源图/源句的信息。',
        '修饰/结构是净增错误最大的主类（+6行），而当前错误最多的主类是新闻/场景指称（16行）。', '',
        '## 带介词的错误确实多，但不能全部归因于介词', '',
        f'按逐组PP框架标记：当前{facts["current_PP_frame_error_groups"]}组、{new_pp}/61行（{new_pp / 61:.1%}）含介词短语框架，包含嵌套的in May。',
        f'参考版同类{old_pp}行；本次破坏{pp_broken}行、修好{pp_fixed}行，净增{new_pp - old_pp}行。',
        'to deliver、to slide、to announce、to do中的to是不定式标记，没有因此计入PP。',
        '本次主类中归到“介词关系与路径角色”的是5个N单例；其他带PP的错误同时包含中心词、范围、指称和语境问题。这不是已证明的因果划分。', '',
        '| 例子 | 主要需要分辨什么 | 不能直接教什么 |',
        '|---|---|---|',
        '| over police crackdown / over police violence | 两侧over相同；不同事件称呼是否在源语境指同一事件 | 不能教成换介词问题，也不能把crackdown/violence普遍视为E |',
        '| on green grass / on a lawn | 两侧on相同；描述与场景指称 | 不能说换了介词；这组在当前模型已经正确 |',
        '| of some runway / from the airport | 部分/整体、两个地点名词与关系角色一起变化 | 不能只替换of/from便给固定标签 |',
        '| of wooden debris / of used wooden building materials | 两侧of相同；中心词、来源/状态限定与场景指称 | 不能直接教删used/building就必定E；一般木屑不都来自建筑 |',
        '| from a bird feeder / from the feeder | 保持from，判断bird限定是否缩小类别 | 适合研究自然PP中的范围限制 |',
        '| May Housing / Housing in May | 月份修饰位置与介词结构展开 | 有机会研究结构改写E，但需固定月份读法 |',
        '| into the camera / at the camera | 方向/目标与被省略动作 | 不知道是looking等何种谓词，不能把裸介词差异一律N |', '',
        '## 已经修好的旧例子不应继续作为当前必修清单', '',
        '当前已答对：on green grass/on a lawn，on rocks next to water/on a rocky seashore，convicted/get jail time，',
        'A woman/An adult female，is playing/is skijoring，以及其他若干旧错。',
        '这些仍可用于保留性检查，但不要按旧25组列表给每一类继续追加同样数量。尤其“事件状态”不再是本轮最大剩余错误簇。',
        '是否真实学到了这些关系，仍需后续稳定性/独立诊断，单次修好不等于已掌握。', '',
        '## 是否要造新的', '',
        '有研究同域片段增强的理由，但现在先不造，等copy运行完成。当前证据支持优先排查以下三种判别缺口，不支持把下一批全部改成介词换词：', '',
        '1. 自然短语中的范围限制与结构等价：同一PP内，真正增加限定的F/B和只换表达的E必须成对出现。当前净增最大。',
        '2. 明确对象下的动词改写与动作粒度：保留相同体貌和角色，避免pay off/set out/Opens等缺论元多义片段被机械赋标签。',
        '3. 关系角色与实体类型：位置/路径变化、人与机构、地点实体与处于地点的关系分开，不因表面共享词而一概蕴含。',
        '新闻同事件的指称、异常语法和疑似方向问题先核验源语境/Train约定，不拿官方Dev的单例反推通用规则。',
        '“子类型一致”应理解为覆盖当前缺失的判别关系，而不是复制Dev题材或介词外观；其他子类型也可能迁移，不能说不一致就必然无效。', '',
        '## copy结束后怎么用', '',
        '它提供相同行数、类别、组数和更新预算的对照。先看同一选模协议下的Dev、逐类正确数、fixed/broken及完整曲线。',
        '如果复制版明显优于手工版，会增加“当前合成内容或其分布不合适”的证据；单次运行仍不证明跨种子因果效果。',
        '如果复制版也明显低于参考峰值，优先查训练路径/阶段偏置，不能据此证明所有增强都无效。',
        '这轮不改模型、损失或数据。严格功能词零交集的要求继续保留：新文本无法直接使用Dev已出现的of/in/on/to等词，之后设计介词增强时需明确这一限制。', '',
        '## 文件', '',
        '`current_error_groups.csv`逐条列出当前37组；`current_error_rows.csv`为61行原始错例及标签。',
        '`error_type_groups.csv`包括原错、已修好和新增错共45组，保留每组复核理由和拟研究方向。',
        '`error_type_summary.csv`与`error_type_facts.json`保存统计、限制及输入哈希。',
    ]
    (OUTPUT_DIR / 'ERROR_TYPES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(confusion_types.to_string())
    print(summary.to_string())
    print('PP框架错误行：', old_pp, '->', new_pp, '；修好/破坏：', pp_fixed, '/', pp_broken)
    print('当前/参考错误组数：', facts['current_error_groups'], '/', facts['old_error_groups'])
    print('报告：', OUTPUT_DIR / 'ERROR_TYPES.md')


if __name__ == '__main__':
    main()
