"""本次 74 个错误组的人工初审笔记；不是重新标注的金标。"""

from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs/row_ce_frozen_epoch8'

# 现象、语境依赖候选、标签口径审核候选、初审理由。这里记录假设，不宣称根因已证实。
NOTES = {
    18: ('量词/改写', '可能', '是', 'all 的范围与 rallies/march 的数和改写同时变化；无完全匹配的公开源记录。'),
    87: ('指代/角色', '是', '是', 'his 指谁、该人的 security adviser 是否 national，必须借助原新闻。'),
    125: ('介词/修饰', '可能', '待审', 'in/through 改变空间或运动关系，overgrown/field 还增加限定，不能只按词数判断。'),
    190: ('数量/颜色', '可能', '待审', 'dogs/dog 与 brownish/tan 同时变化；需区分存在数量与精确数量解释。'),
    198: ('词汇上下位', '低', '低', 'chocolate bar/candy bar 的范围关系未稳定迁移；需核查词义地域约定。'),
    221: ('地点/语境', '是', '是', 'grassy field 不普遍蕴含 yard，原图片可能限定为同一处地点。'),
    249: ('事件/名词化', '是', '是', 'memorial/remembered 牵涉事件、词性和 its 指代，不能从两个孤立短语确定。'),
    375: ('动作范围', '可能', '待审', 'deliver statement/speak 的范围、书面或口头语境需要区分。'),
    432: ('属性不对称', '是', '是', 'black top + necklace 没有 big necklace 的大小信息；不满足简单加修饰词即 F。'),
    452: ('颜色/限定', '可能', '是', 'orange/brownish 并非严格同义，不能把全部差异都解释为添加 small。'),
    497: ('部分整体/介词', '是', '是', 'runway/airport 是部分整体，of/from 也不同；需原句事件角色。'),
    499: ('复杂描述/文本质量', '是', '是', 'cow-schemed 与 black and white 涉及描述性改写，第二个短语含 inside and，需核查切块质量。'),
    514: ('补充限定', '低', '低', 'bird feeder 是 feeder 的特定类型，模型一向把限定忽略而判 E。'),
    616: ('部分整体/角色', '是', '待审', 'TV/screen、实体与 of 短语混在一起；不是单一 wall-mounted 限定问题。'),
    641: ('事件/背景事实', '是', '是', 'death/murder 只有在 Bhutto 的具体事件背景中才接近等价。'),
    711: ('短语动词/多义', '是', '待审', 'pay off 不总是 bribe，必须选择对应词义。'),
    729: ('冠词/同义释义', '低', '低', 'Train 的 Woman/an adult female 正确；添加 A 后两向转为 B/F，另有逐因素探针。'),
    733: ('介词/词形改写', '可能', '待审', 'through/across 与 grass/grassy area 的联合替换；Train 有近邻但非严格单变量。'),
    895: ('介词/拼写', '是', '待审', 'fertiliser/fertilizer 拼写等价，但 of/in 的事件关系依赖新闻语境。'),
    936: ('名词化/词序', '低', '低', 'presidential election/election for president，整体语义不变但结构不同。'),
    943: ('短语动词/省略', '是', '待审', 'slide up/zipping up 缺少动作对象，zip 的词义需原句。'),
    948: ('事件/近义边界', '是', '是', 'crackdown 不必然是 violence；特定新闻事件可指同一事件。'),
    950: ('短语动词/多义', '是', '是', 'set out 在阐述计划语境可接近 announce，孤立时有多个义项。'),
    979: ('介词/事件角色', '是', '是', 'after/to 分别表达时间和目标，是否仍标 E 属于切块关系约定。'),
    980: ('事件/改写', '是', '待审', 'collapse/disaster 在 Bangladesh 的特定事件中共指；泛称 disaster 范围更宽。'),
    994: ('词序/名词短语', '可能', '低', 'May Housing/Housing in May 是时间修饰重排；需确认 May 为月份义。'),
    1069: ('事件/省略', '是', '是', 'granting asylum/for asylum 有事件动作与目标的差异，不能默认删词仍等价。'),
    1103: ('颜色同义/文本质量', '低', '待审', '猫毛色 ginger/orange 常可同义；A orange 还含冠词语法问题。'),
    1136: ('地点/语境', '是', '是', 'street/city path 不是普遍等价；可能由同一图片场景确定。'),
    1174: ('部分整体/改写', '是', '是', 'used wooden building materials 不必然是 debris，依赖具体废墟场景。'),
    1386: ('数量/方向', '可能', '待审', '7 Palestinians/Palestinian 涉及单复数和存在量词约定；原向 B、反向 F 需分开看。'),
    1456: ('动作上下位', '可能', '待审', 'skijoring 的词义及是否属于 playing 是知识和具体动作语境问题。'),
    1505: ('修饰范围/并列', '低', '低', 'Asian 是否修饰 woman 是决定性差别，表面词汇近似仍不能判 E。'),
    1551: ('方向/源关系', '是', '是', 'Saudi Women/Saudis 金标 B 与常见集合包含方向相反；源句两侧主语的角色不同。'),
    1562: ('地点/介词', '是', '是', 'from/in 与 collapsed 同时变化；地点、来源和事件背景不可混为一谈。'),
    1567: ('事件/切块', '是', '是', 'building toll 与 after building collapse 不是独立命题间可直接推出的关系。'),
    1590: ('序数/指代', '是', '待审', 'another term 依赖前任期语境，seventh 不能不经语境就完全替换 another。'),
    1659: ('数值/标注口径', '是', '是', '$25bn/$30 billion 金额不同；是否作下界、估值更新或事件共指需要解释。'),
    1781: ('补充限定', '低', '低', 'train station/station 是明确的限定关系，模型正向误判 E。'),
    1841: ('数量/方向', '可能', '待审', 'Several people/People 的方向需和任务的数量语义统一，当前两向都反了。'),
    1940: ('动作范围', '可能', '待审', 'having a meal 通常涉及 eat，但 eat 不一定构成一餐；模型倾向等价。'),
    1979: ('事件/相关', '是', '待审', 'Syria abuse 与 chemical arms 有主题联系，不能从相关直接推蕴含。'),
    2188: ('实体/角色', '是', '待审', 'Cardinals/Vatican 是人员、机构或地点关联，不是简单上下位。'),
    2242: ('词义/任务口径', '是', '是', 'Opens/begins operation 在开业语境可近义，但源 REL 被映为 N。'),
    2244: ('数量/省略', '是', '待审', 'two people/2 的单位和实体被省略；不能只看数字相同。'),
    2251: ('人物/组织', '低', '低', 'Supreme Guide 是人物角色、group 是组织；不能按共有名称判包含。'),
    2314: ('并列/对象', '低', '低', 'dining room 与 sofas + dining table 是房间和家具，部分词重合不等于蕴含。'),
    2406: ('事件/任务口径', '是', '是', 'train explosion/crash 可能是同一新闻事件，但不普遍互相推出。'),
    2436: ('词汇上下位/口径', '是', '是', 'daughter/child 存在词义包含，源 SIMI→N 与严格短语 NLI 有潜在冲突。'),
    2463: ('名词化/指代', '是', '是', "'s integration/to join 包含缺失的所有者、事件参与者和完成性差异。"),
    2487: ('地点限定/口径', '是', '是', 'Southern Italy/Italy 是范围关系，但源 SIMI→N，需核对事件共指与介词差异。'),
    2492: ('程度/范围', '是', '待审', 'Super 与 most powerful 的程度和比较集合不同，源句含 of 2013。'),
    2510: ('身份/背景事实', '是', '待审', 'Ferguson 与 Manager 的关系依赖人物和时间背景。'),
    2597: ('词汇上下位/口径', '可能', '是', 'jungle gym/play structure 是否属于稳定上下位需审核，源 SIMI 不直接等于严格无蕴含。'),
    2599: ('数量/介词', '可能', '待审', 'in/on、waves/wave、medium 同时变化，不是简单加限定。'),
    2613: ('地点/改写', '可能', '待审', 'shore/beach 部分近义，但 on beside the beach 有切块或语法异常。'),
    2616: ('介词/省略', '是', '待审', 'onto/to 的终点、运动和 another 的缺失对象需要原句。'),
    2671: ('多义/动作范围', '是', '待审', 'perform/do 需对象选择具体义项，孤立时不应直接等价。'),
    2738: ('因果/事件', '低', '低', 'convicted 不必然导致 jail time；不能把常见后果当同义。'),
    2792: ('数量/限定口径', '可能', '是', '2 Turkish unions/Unions 含数量和国别范围，源 SIMI→N 与常见蕴含约定有张力。'),
    2804: ('词汇包含/口径', '可能', '是', '2 Turkish Airlines pilots/airline pilots 常见集合解释为单向包含，源 SIMI→N 待审。'),
    2844: ('地点/口径', '是', '是', 'Northwest China/in China 有地点范围和句法角色差异，需任务口径。'),
    2898: ('模态', '可能', '是', 'may be trapped 不推出 trapped；模型预测 F 在严格语义下也不成立，逆向模态关系另需审核。'),
    2900: ('词汇上下位/口径', '可能', '是', 'senator 通常属于 lawmaker，但源 SIMI→N；不能当成纯词义缺失。'),
    2925: ('并列/省略', '是', '待审', 'network-sharing 与 spectrum deal 的并列结构和缺失中心词影响范围。'),
    2933: ('称谓/口径', '是', '是', 'lawmaker/MP 与 Nigeria/Nigerian 近义程度依赖职务和国家制度，源 SIMI→N。'),
    2959: ('实体/粒度', '是', '是', 'Tiananmen 与 Tiananmen Square 的指称粒度和分词异常需审核。'),
    3044: ('动作范围', '可能', '待审', 'pose 不必然站立，stand 不必然摆姿势；不能从常见共现推包含。'),
    3057: ('介词/源冲突', '是', '是', '相同短语对在两个源语境分别为 SIMI/EQUI；Train 的 track 单数近邻 N 答对。'),
    3063: ('属性/近邻迁移', '可能', '待审', 'Train crowded/city street 的 N 答对，Dev 换 public 后答 E；拥挤不保证公共。'),
    3200: ('介词/词义', '是', '待审', 'in a top hat/with a top hat 可能分别指穿戴和携带，需要具体义项。'),
    3241: ('省略/多义', '是', '待审', 'one/as one 的 one 可指数量、代词或整体方式，缺对象时不宜直接断方向。'),
    3264: ('词汇上下位/口径', '可能', '待审', 'truck/van 的分类因地域和车型变化，不能只用单一 WordNet 路径决定。'),
    3283: ('省略/词性', '可能', '待审', 'close-up view/close up 涉及名词、镜头类型与动词多义，需语境。'),
}


def main():
    dev = pd.read_csv(OUT / 'dev_enriched.csv', keep_default_na=False)
    matches = pd.read_csv(OUT / 'source_matches_with_context.csv', keep_default_na=False)
    error_ids = set(dev.loc[~dev['correct'], 'pair_id'])
    if {int(pid.rsplit('_', 1)[1]) for pid in error_ids} != set(NOTES):
        raise ValueError('人工笔记仅对应本次 epoch8 的 74 个错误组。')
    records = []
    for pair_id, group in dev.groupby('pair_id', sort=False):
        if pair_id not in error_ids:
            continue
        row = group.iloc[0]
        phenomenon, context, policy, note = NOTES[int(pair_id.rsplit('_', 1)[1])]
        source = matches[matches['instance_id'] == row['instance_id']].drop_duplicates(['context1', 'context2'])
        contexts = ['A: ' + r['context1'] + '\nB: ' + r['context2'] for _, r in source.iterrows()]
        records.append({'pair_id': pair_id, 'rows': len(group),
                        'independent_errors': int((~group['correct']).sum()),
                        'pair_aware_errors': int((~group['pair_correct']).sum()),
                        'text1': row['text1'], 'text2': row['text2'],
                        'gold_in_row_order': '|'.join(group['gold']),
                        'independent_in_row_order': '|'.join(group['prediction']),
                        'pair_aware_in_row_order': '|'.join(group['pair_prediction']),
                        'phenomenon': phenomenon, 'context_review_candidate': context,
                        'label_policy_review_candidate': policy, 'review_note': note,
                        'source_relations': row['source_relations'],
                        'source_contexts_partial': '\n---\n'.join(contexts),
                        'status': '一次研究性初审，未经第二标注者复核；没有修改金标'})
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'reviewed_error_groups.csv', index=False, encoding='utf-8-sig')
    print('初审错误组：', len(result), '；原始错误行：', result['independent_errors'].sum())
    print('语境审核“是”组数：', (result.context_review_candidate == '是').sum())
    print('口径审核“是”组数：', (result.label_policy_review_candidate == '是').sum())
    print('这些数量是初审候选数，不是已确认的标签错误率。')


if __name__ == '__main__':
    main()
