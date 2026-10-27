"""32组Train的语义与来源复核，只分析已有文件，不训练、不改标签。"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'outputs/run10_pair_nn/generalization_analysis'
SOURCE_DIR = ROOT / 'experiments/deberta_generalization_audit/sources'
SOURCE_AUDIT = ROOT / 'experiments/deberta_generalization_audit/outputs/row_ce_frozen_epoch8'
TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
SHORT = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
         'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
BUCKETS = {'A': '常规读法支持金标', 'B': '需要固定语境或读法', 'C': '字面判据与金标有张力，待核验'}
STATUS = 'Assistant post-hoc review with official gold visible; not blind or independently adjudicated.'
SOURCE_LABELS = {'FORW': 'F', 'BACK': 'B', 'EQUI': 'E', 'SIMI': 'N', 'REL': 'N', 'OPPO': 'N'}
PILOT_PAPER = 'https://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.423.pdf'
PHRASIS_PAPER = 'https://addi.ehu.es/bitstream/handle/10810/69010/PHRASE_SIMILARITY.pdf?isAllowed=n&sequence=7'

# phrase_candidates不是新训练标签。列出读法与理由，不写回任何官方CSV。
# A也不等于已经通过人工审核；这里只说明有清楚的诊断判据。
NOTES = {
    'review_01': ('B', 'B或N', 'sail读成泛指乘船行进时可能B；读成风帆航行时，kayaking不保证sailing。',
                  '官方多了目的介词to，公开近似源只有sails。词义与版本差异同时存在，不能用它教所有kayak都蕴含sail。',
                  '动词义项与目的补语', '先补齐读法和来源，暂不增强'),
    'review_02': ('A', 'N', '按字面区分水域中的位置与海滩位置，不假定两者重合。',
                  '在深蓝水域不保证在海滩，在海滩也不保证在水中。源句一侧游泳，一侧在沙地活动，支持分开理解。',
                  '位置关系', '保留为N诊断对照'),
    'review_03': ('C', 'N；宽松地形约定下可能B', '按通常大小区别理解hill和mountain，不预设二者包含关系。',
                  'hill推不出mountain。源标注经方向对齐后已经是FORW，官方F不是本次增强或反转程序新制造的。不能据此训练hill必然比mountain更具体。',
                  '地形类别与方向', '核验原标注，保留官方金标，不生成通用规则'),
    'review_04': ('B', 'E、F或N依racing义项而定', '把racing限定为在赛道高速行进，而非必然参加竞赛。',
                  '源语境两侧均为红色跑车在赛道上。若racing包含竞赛条件，单独driving quickly并不充分；若只是快速移动，E较合理。',
                  '动作义项与竞赛条件', '用明确主语和事件框架做诊断，不直接加入E增强'),
    'review_05': ('A', 'N', 'bar按金属杆件、railing按栏杆结构理解，不假设是同一物体。',
                  '杆件可能属于栏杆，但部分与整体名称不能无条件互换；在金属杆上不保证在栏杆上，反之亦然。',
                  '物体类别与部分整体', '保留为N诊断对照'),
    'review_06': ('A', 'N', 'grassy、dirt为不同限定，on、over也不直接互换。',
                  '两种属性和位置关系均不保证对方成立。源句分别涉及羊和越野车，不能用词面hill重合推出蕴含。',
                  '独立属性与空间关系', '保留为N诊断对照'),
    'review_07': ('A', 'B', 'stock固定为金融股票义，不读成存货。',
                  'Boeing stock是股票的具体类别，反向不成立。源语境确为波音股票。',
                  '公司限定与类别范围', '作为F/B限制对照锚点'),
    'review_08': ('C', 'F', '按26名新增巴勒斯坦囚犯属于巴勒斯坦囚犯的常规集合读法。',
                  '在存在性/限制读法下，具体数量描述支持一般描述。公开源原方向把Palestinian prisoners到26 more Palestinian prisoners标FORW，交换后得到官方B。源论文对数量另有广义量词解释；这条需按该约定核验，不能直接判错标或归罪导出程序。',
                  '数量限定与方向', '优先确认数量语义约定；不改金标、不自动扩写'),
    'review_09': ('A', 'N', 'Mexico与New Mexico按不同地名理解；后者不能拆开按Mexico匹配。',
                  '国家与另一地名修饰的人物称谓没有蕴含。源标题一侧墨西哥总统，一侧前新墨西哥州州长。',
                  '专名边界与实体类型', '优先构造专名边界诊断'),
    'review_10': ('C', 'F或E，依opposition的集合读法', 'opposition若指反对派群体，coalition增加组织形式；若同指同一组织，则可能E。',
                  '源语境两侧均涉及叙利亚反对派及代表资格，源SIMI映N。未找到能推广的纯短语规则使这条必为N，不能简单把正常限定关系都改成N。',
                  '群体限定与SIMI口径', '待核验，不作为硬负例自动加权'),
    'review_11': ('C', 'E', 'after、following均取时间在后；ski accident与skiing accident取滑雪事故义。',
                  '源语境两侧均是Schumacher滑雪事故后状态，公开源仍标SIMI并映N。这里的N与自然改写等价有明显张力，补回主体也未直接解决。',
                  '近义改写与SIMI/EQUI边界', '高优先级口径审核，不复制这种N规则'),
    'review_12': ('B', '字面N；同一场景近似读法下可能E', '需决定messy与crowded是否在描述同一桌面状态，computer desk与office desk是否同指。',
                  '源句均描述带电脑显示器的桌面，但两侧场景相近不等于严格信息等价。不能从源EQUI推出messy和crowded是通用同义词；也未据此证明两幅描述必为同一物体。',
                  '场景近似与修饰语', '先审核场景级E口径，暂不做定义式同义增强'),
    'review_13': ('C', 'F；cow泛指牛时可能E', 'cow若专指成年母牛，比cattle窄；若泛指牛则更接近等价。',
                  '两种常见读法均难直接支持N。源SIMI映N，需核验具体图景或标注政策，不能教模型cows和cattle互不蕴含。',
                  '词义范围与SIMI口径', '待核验，不把上位关系普遍标N'),
    'review_14': ('A', 'N', 'for Christmas取目的或节庆用途，in front of a Christmas tree取位置。',
                  '目的与空间位置不是同一语义角色。共享Christmas不构成两向蕴含。',
                  '目的与地点角色', '优先构造关系角色诊断'),
    'review_15': ('A', 'N', '一个是担任职务的人，一个是国家，不把转喻自动当类别蕴含。',
                  '总理属于某国不意味着总理这个对象就是国家。源句还涉及不同事件，标签N有清楚依据。',
                  '人物与国家实体类型', '优先构造实体类型诊断'),
    'review_16': ('A', 'E', 'seated取坐着的状态，are sitting也取坐姿状态，不强调时态或使役。',
                  '在这一明确读法下意义等价，适合作为普通词形/状态改写对照。但公开源没有找到完全一致的这对文本，不能冒称精确溯源。',
                  '姿态状态改写', '保留为E锚点；补来源前不做源语境推断'),
    'review_17': ('B', 'F或N', '需明确passenger bus是bus的同义说明，还是相对其他用途的额外类别限制。',
                  '若只是同义说明，red是单向限制；若颜色与用途是独立条件，则两边不可互推。源SIMI映N，不能仅凭长度断定F，也不能不说明用途就造N。',
                  '冗余定义与独立限制', '用明确独立属性和定义对照诊断'),
    'review_18': ('B', 'B或N', 'dressed可指穿着衣服或dressed in black，dressed up可指盛装或装扮。',
                  '若dressed泛指穿衣、dressed up指盛装，可支持B；源句一侧dressed in black、一侧儿童装扮，补语与义项被切掉。',
                  '短语动词与补语', '固定义项后再构造最小对照'),
    'review_19': ('C', 'E；male非人类读法下可能B', '在人类读法下，adult male与man通常接近定义等价。',
                  '源照片语境也指成年男性，但源FORW给出官方F。是否将man视为年龄不确定等需核验，不能据此宣称普通定义等价应普遍标单向。',
                  '定义等价与词义粒度', '核验判据，暂不扩写这条方向规则'),
    'review_20': ('B', 'F、E或N依主体和义项', '驾驶者driving、乘客riding与车辆自身运动应分开。',
                  '公开源两侧主语都是train，而不是一位司机和一位乘客。孤立动词没有主语，不能固定套用驾驶必然乘坐，或两词永远等价。',
                  '施事角色与车辆运动', '诊断时显式补主语，区分人和车辆'),
    'review_21': ('B', '金融读法下可E；一般读法不确定', 'close取收盘，end up取交易结束时上涨，不把close读成关门。',
                  '源为close 0.54 pct higher与end up 0.11 pct；金融补语决定含义。原始短语本身省略了这些信息。',
                  '金融义项与省略补语', '只在显式金融框架下研究E，不教close=end up通用规则'),
    'review_22': ('B', '无法可靠定向', '第二段缺少谓词或标点，参与者、数量与伤亡角色不清。',
                  '近似公开源为Afghan police kill 5 Taliban fighters对另一句双方伤亡描述；提取的非连续chunk漏掉kill。官方还把另一侧逗号改成and。不能把近似源当精确匹配，也不能只按词集大小定向。',
                  '非连续切块与事件角色', '先核验源版本和切块，暂不增强'),
    'review_23': ('A', 'E', '两侧取同一跑步动作，暂不区分进行体和裸动词形式。',
                  '这是任务采用的常见动作词形等价读法；不能推广为所有时态差异都不改变信息。',
                  '动作词形改写', '作为E锚点，固定动作读法'),
    'review_24': ('B', '字面N；特定同指读法可E', '金属带和金属杆一般不是同义类别；需知道是否同一栖息支架。',
                  '源句两侧都是鸟栖在金属物上，源EQUI可能采用场景级同指/近似。上下文只提供这种可能，不能独立证明物体相同。',
                  '物体同指与材料形状', '保留为语境研究，不造band=bar通用同义规则'),
    'review_25': ('A', 'B', 'young woman是woman增加年龄限制，其他对象条件相同。',
                  '一向成立、另一向不保证。找到两个精确公开源命中，方向均为BACK。',
                  '年龄属性限制', '作为F/B限制对照锚点'),
    'review_26': ('B', 'E或N依场景', 'at a table可以是桌边活动，beside a table只表示在旁边。',
                  '一些场景中两者描述同一位置，但一般不能互推。没有找到完全一致的公开源对，不能用别的at/by/table对替代确认。',
                  '空间关系与活动框架', '先核验来源；使用明确几何/活动条件作诊断'),
    'review_27': ('A', 'N', '把6.3和6当精确数值，不自行引入四舍五入或至少6的范围义。',
                  '精确数值不同不能互推；源语境还分别提到Indonesia和Thailand。若写成at least 6则标签会不同，应在诊断里区分。',
                  '精确数值与区间', '优先构造数值精度诊断'),
    'review_28': ('A', 'N', '穿过正在飞溅的水，与某个主体溅水，是路径/环境和动作的区别。',
                  '前者不保证主体造成飞溅，后者不保证穿越该路径。源句walking through water与dog splashes也不能仅凭water词面判断单向。',
                  '动作与路径角色', '优先构造路径/动作角色诊断'),
    'review_29': ('A', 'N', 'of the ramp表示与斜坡的补足关系，at a skater park表示地点，不补造隐含必然联系。',
                  '坡道不一定在滑板公园，公园内的位置也不一定是坡道相关位置。源句涉及坡顶与pipe wall，支持保持角色差别。',
                  '部分补语与地点', '保留为角色差异N对照'),
    'review_30': ('A', 'B', '两侧均按突尼斯政治反对派人物理解，leader是更具体职务。',
                  '政治反对派领袖可视为反对派政治人物，反向不保证。国家形容词和国家名修饰的差异不应掩盖职务限制。',
                  '政治职务层级', '优先构造固定政治场景的职务范围诊断'),
    'review_31': ('B', '金融读法下可E；字面不保证', 'takes a beating取股价大跌，tumbles取股价下挫，不读成身体挨打和摔倒。',
                  '源语境明确为Boeing股票。词典式beating=tumbling会教错其他场景，金融主体应显式出现。',
                  '金融习语与主体', '限定金融框架后作诊断，不无条件增广'),
    'review_32': ('C', 'B或E依称谓/同指读法', '若两侧均指里根总统的前发言人，Larry Speakes增加具体身份；若已同指本人则可能E。',
                  '源语境均为该前发言人逝世新闻，源SIMI仍映N。这里只能标为任务口径张力，不能教加入人名必然变N，也不能自行改官方标签。',
                  '专名、职务与同指', '核验SIMI口径，不作为自动硬负例'),
}

MODEL_PATHS = {
    'original_0p938': 'experiments/deberta_manual_aug/outputs/run06_relation48_staged/error_analysis/train_readout.csv',
    'nn_epoch2_diagnostic': 'experiments/deberta_manual_aug/outputs/run10_pair_nn/epoch_2/after_pair/train/predictions_with_scores.csv',
    'n_head_best': 'experiments/deberta_frozen_n_head/outputs/run01_n_head/best/train/predictions_with_scores.csv',
    'top2_keep_best': 'experiments/deberta_frozen_n_head/outputs/run02_top2_keep/best/train/predictions_with_scores.csv',
    'top2_row_best': 'experiments/deberta_frozen_n_head/outputs/run03_top2_row_only/best/train/predictions_with_scores.csv',
}


def norm(text):
    text = unicodedata.normalize('NFKC', str(text)).replace('’', "'").replace('‘', "'")
    return ' '.join(text.casefold().split())


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def source_evidence(key):
    all_matches = pd.read_csv(SOURCE_AUDIT / 'source_matches_with_context.csv', keep_default_na=False)
    # 只取这32个官方Train实例，不读取Dev预测或挑选Dev错误。
    matches = all_matches[all_matches['split'].eq('train') & all_matches.instance_id.isin(key.instance_id)].copy()
    public = {path.name: list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'))
              for path in SOURCE_DIR.glob('PhrasIS.*.*.all.txt')}
    records = []
    key_by_id = key.set_index('instance_id')
    for hit in matches.itertuples(index=False):
        row = key_by_id.loc[hit.instance_id]
        fields = [value.strip() for value in public[hit.source_file][hit.source_line - 1]]
        first, second = fields[2], fields[3]
        if hit.matched_swapped:
            first, second = second, first
        assert norm(first) == norm(row.text1) and norm(second) == norm(row.text2)
        relation = fields[1]
        if hit.matched_swapped:
            relation = {'FORW': 'BACK', 'BACK': 'FORW'}.get(relation, relation)
        assert relation == hit.oriented_source_relation and SOURCE_LABELS[relation] == row.gold
        records.append({
            'review_id': row.review_id, 'instance_id': hit.instance_id,
            'source_file': hit.source_file, 'source_line': hit.source_line, 'context_id': hit.context_id,
            'raw_relation': fields[1], 'oriented_relation': hit.oriented_source_relation,
            'source_score': hit.source_score, 'matched_swapped': hit.matched_swapped,
            'raw_text1': fields[2], 'raw_text2': fields[3], 'raw_positions1': fields[4], 'raw_positions2': fields[5],
            'context1_partial': hit.context1, 'context2_partial': hit.context2,
            'context_slots_conflict': hit.context_has_conflicting_slots,
            'source_url': f'https://github.com/Jonapa/assessing_logical_coherence/blob/main/data/{hit.source_file}#L{hit.source_line}',
            'match_status': 'exact_normalized_phrase_pair',
        })
    sources = pd.DataFrame(records)
    save(sources, 'train_review_sources_verified.csv')

    # 只列两个可追踪的近似候选。其余未匹配案例保持缺失，不用模糊检索充当溯源。
    contexts = pd.read_csv(SOURCE_AUDIT / 'source_reconstructed_contexts.csv', keep_default_na=False)
    candidates = []
    for review_id, filename, number, swapped, difference in [
        ('review_01', 'PhrasIS.test.images.all.txt', 823, False, '官方sails to，候选只有sails；多出的to未解释。'),
        ('review_22', 'PhrasIS.train.headlines.all.txt', 2666, True, '官方and对应源逗号；另一侧是非连续提取，kill被排除。'),
    ]:
        fields = [value.strip() for value in public[filename][number - 1]]
        cid = filename + ':' + fields[6]
        sides = contexts[contexts.context_id.eq(cid)].set_index('side')
        candidates.append({
            'review_id': review_id, 'source_file': filename, 'source_line': number,
            'raw_relation': fields[1], 'raw_text1': fields[2], 'raw_text2': fields[3],
            'candidate_swapped': swapped, 'context_id': cid,
            'candidate_context1': sides.loc[int(swapped), 'reconstructed_context'],
            'candidate_context2': sides.loc[int(not swapped), 'reconstructed_context'],
            'difference': difference, 'status': 'near-source candidate only; identity not established',
        })
    save(pd.DataFrame(candidates), 'train_review_source_candidates.csv')
    return sources


def model_comparison(key, review):
    values = key[['review_id', 'instance_id', 'gold', 'prediction', 'review_role']].rename(columns={'prediction': 'old_oof'})
    frames = {}
    for name, relative in MODEL_PATHS.items():
        frame = pd.read_csv(ROOT / relative, keep_default_na=False).set_index('instance_id')
        selected = frame.loc[key.instance_id]
        assert selected.text1.tolist() == key.text1.tolist() and selected.text2.tolist() == key.text2.tolist()
        values[name + '_independent'] = selected.prediction.tolist()
        values[name + '_pair_aware'] = selected.pair_prediction.tolist()
        frames[name] = frame
    values = values.merge(review[['review_id', 'bucket', 'family']], on='review_id', validate='one_to_one')
    save(values, 'train_review_models.csv')
    summaries = []
    for bucket, group in values.groupby('bucket'):
        columns = ['old_oof'] + [name + suffix for name in MODEL_PATHS for suffix in ['_independent', '_pair_aware']]
        for column in columns:
            summaries.append({'bucket': bucket, 'model_readout': column, 'review_groups': len(group),
                              'correct_representatives': int(group[column].eq(group.gold).sum())})
    save(pd.DataFrame(summaries), 'train_review_model_summary.csv')
    return values, frames


def existing_neighbours(train, review, frames):
    # 只是寻找现有Train表面类比，不生成新文本，不把近邻称作受控最小对照。
    train = train.copy().reset_index(drop=True)
    train['gold'] = train.label.map(SHORT)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(train.text1 + ' || ' + train.text2)
    queries = vectorizer.transform(review.text1 + ' || ' + review.text2)
    scores = (queries @ matrix.T).toarray()
    excluded_groups = set(review.pair_id)
    records = []
    current = frames['top2_row_best']
    for index, query in review.iterrows():
        seen = set()
        for selected in scores[index].argsort()[::-1]:
            row = train.iloc[selected]
            if row.pair_id in excluded_groups or row.pair_id in seen:
                continue
            seen.add(row.pair_id)
            records.append({'review_id': query.review_id, 'rank': len(seen),
                            'neighbour_instance_id': row.instance_id, 'neighbour_pair_id': row.pair_id,
                            'text1': row.text1, 'text2': row.text2, 'gold': row.gold,
                            'current_train_prediction': current.loc[row.instance_id, 'prediction'],
                            'char_similarity': float(scores[index, selected]),
                            'scope': 'Existing Train surface analogy; not a semantic minimal edit or label-conflict proof'})
            if len(seen) == 3:
                break
    neighbours = pd.DataFrame(records)
    save(neighbours, 'train_review_existing_neighbours.csv')
    return neighbours


def screen_prefixes(train, frames):
    # 机械筛选现有Train，不合成新句子。reference_reading只是显式读法下的对照，不改金标。
    numbers = set('two three four five six seven eight nine ten eleven twelve'.split())
    attributes = set('young red black white blue brown green adult little small large old dead wooden metal big tall yellow'.split())
    prepositions = set('of in on at to from for with by over under after before'.split())

    def tokens(text):
        terms = re.findall(r'[a-z]+|[0-9]+', norm(text))
        while terms and terms[0] in {'a', 'an', 'the'}:
            terms = terms[1:]
        return terms

    records = []
    # 每个官方pair_id只取首行，反向行不重复计数。
    for row in train.drop_duplicates('pair_id').itertuples(index=False):
        first, second = tokens(row.text1), tokens(row.text2)
        for side, longer, shorter in [(1, first, second), (2, second, first)]:
            if not shorter or len(longer) != len(shorter) + 1 or longer[1:] != shorter:
                continue
            prefix = longer[0]
            is_count = prefix in numbers or (prefix.isdigit() and int(prefix) > 1)
            if is_count and shorter[0] not in prepositions:
                kind = 'number_prefix'
            elif prefix in attributes:
                kind = 'attribute_prefix'
            else:
                continue
            reference = 'F' if side == 1 else 'B'
            record = {'instance_id': row.instance_id, 'pair_id': row.pair_id,
                      'text1': row.text1, 'text2': row.text2, 'gold': SHORT[row.label],
                      'screen': kind, 'prefix': prefix, 'prefix_side': side,
                      'reference_reading': reference,
                      'reference_assumption': 'existential/restrictive reading with a shared object category; not an adjudicated task label',
                      'gold_agrees_with_reference': SHORT[row.label] == reference}
            for name, frame in frames.items():
                record[name + '_independent'] = frame.loc[row.instance_id, 'prediction']
                record[name + '_pair_aware'] = frame.loc[row.instance_id, 'pair_prediction']
            records.append(record)
    screened = pd.DataFrame(records)
    assert screened.groupby('screen').size().to_dict() == {'attribute_prefix': 45, 'number_prefix': 12}
    oof = pd.read_csv(ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/oof01_train_only/train_oof_scores.csv',
                      keep_default_na=False).set_index('instance_id')
    screened['old_oof_prediction'] = oof.loc[screened.instance_id, 'prediction'].tolist()
    for kind in ['number_prefix', 'attribute_prefix']:
        save(screened[screened.screen.eq(kind)], 'train_review_' + kind + '.csv')

    # 对数量组逐行验证公开源原始文本、方向反转和官方标签，不只相信旧的映射CSV。
    matches = pd.read_csv(SOURCE_AUDIT / 'source_matches_with_context.csv', keep_default_na=False)
    numeric = screened[screened.screen.eq('number_prefix')].set_index('instance_id')
    matches = matches[matches['split'].eq('train') & matches.instance_id.isin(numeric.index)]
    public = {path.name: list(csv.reader(path.read_text(encoding='utf-8-sig').splitlines(), delimiter='\t'))
              for path in SOURCE_DIR.glob('PhrasIS.*.*.all.txt')}
    sources = []
    for hit in matches.itertuples(index=False):
        fields = [value.strip() for value in public[hit.source_file][hit.source_line - 1]]
        row = numeric.loc[hit.instance_id]
        first, second, relation = fields[2], fields[3], fields[1]
        if hit.matched_swapped:
            first, second = second, first
            relation = {'FORW': 'BACK', 'BACK': 'FORW'}.get(relation, relation)
        assert (norm(first), norm(second)) == (norm(row.text1), norm(row.text2))
        assert relation == hit.oriented_source_relation and SOURCE_LABELS[relation] == row.gold
        sources.append({'pair_id': row.pair_id, 'instance_id': hit.instance_id,
                        'text1': row.text1, 'text2': row.text2, 'official_gold': row.gold,
                        'raw_text1': fields[2], 'raw_text2': fields[3], 'raw_relation': fields[1],
                        'source_file': hit.source_file, 'source_line': hit.source_line,
                        'matched_swapped': hit.matched_swapped, 'oriented_relation': relation,
                        'context1_partial': hit.context1, 'context2_partial': hit.context2,
                        'source_url': f'https://github.com/Jonapa/assessing_logical_coherence/blob/main/data/{hit.source_file}#L{hit.source_line}'})
    source_frame = pd.DataFrame(sources)
    assert source_frame.pair_id.nunique() == 12
    save(source_frame, 'train_review_number_sources.csv')
    summaries = []
    for kind, group in screened.groupby('screen'):
        columns = ['old_oof_prediction'] + [name + suffix for name in MODEL_PATHS
                   for suffix in ['_independent', '_pair_aware']]
        for column in columns:
            summaries.append({'screen': kind, 'groups': len(group), 'model_readout': column,
                              'official_gold_correct': int(group[column].eq(group.gold).sum()),
                              'reference_reading_agreement': int(group[column].eq(group.reference_reading).sum())})
    save(pd.DataFrame(summaries), 'train_review_prefix_models.csv')
    numbers_frame = screened[screened.screen.eq('number_prefix')]
    return {'number_groups': len(numbers_frame),
            'number_reference_agreements': int(numbers_frame.gold_agrees_with_reference.sum()),
            'number_distinct_normalized_unordered_pairs': len({tuple(sorted((norm(row.text1), norm(row.text2))))
                                                              for row in numbers_frame.itertuples(index=False)}),
            'number_verified_source_hits': len(source_frame),
            'attribute_groups': int(screened.screen.eq('attribute_prefix').sum()),
            'attribute_reference_agreements': int(screened.loc[screened.screen.eq('attribute_prefix'), 'gold_agrees_with_reference'].sum()),
            'screen_rules': {'token_pattern': '[a-z]+|[0-9]+', 'removed': 'leading a/an/the only',
                             'number_prefix': 'integer >1 or English two through twelve; one extra leading token; remaining tokens identical; preposition core excluded',
                             'attributes': sorted(attributes), 'excluded_core_prepositions': sorted(prepositions),
                             'unit': 'first row per official Train pair_id; no Dev queries'},
            'interpretation': 'Observed source-inherited label convention tension under the stated reading; not 12 confirmed annotation errors',
            'quantity_guideline': 'Pilot paper Appendix A.1 explicitly treats quantification/plurality using generalized-quantifier denotations; it does not adjudicate these 12 pairs',
            'quantity_guideline_url': PILOT_PAPER}


def exact_input_conflicts(train):
    # 严格区分“同输入不同标签”与上面的语义/口径疑点。
    snapshots = {'official_3042': train,
                 'current_3290': pd.read_csv(ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv',
                                           keep_default_na=False)}
    records, summaries = [], {}
    for name, frame in snapshots.items():
        frame = frame.copy()
        frame['input_key'] = [json.dumps([norm(first), norm(second)], ensure_ascii=False)
                              for first, second in zip(frame.text1, frame.text2)]
        labels = frame.groupby('input_key').label.nunique()
        bad = set(labels[labels.gt(1)].index)
        selected = frame[frame.input_key.isin(bad)]
        summaries[name] = {'rows': len(frame), 'conflicting_ordered_input_keys': len(bad),
                           'affected_rows': len(selected)}
        for row in selected.itertuples(index=False):
            records.append({'dataset': name, 'instance_id': row.instance_id, 'pair_id': row.pair_id,
                            'text1': row.text1, 'text2': row.text2, 'gold': SHORT[row.label],
                            'normalized_ordered_input': row.input_key})
    save(pd.DataFrame(records), 'train_review_exact_input_conflicts.csv')
    return summaries


def lexical_isolation_cost(train):
    # Dev只读文本用于用户要求的拒绝词表，不读标签、不按Dev错误挑规则。
    dev_path = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
    dev = pd.read_csv(dev_path, usecols=['text1', 'text2'], keep_default_na=False)
    extra = pd.read_csv(ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged/extra_train_rows.csv',
                        keep_default_na=False)

    def vocabulary(frame):
        return {word for column in ['text1', 'text2'] for text in frame[column]
                for word in re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', str(text)).casefold())}

    def row_words(row):
        return set(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', row.text1 + ' ' + row.text2).casefold()))

    train_vocab, dev_vocab, extra_vocab = vocabulary(train), vocabulary(dev), vocabulary(extra)
    stemmer = PorterStemmer()
    stems = {word: stemmer.stem(word) for word in train_vocab | dev_vocab | extra_vocab}
    dev_stems = {stems[word] for word in dev_vocab}
    allowed = {word for word in train_vocab if word not in dev_vocab and stems[word] not in dev_stems}
    markers = set('a an the of in on at to from for with by over under after before'.split()) & dev_vocab
    summaries = []
    for name, frame in [('official_3042', train), ('new_relation48', extra)]:
        terms = [row_words(row) for row in frame.itertuples(index=False)]
        summaries.append({'dataset': name, 'rows': len(frame), 'vocabulary': len(vocabulary(frame)),
                          'rows_passing_dev_word_and_stem_exclusion': sum(
                              not (words & dev_vocab) and not ({stems[word] for word in words} & dev_stems)
                              for words in terms),
                          'rows_containing_blocked_grammar_markers': sum(bool(words & markers) for words in terms)})
    save(pd.DataFrame(summaries), 'train_review_lexical_isolation_cost.csv')
    return {'train_vocabulary': len(train_vocab), 'dev_vocabulary': len(dev_vocab),
            'allowed_train_words': len(allowed), 'blocked_grammar_markers': sorted(markers),
            'datasets': summaries, 'dev_use': 'text-only rejection blacklist; labels not loaded',
            'interpretation': 'Measured restriction on expression choices; not a causal estimate of its effect on Dev F1',
            'policy_changed': False}


def write_report(review, sources, models, facts):
    lines = [
        '# 32组Train审核：输入信息、普通语义与任务标注口径',
        '',
        '审核日期：2026-10-23。只读取已有Train审核表、公开源语境及已导出的Train预测。没有训练、改官方标签或生成新增强行。',
        '',
        '**这不是盲审或独立人工仲裁。** 助手复核时已经看到官方金标，下面的候选判断是带读法条件的分析，不是可直接训练的新标签。',
        '',
        '## 主要结论',
        '',
        f'- A：{facts["bucket_counts"]["A"]}组，明确的常规读法可以支持官方标签。',
        f'- B：{facts["bucket_counts"]["B"]}组，存在词义、主体/补语、指称或来源版本问题，需要明确读法；不等于官方允许用原句替短语定标。',
        f'- C：{facts["bucket_counts"]["C"]}组，普通短语判据与金标存在张力，列为待核验，未宣布错标。',
        '- 审核题是24组极难OOF错误加8组很容易的OOF正确对照，不是随机样本。上述比例不能当作全Train标签噪声率、不可学比例或模型上限。',
        '',
        '因此，不能把这些题都解释成缺词义，也不能把模型按普通语义作出的回答一律当作能力不足。反过来，发现疑点也不能证明所有剩余错误都来自数据。',
        '',
        '## 原始标注协议对本次分析的修正',
        '',
        f'[PhrasIS原始论文第2页]({PHRASIS_PAPER})区分了iSTS的语境标注与PhrasIS的重新标注：PhrasIS要求对齐短语脱离原句语境判定。原文说明为“relabeled all aligned pairs independently of the context of occurrence”。',
        '所以，查到close/end up来自金融新闻，只能提供一种候选读法，不能证明官方E标签是按金融原句判出来的，也不能宣布给模型补原句就是合法修复。本报告所有源语境解释都属于事后读法分析，不是标注协议的证据。',
        'B桶应读作“短语读法有待澄清”，不能读作“这11组已被证明因缺原句而不可学”。同样，不能从这32个选择性案例估计整个任务的不可学比例。',
        '',
        '## 确定的输入冲突：与语义疑点分开计数',
        '',
        f'对官方3042行及当前3290行分别做归一化有序文本对检查，两份数据各有{facts["exact_input_conflicts"]["official_3042"]["conflicting_ordered_input_keys"]}个同输入不同标签的键：South Africans / South Africa，同时标F和N，各对应一行。',
        '它已经存在于官方Train，不是本轮增强新增的。对只读取text1/text2的确定性分类器，同一输入无法同时满足两个不同金标。后续应把它单列为冲突项，不将这一项的训练错误解释为模型没学会；本次没有删除或改写它。',
        '这个冲突数量很小，不能解释整个Train/Dev差距。C桶的7组也不等于7个这种确定冲突：C多数仍属于读法或任务政策待核验。',
        '',
        '## 新发现：数量和普通属性限定的方向不能混作一个模板',
        '',
        f'从官方Train机械筛选“除冠词外只多一个数量前缀、其余词完全相同”的组：{facts["prefix_audit"]["number_groups"]}组，按存在性/限制读法推导的方向与金标一致{facts["prefix_audit"]["number_reference_agreements"]}组。对照筛出的普通属性前缀组：{facts["prefix_audit"]["attribute_groups"]}组，一致{facts["prefix_audit"]["attribute_reference_agreements"]}组。',
        '',
        '| Train现有文本 | 官方 | 存在性/限制读法下的对照 |',
        '|---|---|---|',
        '| The dogs / The two dogs | F | B |',
        '| Four children / Children | B | F |',
        '| 26 Palestinian Prisoners / Palestinian prisoners | B | F |',
        '| A woman / A young woman | B | B |',
        '',
        f'数量12组均逐一核对公开源，{facts["prefix_audit"]["number_verified_source_hits"]}条源记录中的文本和方向映射与官方金标一致。这说明不是我们最近的数据增强或反转代码新制造的方向差异。按归一化无序文本对计，12个pair_id实际只有{facts["prefix_audit"]["number_distinct_normalized_unordered_pairs"]}种输入；含重复，并非12次独立证据。',
        '',
        f'**尚不能宣布12组都是错标。** [来源任务的论文附录A.1]({PILOT_PAPER}#page=13)明确对数量、复数采用广义量词的解释，而不是直接套用普通属性的集合限制。论文没有逐一解释这12组的数量方向，因此具体适用规则仍待澄清。这是标注约定核验问题，不能通过把所有F/B翻转解决。',
        '',
        '筛选去掉开头a/an/the，按[a-z]+|[0-9]+分词，只比较单个前缀增加。数量限定为大于1的整数或two至twelve，排除核心以介词开头的条目；属性使用固定词表中的18个词。完整规则见facts.json。这是局部结构筛选，不是全Train无噪声证明，也不涉及Dev上的数量分布。',
        '',
        '下一批数量诊断/增强先分开“名词短语按广义量词比较”与“明确存在/至少/恰好”的完整断言，不直接把两者用同一条“加词=更窄”模板生成。先确认标签解释，再给新例子定标。',
        '',
        '## 来源证据',
        '',
        f'{facts["exact_source_groups"]}/32组找到归一化后完全一致的公开源文本，共{len(sources)}条命中；review_25有两个命中。',
        'review_01、16、22、26没有找到精确源对。只为01和22列了近似候选，不能继承候选的上下文/标签作为精确溯源。16和26保持来源未解决。',
        '源语境是同一parent的词位拼接重建，可能有[GAP]、遗漏尾部，不能当作完整原句或已经找到了对应图片。即使两个场景相似，也不独立证明两个描述同指。',
        'PhrasIS.test是公开旧源任务的分区名称，不是隐藏的DiCo测试集；这32个审核实例本身全部来自官方DiCo Train。公开源语境只用于审计，没有加入训练。',
        '',
        '## 最值得核验的标注张力',
        '',
        '| 案例 | 官方 | 普通读法与证据 |',
        '|---|---|---|',
        '| 26 more Palestinian prisoners / Palestinian prisoners | B | 数量限定通常支持F；精确源原方向已经标FORW，交换得到B。疑点继承自源关系，不是本次增强新写反。 |',
        '| after ski accident / following skiing accident | N | 常规时间改写接近E，源也均谈Schumacher事故；原SIMI映成N。 |',
        '| of cows / of cattle | N | 严格母牛义通常支持F，泛指牛时可能E；源SIMI仍映N。 |',
        '| opposition coalition / opposition | N | 群体限定可支持F、同指可能E；源SIMI需要核验任务口径。 |',
        '',
        '这些例子不应被自动改标签或放大权重。尤其不能新增“after和following永远不等价”“cows不属于cattle”的通用监督；数量则必须先说明采用哪种语义解释。',
        '',
        '## 有上下文才好确定的E',
        '',
        '- driving quickly / racing：源为赛道红色跑车，但racing是快速移动还是参加竞赛会改变关系。',
        '- close / end up：源为金融收盘及上涨幅度；孤立close可以是关闭，end up也有其他读法。',
        '- takes a beating / tumbles：源为波音股票下跌；不是说挨打与摔倒普遍等价。',
        '- on a metal band / on a metal bar：源描述鸟的栖息物，E可能依赖特定物体读法；不能创造band=bar词典规则。',
        '',
        '审核中六个被选中的OOF困难E均落在B；另两个简单E对照落在A。这由选题和本次读法复核共同决定，不能代表所有E。它说明“这些难E缺少词典同义词”并不是充分解释。',
        '',
        '## 看过Train与没看过Train的区别如何读',
        '',
        'train_review_models.csv并列旧OOF与5组已有模型读出。当前全Train模型在这32组上见过训练金标；它的正确不能当作新表达迁移。旧OOF与这些模型结构/训练配方也不同，差异不是单一的是否见过样本效应。',
        '',
    ]
    for bucket in ['A', 'B', 'C']:
        subset = models[models.bucket.eq(bucket)]
        ind = int(subset.top2_row_best_independent.eq(subset.gold).sum())
        pair = int(subset.top2_row_best_pair_aware.eq(subset.gold).sum())
        lines.append(f'- {bucket}组：当前run03最佳模型代表行独立{ind}/{len(subset)}、结构{pair}/{len(subset)}。这些是Train上的已见样本，不是Dev分数。')
    lines += [
        '',
        '特别是after ski accident/following skiing accident，在这些不同全Train模型中仍统一倾向E。它不是把Pair约束再加大就一定能补上的错误；先确认N的任务判据更有信息量。',
        'run03在32个已见代表行中独立错6个：review_03/10/11/17/21/28；结构解码修好03和21后仍错4个。剩下4个包括2个C口径疑点、1个B读法依赖和1个A清楚的角色区分错误，不能统一当成同一种难负例处理。',
        '在额外筛出的45组属性限定上，旧OOF正确45/45；12组数量上，按官方金标正确10/12。当前run03两批已见代表行均全对。它说明这些局部标签模式并非完全学不进模型，但不是新词/新结构迁移的证明，更不能据此把12组数量都宣布为错误。',
        '',
        '## 新增强的表达限制：一个可量化的分布问题',
        '',
        '按用户既定的新增词形/词干与Dev零交集规则重新检查：官方Train的3042行只有112行能通过；2227个Train词形中1565个仍可用。也就是说，不是所有词都用不了，而是包含常见功能词的一般短语大多不能照原风格构造。',
        'Dev拒绝词表包括a/an/the/in/on/at/of/to/with等。官方Train有2033/3042行包含被排除的这组语法词，当前新增48行是0/48。新增48行实际只使用38种词形。',
        '这证明限制明显改变了可用表达形式，不证明它单独造成了掉分。换用under/among/during等不重合的表达可以满足隔离，但不能把“关键词不重合”当成“增强仍与官方文本同分布”。本次保留隔离规则，没有放宽它。',
        '普通训练/验证共享词汇本身不等于标签泄露；当前更严格的词汇隔离是用户选择的实验约束。另一方面，过去增强的模式曾受Dev错误分析影响，词汇零交集也不能抹掉这种开发过程依赖。应如实披露二者。',
        '',
        '## 下一步：先做诊断，不立即再训练',
        '',
        '保留0.938版本作为已有结构系统，同时保留混合训练/现有头模型的独立读出作对照。不要把这次审核变成再次扫描Pair权重、N偏置、解冻层数或SWA窗口；近期N头、解冻顶部2层和去掉保护KL没有达到同轮independent≥0.87、pair-aware≥0.94的目标。',
        '从A中6个旧OOF错误出发，优先验证专名边界、人物/国家类型、目的/地点角色、动作/路径角色、固定政治场景的职务层级。精确数字与范围另列，先固定数量语义约定。用A中的E和F/B正确对照作为锚点，避免只补N导致整体偏置。',
        '新诊断应显式固定读法，区分真正限制、保义改写与独立属性；不要求每个含糊词都硬凑齐四个标签。B可用于研究有/无显式主体或补语时判断如何改变，但添加语境后的文本必须重新判断标签，不能继承原官方标签。C暂时只做口径核验，不纳入新监督。',
        '这次尚未生成这些新诊断文本。之后若构造，会继续执行用户要求的新词形、词干、整短语和无序文本对与Dev零交集，并排除现有Train完全相同/反向相同的文本。旧Train审核材料本身不宣称与Dev词汇隔离。',
        '当前只有已见Train预测、旧OOF和现有文本近邻，还没有同一模型在已审核保义/变义最小对照上的实测。因此“换词或换结构必然导致错误”尚未被当前32组审核严格验证，下一步的独立诊断应补的正是这块证据。',
        '先用已保存的模型预测相同诊断，再决定是否制作对照增强；新诊断不能一边参与训练一边宣称迁移。审核标注未独立仲裁前，不自动把C类删除、改标或用自信度提高权重。',
        '',
        '## 需要澄清的三个具体判据',
        '',
        '1. The dogs/The two dogs这类短语，应采用哪种广义量词/数量读法？能否提供与普通形容词限定相对照的示例？',
        '2. after ski accident/following skiing accident、of cows/of cattle的SIMI→N，依据是不同词义、不同指称，还是源标签需要复核？',
        '3. 已知PhrasIS要求脱离语境，close/end up和metal band/metal bar这类E采用的独立短语读法是什么？是否需要复核其与源iSTS标签的差别？',
        '',
        '这里只整理问题，没有向组织者发送消息。官方训练标签、训练流程和模型权重均未更改。',
        '',
        '## 逐组复核',
        '',
    ]
    for row in review.itertuples(index=False):
        lines += [f'### {row.review_id}：{row.text1} / {row.text2}', '',
                  f'- 官方：{row.gold}；分桶：{row.bucket}（{row.bucket_name}）；短语候选读法：{row.phrase_candidates}。',
                  f'- 读法条件：{row.assumptions}', f'- 判断：{row.assessment}',
                  f'- 后续：{row.action}', f'- 精确源命中：{row.exact_source_hits}；角色：{row.review_role}。']
        hits = sources[sources.review_id.eq(row.review_id)]
        for hit in hits.itertuples(index=False):
            lines += [f'- 来源：[{hit.source_file}:{hit.source_line}]({hit.source_url})；源方向关系{hit.oriented_relation}。',
                      f'  - Text1部分语境：{hit.context1_partial}', f'  - Text2部分语境：{hit.context2_partial}']
        lines.append('')
    lines += ['## 文件与复现', '',
              '- train_review_assessed.csv：逐组读法、解释、后续动作；官方gold不变。',
              '- train_review_sources_verified.csv：逐行核对原七列后的精确来源。',
              '- train_review_source_candidates.csv：2个近似候选，明确未证实身份。',
              '- train_review_models.csv / train_review_model_summary.csv：代表行已有模型读出。',
              '- train_review_existing_neighbours.csv：每组3个现有Train表面近邻；不把它们称作受控最小对照或硬冲突。',
              '- train_review_number_prefix.csv / train_review_attribute_prefix.csv：12组数量与45组属性限定筛选、官方标签和已有Train读出。',
              '- train_review_number_sources.csv：12组数量逐一核对的公开源记录。',
              '- train_review_prefix_models.csv：上述筛选的旧OOF/全Train模型读出汇总；不作控制训练实验比较。',
              '- train_review_exact_input_conflicts.csv：官方3042行和当前3290行中同输入不同标签的严格归一化检查。',
              '- train_review_lexical_isolation_cost.csv：现有零Dev词/词干规则对Train及新增48行表达形式的限制。',
              '- train_review_facts.json：数量、范围、限制和输入哈希。',
              '- 复现脚本：experiments/deberta_manual_aug/review_train_cases.py；不加载神经模型，不更新参数。',
              '',
              '任务标签映射参考：[DiCo-NLI官方说明](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/)。源数据：[公开PhrasIS仓库](https://github.com/Jonapa/assessing_logical_coherence/tree/main/data)。',
              '官方将源SIMI/REL/OPPO等映为N；这说明任务N不只是不相关，但不意味着所有源标签都已经通过本次独立语义仲裁。', '']
    (OUT / 'TRAIN_REVIEW_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    original_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
    key = pd.read_csv(OUT / 'train_review_key.csv', keep_default_na=False)
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    assert len(key) == 32 and key.pair_id.is_unique and set(key.review_id) == set(NOTES)
    source = source_evidence(key)
    records = []
    for row in key.to_dict('records'):
        bucket, candidates, assumptions, assessment, family, action = NOTES[row['review_id']]
        hits = source[source.review_id.eq(row['review_id'])]
        records.append({**row, 'bucket': bucket, 'bucket_name': BUCKETS[bucket],
                        'phrase_candidates': candidates, 'assumptions': assumptions,
                        'assessment': assessment, 'family': family, 'action': action,
                        'exact_source_hits': len(hits), 'source_status': 'exact' if len(hits) else 'not_exactly_located',
                        'diagnostic_priority': 'high' if bucket == 'A' and row['review_role'] == 'OOF_wrong' else 'review_or_control',
                        'new_training_label': '', 'annotation_status': STATUS})
    review = pd.DataFrame(records)
    save(review, 'train_review_assessed.csv')
    models, frames = model_comparison(key, review)
    neighbours = existing_neighbours(train, review, frames)
    prefix_audit = screen_prefixes(train, frames)
    conflicts = exact_input_conflicts(train)
    isolation = lexical_isolation_cost(train)
    assert hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest() == original_hash
    facts = {
        'review_groups': len(review), 'bucket_counts': dict(Counter(review.bucket)),
        'selection': '24 old Train-only OOF errors and 8 easy OOF-correct controls; not random',
        'OOF_role_counts': dict(Counter(review.review_role)),
        'exact_source_groups': source.review_id.nunique(), 'exact_source_hits': len(source),
        'missing_exact_source_ids': review.loc[review.exact_source_hits.eq(0), 'review_id'].tolist(),
        'near_source_candidates': 2, 'existing_neighbour_rows': len(neighbours),
        'prefix_audit': prefix_audit,
        'exact_input_conflicts': conflicts,
        'lexical_isolation': isolation,
        'clear_OOF_error_ids': review.loc[review.diagnostic_priority.eq('high'), 'review_id'].tolist(),
        'official_train_sha256_before_and_after': original_hash,
        'annotation_status': STATUS, 'new_training_rows': 0, 'new_diagnostic_rows': 0,
        'training_performed': False, 'official_labels_changed': False,
        'model_comparison_scope': 'already-trained Train representatives; old OOF has different architecture/recipe',
        'source_context_scope': 'partial reconstructions from public PhrasIS, not full verified original sentences',
        'source_annotation_protocol': 'PhrasIS paper p2 says phrases were relabeled independently of their context; reconstructed context explanations are post-hoc hypotheses only',
        'source_annotation_protocol_url': PHRASIS_PAPER,
        'source_file_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in SOURCE_DIR.glob('PhrasIS.*.*.all.txt')},
    }
    (OUT / 'train_review_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(review, source, models, facts)
    print(json.dumps(facts, ensure_ascii=False, indent=2))
    print('报告：', OUT / 'TRAIN_REVIEW_REPORT.md')


if __name__ == '__main__':
    main()
