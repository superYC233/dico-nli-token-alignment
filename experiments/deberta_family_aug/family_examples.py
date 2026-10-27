"""人工选择词项与关系，不自动把任意换词判为N。

每类10个家族：前8个训练、后2个保留诊断。每家族10个源对照，
展开后F/B/E/N各4行，共16行。不是新增官方标注。
"""
import pandas as pd


# 属性1、独立属性2、对象、无上下位关系的替换对象、地点。
ATTRIBUTE_FAMILIES = [
    ('beige', 'metal', 'cabinets', 'stools', 'depots'),
    ('pink', 'woolly', 'blankets', 'cushions', 'cabins'),
    ('blue', 'glass', 'bottles', 'jars', 'cupboards'),
    ('crimson', 'silk', 'scarves', 'mittens', 'lockers'),
    ('grey', 'rubber', 'mats', 'trays', 'garages'),
    ('golden', 'ceramic', 'mugs', 'kettles', 'shelves'),
    ('plaid', 'cotton', 'shirts', 'skirts', 'wardrobes'),
    ('tiny', 'metal', 'padlocks', 'dolls', 'bedrooms'),
    ('beige', 'knitted', 'duvets', 'pillowcases', 'lofts'),
    ('pink', 'ceramic', 'saucers', 'pitchers', 'pantries'),
]

# 对象、属性、参照物1、参照物2。under/underneath仅按静态空间关系读。
SPATIAL_FAMILIES = [
    ('canoes', 'blue', 'bridges', 'piers'),
    ('mopeds', 'grey', 'shelters', 'garages'),
    ('laptops', 'beige', 'desks', 'cabinets'),
    ('ladders', 'metal', 'balconies', 'scaffolds'),
    ('crates', 'heavy', 'shelves', 'platforms'),
    ('rucksacks', 'pink', 'stools', 'armchairs'),
    ('lanterns', 'tiny', 'awnings', 'verandas'),
    ('tractors', 'crimson', 'hangars', 'depots'),
    ('wheelbarrows', 'beige', 'gazebos', 'trellises'),
    ('hammocks', 'blue', 'pergolas', 'arbours'),
]

# 人员、人员属性、动作、物品单数/复数、物品属性、独立动作、地点。
PARTICIPANT_FAMILIES = [
    ('drivers', 'gloved', 'carrying', 'parcel', 'parcels', 'sealed', 'inspecting', 'depots'),
    ('tourists', 'elderly', 'carrying', 'backpack', 'backpacks', 'heavy', 'unpacking', 'cabins'),
    ('workers', 'helmeted', 'lifting', 'crate', 'crates', 'metal', 'inspecting', 'warehouses'),
    ('sailors', 'gloved', 'painting', 'canoe', 'canoes', 'blue', 'towing', 'harbours'),
    ('firefighters', 'helmeted', 'carrying', 'ladder', 'ladders', 'metal', 'inspecting', 'garages'),
    ('players', 'gloved', 'lifting', 'bat', 'bats', 'heavy', 'inspecting', 'stadiums'),
    ('riders', 'helmeted', 'carrying', 'helmet', 'helmets', 'blue', 'inspecting', 'depots'),
    ('mechanics', 'gloved', 'repairing', 'moped', 'mopeds', 'grey', 'inspecting', 'workshops'),
    ('weavers', 'elderly', 'carrying', 'basket', 'baskets', 'wicker', 'sorting', 'studios'),
    ('sculptors', 'gloved', 'polishing', 'figurine', 'figurines', 'bronze', 'inspecting', 'galleries'),
]

# 动作、一般对象、增加限制的对象、地点、月份、独立动作。
ACTIVITY_FAMILIES = [
    ('cooking', 'dinner', 'vegetarian dinner', 'cabins', 'November', 'serving'),
    ('drinking', 'beer', 'chilled beer', 'pubs', 'September', 'pouring'),
    ('skiing', 'slopes', 'icy slopes', 'lodges', 'February', 'hiking'),
    ('rowing', 'canoes', 'blue canoes', 'harbours', 'June', 'painting'),
    ('flying', 'aircraft', 'passenger aircraft', 'hangars', 'July', 'inspecting'),
    ('drawing', 'diagrams', 'detailed diagrams', 'studios', 'April', 'reading'),
    ('sewing', 'curtains', 'silk curtains', 'workshops', 'December', 'washing'),
    ('sorting', 'parcels', 'sealed parcels', 'depots', 'January', 'dispatching'),
    ('weighing', 'flour', 'rye flour', 'silos', 'August', 'pouring'),
    ('polishing', 'cutlery', 'metal cutlery', 'cafeterias', 'August', 'storing'),
]

# 两个不同活动，不以“常见先后流程”推导必然蕴含。
EVENT_FAMILIES = [
    ('inspecting', 'dispatching', 'parcels', 'sealed', 'depots', 'November'),
    ('approving', 'enforcing', 'regulations', 'municipal', 'annexes', 'September'),
    ('checking', 'renewing', 'permits', 'temporary', 'depots', 'February'),
    ('planning', 'completing', 'repairs', 'electrical', 'workshops', 'June'),
    ('measuring', 'coiling', 'cables', 'insulated', 'garages', 'July'),
    ('counting', 'charging', 'batteries', 'rechargeable', 'warehouses', 'April'),
    ('sealing', 'mailing', 'envelopes', 'padded', 'annexes', 'December'),
    ('drafting', 'renewing', 'contracts', 'rental', 'cabins', 'January'),
    ('weighing', 'selling', 'harvests', 'barley', 'silos', 'August'),
    ('sketching', 'unveiling', 'murals', 'outdoor', 'galleries', 'August'),
]

# 明确exactly读法：不把“17个”到“14个”错误标成单向蕴含。
# 数字、英文数字、另一基数、对象、属性1、独立属性2、替换对象、地点。
QUANTITY_FAMILIES = [
    ('14', 'fourteen', '15', 'cartons', 'sealed', 'heavy', 'sacks', 'depots'),
    ('15', 'fifteen', '16', 'bottles', 'blue', 'glass', 'jars', 'cupboards'),
    ('16', 'sixteen', '17', 'chairs', 'beige', 'metal', 'stools', 'cabins'),
    ('17', 'seventeen', '18', 'umbrellas', 'pink', 'folded', 'backpacks', 'shelters'),
    ('18', 'eighteen', '19', 'kites', 'crimson', 'tiny', 'dolls', 'warehouses'),
    ('19', 'nineteen', '20', 'mugs', 'ceramic', 'beige', 'kettles', 'shelves'),
    ('20', 'twenty', '40', 'helmets', 'blue', 'padded', 'gloves', 'garages'),
    ('40', 'forty', '14', 'radios', 'portable', 'grey', 'chargers', 'workshops'),
    ('40', 'forty', '50', 'thermoses', 'insulated', 'pink', 'goblets', 'pantries'),
    ('50', 'fifty', '40', 'prams', 'folded', 'beige', 'cradles', 'lofts'),
]


def build_examples():
    records = []

    def emit(family_type, index, contrast, text1, text2, label, reason, heads):
        family = f'{family_type}_{index + 1:02d}'
        records.append({
            'example_id': family + '_' + contrast, 'family_id': family,
            'split': 'train' if index < 8 else 'diagnostic',
            'family_type': family_type, 'contrast_type': contrast,
            'text1': text1, 'text2': text2, 'label': label,
            'label_reason': reason, 'heldout_heads': ' '.join(heads),
            'reading_assumption': 'Literal ordinary reading; no unstated story or causal prerequisite.',
        })

    for i, (a, b, head, foil, place) in enumerate(ATTRIBUTE_FAMILIES):
        add = lambda c, x, y, lab, why: emit('attribute_scope', i, c, x, y, lab, why, [head, foil])
        specific, other = f'{a} {head}', f'{a} {foil}'
        add('restriction1', specific, head, 'F', '颜色/大小等实际属性限制同一中心对象，反向不保证该属性。')
        add('restriction2', f'{a} {b} {head}', specific, 'F', '保留中心对象和属性1，增加独立的材质等属性2。')
        add('restriction3', f'{specific} outside {place}', specific, 'F', '同一对象增加静态地点条件。')
        add('restriction4', f'{specific} or {other}', f'{head} or {foil}', 'F', '两支分别增加属性限制；析取按集合并集读，不解释为时间顺序。')
        add('equivalence1', f'{specific} or {foil}', f'{foil} or {specific}', 'E', '静态对象备选项交换顺序，属性仍绑定同一对象。')
        add('equivalence2', f'{a} {b} {head} outside {place}', f'outside {place}, {a} {b} {head}', 'E', '同一个对象描述的地点状语前置；没有更换中心对象。')
        add('independent1', specific, f'{b} {head}', 'N', '属性1和属性2互不保证，不把共有中心词当作等价。')
        add('independent2', specific, other, 'N', '人工选定的不同对象没有上下位或同义包含关系。')
        add('independent3', f'{specific} or {foil}', f'{head} or {other}', 'N', '属性限制分别绑定不同析取支，两个方向各有反例。')
        add('independent4', f'{specific} outside {place}', f'{b} {head} outside {place}', 'N', '即使地点相同，两个独立对象属性仍不能互推。')

    for i, (head, attribute, ground, alternative) in enumerate(SPATIAL_FAMILIES):
        add = lambda c, x, y, lab, why: emit('spatial_reference', i, c, x, y, lab, why, [head])
        base = f'{head} under {ground}'
        add('restriction1', f'{attribute} {base}', base, 'F', '固定空间关系和参照物，只增加对象属性。')
        add('restriction2', base, head, 'F', '处于指定位置的对象仍是该类对象，反向没有位置保证。')
        add('restriction3', f'{attribute} {base}', f'{attribute} {head}', 'F', '保留属性，去掉位置限制。')
        add('restriction4', f'{base} or {head} under {alternative}', head, 'F', '无论哪个位置分支成立，都属于同一中心对象类别。')
        add('equivalence1', base, f'{head} underneath {ground}', 'E', 'under与underneath均固定为静态下方义，不用监督、控制等抽象义。')
        add('equivalence2', f'{attribute} {base}', f'under {ground}, {attribute} {head}', 'E', '同一位置关系的状语前置，参照物和对象未互换。')
        add('independent1', base, f'{head} above {ground}', 'N', '同一参照物的下方与上方不互推。')
        add('independent2', base, f'{head} under {alternative}', 'N', '参照物不同；不假设两个参照物同处或重合。')
        add('independent3', f'{attribute} {head}', base, 'N', '对象属性不保证位置，位置不保证对象属性。')
        add('independent4', f'{attribute} {base}', f'{head} above {alternative}', 'N', '不能从共有中心对象推得另一空间关系。')

    for i, (head, attribute, action, singular, obj, obj_attr, other_action, place) in enumerate(PARTICIPANT_FAMILIES):
        add = lambda c, x, y, lab, why: emit('participant_binding', i, c, x, y, lab, why, [head])
        base = f'{head} {action} {obj}'
        add('restriction1', f'{attribute} {base}', base, 'F', '动作和受事相同，只增加施事人员属性。')
        add('restriction2', f'{head} {action} {obj_attr} {obj}', base, 'F', '动作和施事相同，只增加受事物品属性。')
        add('restriction3', f'{base} outside {place}', base, 'F', '同一人员活动增加地点条件。')
        add('restriction4', f'{attribute} {head} {action} {obj_attr} {obj}', f'{head} {action} {obj_attr} {obj}', 'F', '保留受事属性，仅去掉施事属性限制。')
        add('equivalence1', base, f'{singular}-{action} {head}', 'E', '物品类别按泛指读，后置分词修饰改成同一动作的复合修饰，不断言精确基数。')
        add('equivalence2', f'{base} outside {place}', f'outside {place}, {base}', 'E', '地点前置，施事、动作和受事保持不变。')
        add('independent1', base, f'{head} {other_action} {obj}', 'N', '两种活动不能只因人员和受事相同就互推；不补流程前提。')
        add('independent2', f'{attribute} {base}', f'{head} {action} {obj_attr} {obj}', 'N', '施事属性和受事属性是独立条件，分别附加不能互推。')
        add('independent3', f'{attribute} {head}', base, 'N', '人员属性不保证正在做此动作；执行动作不保证该人员属性。')
        add('independent4', f'{base} outside {place}', f'{attribute} {head} {other_action} {obj}', 'N', '有共有人员和物品仍须分别验证动作、属性和地点。')

    for i, (action, obj, restricted, place, month, other_action) in enumerate(ACTIVITY_FAMILIES):
        add = lambda c, x, y, lab, why: emit('activity_restriction', i, c, x, y, lab, why, [obj.split()[-1]])
        base = f'{action} {obj}'
        specific = f'{action} {restricted}'
        add('restriction1', specific, base, 'F', '固定动作，缩小直接对象的类别或属性范围。')
        add('restriction2', f'{base} during {month}', base, 'F', '活动增加月份条件，反向没有月份保证。')
        add('restriction3', f'outside {place}, {base}', base, 'F', '前置地点仍是对同一活动增加限制，不把逗号仅关联E。')
        add('restriction4', f'{specific} during {month}', f'{base} during {month}', 'F', '时间相同，只增加受事对象限制。')
        add('equivalence1', f'{base} during {month}', f'during {month}, {base}', 'E', '月份状语前后移动，不改变事件信息。')
        add('equivalence2', f'{specific} outside {place}', f'outside {place}, {specific}', 'E', '地点状语前后移动，修饰范围和受事属性保持一致。')
        add('independent1', base, f'{other_action} {obj}', 'N', '相关动作也不互相保证，不根据典型活动流程补信息。')
        add('independent2', specific, f'{base} outside {place}', 'N', '受事属性和活动位置为独立条件。')
        add('independent3', f'{base} during {month}', f'{base} during midday', 'N', '月份不指定一天中的时段；中午也不指定月份。')
        add('independent4', f'outside {place}, {base}', f'{specific} during {month}', 'N', '地点与受事属性加月份不能互相保证。')

    for i, (action, other, obj, attribute, place, month) in enumerate(EVENT_FAMILIES):
        add = lambda c, x, y, lab, why: emit('event_distinction', i, c, x, y, lab, why, [obj])
        base, next_activity = f'{action} {obj}', f'{other} {obj}'
        add('restriction1', f'{action} {attribute} {obj}', base, 'F', '固定当前活动，只缩小其对象范围。')
        add('restriction2', f'{base} during {month}', base, 'F', '给当前活动增加月份条件。')
        add('restriction3', f'{next_activity} outside {place}', next_activity, 'F', '给另一活动增加地点，仍不依赖流程推理。')
        add('restriction4', f'{base} or {next_activity}', f'{next_activity} or {base} or sleeping', 'F', '原两个活动的析取可推出增加第三备选项后的析取，反向不保证原备选项。')
        add('equivalence1', f'{base} during {month}', f'during {month}, {base}', 'E', '同一活动的月份状语前置，活动状态未改变。')
        add('equivalence2', f'{next_activity} outside {place}', f'outside {place}, {next_activity}', 'E', '同一活动的地点状语前置，不把活动阶段换成另一阶段。')
        add('independent1', base, next_activity, 'N', '检查/运输、计划/完成等是不同当前活动；不假设先后活动都发生过。')
        add('independent2', f'{base} during {month}', f'{next_activity} during {month}', 'N', '给定相同月份，活动区别仍存在。')
        add('independent3', f'{action} {attribute} {obj}', f'{next_activity} {attribute} {obj}', 'N', '给定相同对象属性，活动区别仍存在。')
        add('independent4', f'outside {place}, {base}', f'{next_activity} during {month}', 'N', '不以地点和月份重合的可能性替代事件蕴含。')

    for i, (number, number_word, alternative, head, a, b, foil, place) in enumerate(QUANTITY_FAMILIES):
        add = lambda c, x, y, lab, why: emit('quantity_description', i, c, x, y, lab, why, [head, foil])
        base, specific = f'exactly {number} {head}', f'exactly {number} {a} {head}'
        add('restriction1', base, head, 'F', '精确正数量的对象组属于该对象类别；类别本身不保证数量。')
        add('restriction2', specific, f'{a} {head}', 'F', '固定属性，精确正数量的对象组可推出这种对象存在。')
        add('restriction3', specific, head, 'F', '精确正数量的带属性对象也属于一般对象类别，不要求一般类别的总数恰好相同。')
        add('restriction4', f'{a} {head} under {place}', f'{a} {head}', 'F', '数量家族中也保留普通空间限制对照；不把exactly作用域内增加修饰语错误当作单调限制。')
        add('equivalence1', specific, f'exactly {number_word} {a} {head}', 'E', '阿拉伯数字与英文数字表示同一个精确正基数。')
        add('equivalence2', f'{specific} outside {place}', f'outside {place}, {specific}', 'E', '同一数量对象组的地点状语前置。')
        add('independent1', base, f'exactly {alternative} {head}', 'N', '同一所述对象组的两个不同exactly基数不能互推；禁止按至少数量读。')
        add('independent2', f'{a} {head}', base, 'N', '属性条件与精确数量条件独立。')
        add('independent3', specific, f'exactly {number} {a} {foil}', 'N', '数量和属性相同，不同且无上下位关系的对象仍不互推。')
        add('independent4', specific, f'exactly {number} {b} {head}', 'N', '数量相同不代表两个独立对象属性相互保证。')

    return pd.DataFrame(records)
