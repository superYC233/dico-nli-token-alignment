"""构造范围增强和匹配复制数据；不依赖冻结配置，不加载模型。"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_lexical.lexical_features import tokenize_words
from experiments.deberta_minimal_contrasts.prepare_diagnostics import pair_key


DIAGNOSTIC_PATH = ROOT / 'experiments/deberta_minimal_contrasts/prepared/run02_binding/examples.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 新增训练对象组合，与保留诊断的中心词/替换对象分开。
# 这些词都在原 Train 中出现，标签按字面名词短语的逻辑关系构造。
# 这是人工模板监督，不是新增官方标注；尚无独立标注者复核。
FAMILIES = [
    ('red', 'truck', 'bridge'),
    ('brown', 'horse', 'cow'),
    ('black', 'duck', 'wheel'),
    ('gray', 'building', 'phone'),
    ('red', 'fence', 'book'),
    ('gray', 'road', 'river'),
    ('blue', 'shirt', 'hat'),
    ('red', 'coat', 'wheel'),
    ('green', 'dress', 'bottle'),
    ('white', 'book', 'window'),
    ('blue', 'bottle', 'desk'),
    ('brown', 'guitar', 'cow'),
    ('black', 'drum', 'bench'),
    ('black', 'phone', 'jacket'),
    ('gray', 'laptop', 'carpet'),
    ('green', 'tree', 'couch'),
    ('red', 'skateboard', 'sofa'),
    ('yellow', 'kite', 'bridge'),
    ('gray', 'shed', 'lake'),
    ('brown', 'door', 'horse'),
]


def make_scope_rows(train):
    word_sources = {}
    for row in train.itertuples():
        for word in tokenize_words(row.text1) + tokenize_words(row.text2):
            word_sources.setdefault(word, row.instance_id)

    rows = []
    audit = []
    inverse = {'F': 'B', 'B': 'F', 'E': 'E'}
    full = dict(zip(LABELS, FULL_LABELS))

    def add(family, kind, first, second, label, reason, sources):
        pair_id = f'aug_scope_{family}_{kind}'
        views = [(first, second, label)]
        if label != 'N':
            views.append((second, first, inverse[label]))
        for direction, (text1, text2, target) in enumerate(views):
            instance_id = pair_id + f'__{direction}'
            rows.append({'instance_id': instance_id, 'pair_id': pair_id,
                         'text1_lang': 'en', 'text2_lang': 'en',
                         'text1': text1, 'text2': text2, 'label': full[target]})
            audit.append({'instance_id': instance_id, 'pair_id': pair_id,
                          'family': family, 'kind': kind, 'target': target,
                          'label_basis': reason, 'word_source_ids': '|'.join(sources),
                          'annotation_status': 'synthetic_literal_semantics; no independent adjudication'})

    for index, (color, noun, foil) in enumerate(FAMILIES):
        words = [color, noun, foil]
        missing = [word for word in words if word not in word_sources]
        if missing:
            raise ValueError('增强词不在 Train 中：' + str(missing))
        sources = [word_sources[word] for word in words]
        family = f'{index:02d}_{noun}_{foil}'
        specific = f'a {color} {noun}'
        broad = f'a {noun}'
        other = f'a {foil}'
        other_specific = f'a {color} {foil}'
        expanded = f'a {noun} that is {color}'
        # 简单 F/B、E、N 保持平衡，不能把“出现 and”直接当作某个类别。
        add(family, 'simple_restriction', specific, broad, 'F',
            '颜色限定缩小对象范围，普通对象不保证具有该颜色。', sources)
        add(family, 'simple_equivalence', specific, expanded, 'E',
            '形容词与同一颜色属性的从句展开，按字面解释保义。', sources)
        add(family, 'object_change', specific, other_specific, 'N',
            '保留颜色而改变对象；此处所选对象不是一般上下位包含。', sources)
        add(family, 'modifier_scope', specific + ' and ' + other_specific,
            specific + ' and ' + other, 'F',
            '左侧还要求第二个对象具有颜色，右侧没有该要求。', sources)
        add(family, 'modifier_binding', specific + ' and ' + other,
            broad + ' and ' + other_specific, 'N',
            '颜色分别属于不同对象；存在两个方向各自成立而另一方向不成立的情形。', sources)
        add(family, 'conjunct_order', specific + ' and ' + other,
            other + ' and ' + specific, 'E',
            '静态名词并列只换顺序，颜色仍修饰同一个对象。', sources)

    extra = pd.DataFrame(rows)[train.columns]
    audit = pd.DataFrame(audit)
    if not extra.instance_id.is_unique or set(extra.pair_id) & set(train.pair_id):
        raise ValueError('增强 ID 与原数据重复。')
    return extra, audit


def validate_holdout(extra, dev):
    if not DIAGNOSTIC_PATH.exists():
        raise FileNotFoundError('先保留已有诊断文件：' + str(DIAGNOSTIC_PATH))
    diagnostic = pd.read_csv(DIAGNOSTIC_PATH, keep_default_na=False)
    forbidden = {pair_key(r.text1, r.text2) for r in dev.itertuples()}
    forbidden |= {pair_key(r.text1, r.text2) for r in diagnostic.itertuples()}
    collisions = extra[[pair_key(a, b) in forbidden for a, b in zip(extra.text1, extra.text2)]]
    if len(collisions):
        raise ValueError('增强题与 Dev/保留诊断题重合，需改训练模板后另开 RUN_NAME。')


def make_copy_rows(train, scope, seed):
    # 每个 scope 组对应一个相同标签向量的 Train 复制组。
    # 因而两臂的行数、类别数、组数，以及同 seed 下每步的标签/行数完全一致。
    pools = {('F', 'B'): [], ('E', 'E'): [], ('N',): []}
    short = dict(zip(FULL_LABELS, LABELS))
    for _, group in train.groupby('pair_id', sort=False):
        labels = tuple(group.label.map(short))
        if set(labels) == {'F', 'B'} and len(group) == 2:
            group = group.iloc[[labels.index('F'), labels.index('B')]]
            pools[('F', 'B')].append(group)
        elif labels in pools:
            pools[labels].append(group)
        else:
            raise ValueError('原 Train 存在不合法的分组。')

    rng = random.Random(seed)
    copied = []
    audit = []
    for scope_id, target in scope.groupby('pair_id', sort=False):
        labels = tuple(target.label.map(short))
        source = rng.choice(pools[labels])
        for position, (_, row) in enumerate(source.iterrows()):
            record = row.to_dict()
            record['pair_id'] = scope_id.replace('aug_scope_', 'aug_copy_', 1)
            record['instance_id'] = record['pair_id'] + f'__{position}'
            copied.append(record)
            audit.append({'instance_id': record['instance_id'], 'pair_id': record['pair_id'],
                          'control_source_instance_id': row.instance_id,
                          'control_source_pair_id': row.pair_id,
                          'matched_scope_instance_id': target.iloc[position].instance_id,
                          'target': short[row.label]})
    return pd.DataFrame(copied)[train.columns], pd.DataFrame(audit)


def build_train_variant(train, dev, mode, seed=42):
    if mode == 'none':
        return train.copy(), train.iloc[:0].copy(), pd.DataFrame()
    if mode not in ['scope', 'copy']:
        raise ValueError('AUGMENT_MODE 只能为 none、scope 或 copy。')
    scope, audit = make_scope_rows(train)
    validate_holdout(scope, dev)
    extra = scope
    if mode == 'copy':
        diagnostic = pd.read_csv(DIAGNOSTIC_PATH, keep_default_na=False)
        forbidden = {pair_key(r.text1, r.text2) for r in dev.itertuples()}
        forbidden |= {pair_key(r.text1, r.text2) for r in diagnostic.itertuples()}
        blocked_groups = {r.pair_id for r in train.itertuples()
                          if pair_key(r.text1, r.text2) in forbidden}
        pool = train[~train.pair_id.isin(blocked_groups)]
        extra, audit = make_copy_rows(pool, scope, seed)
        validate_holdout(extra, dev)
    return pd.concat([train, extra], ignore_index=True), extra, audit
