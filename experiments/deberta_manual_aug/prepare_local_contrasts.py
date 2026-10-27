"""四类局部对照：新增Train 128行，另留64行诊断；不训练、不修改旧数据。"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from nltk.stem import PorterStemmer

from experiments.deberta_manual_aug.prepare_manual_data import (
    FULL, INVERSE, normalize, words, overlap_report, pair_key,
)
from experiments.deberta_lexical.data import validate_groups


HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / 'outputs/local_contrasts_prepared'
BASELINE_DIR = HERE / 'outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
STEMMER = PorterStemmer()
COLUMNS = ['instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label']

# 每类4个训练词族、2个诊断词族。品牌名为人工虚构，没有真实品牌知识前提。
# 字段：编号、分区、品牌1、品牌2、属性、物品。
NAME_FAMILIES = [
    ('name01', 'train', 'Kelvori', 'Kelvorin', 'portable', 'radios'),
    ('name02', 'train', 'Meruva', 'Meruvan', 'insulated', 'kettles'),
    ('name03', 'train', 'Solvani', 'Solvanir', 'ceramic', 'ovens'),
    ('name04', 'train', 'Tiravon', 'Tiravona', 'digital', 'thermometers'),
    ('name05', 'diagnostic', 'Delvori', 'Delvorin', 'cordless', 'scanners'),
    ('name06', 'diagnostic', 'Neruvia', 'Neruvian', 'bronze', 'compasses'),
]
# 字段：编号、分区、动作、用途限定、物品、位置介词、地点、月份。
PURPOSE_FAMILIES = [
    ('purpose01', 'train', 'assembling', 'rehearsal', 'equipment', 'beneath', 'awnings', 'April'),
    ('purpose02', 'train', 'storing', 'medical', 'supplies', 'outside', 'sheds', 'November'),
    ('purpose03', 'train', 'repairing', 'farming', 'tools', 'among', 'tents', 'January'),
    ('purpose04', 'train', 'cleaning', 'brewing', 'vats', 'beyond', 'huts', 'February'),
    ('purpose05', 'diagnostic', 'arranging', 'festival', 'lanterns', 'beneath', 'canopies', 'June'),
    ('purpose06', 'diagnostic', 'packing', 'audition', 'props', 'outside', 'pavilions', 'July'),
]
# 字段：编号、分区、主体动作、位置介词、环境中的动作、环境对象、月份。
PATH_FAMILIES = [
    ('path01', 'train', 'working', 'among', 'barking', 'hounds', 'April'),
    ('path02', 'train', 'crawling', 'beneath', 'dripping', 'pipes', 'November'),
    ('path03', 'train', 'strolling', 'among', 'chirping', 'finches', 'January'),
    ('path04', 'train', 'walking', 'beyond', 'clattering', 'shutters', 'February'),
    ('path05', 'diagnostic', 'waiting', 'beneath', 'chiming', 'bells', 'June'),
    ('path06', 'diagnostic', 'wandering', 'among', 'rustling', 'reeds', 'July'),
]
# 字段：编号、分区、职级、职业、动作1、宾语复数、宾语单数、动作2。
ROLE_FAMILIES = [
    ('role01', 'train', 'senior', 'archivists', 'cataloguing', 'manuscripts', 'manuscript', 'restoring'),
    ('role02', 'train', 'junior', 'treasurers', 'examining', 'ledgers', 'ledger', 'shredding'),
    ('role03', 'train', 'senior', 'botanists', 'pressing', 'orchids', 'orchid', 'pruning'),
    ('role04', 'train', 'principal', 'conservators', 'polishing', 'mosaics', 'mosaic', 'repairing'),
    ('role05', 'diagnostic', 'senior', 'librarians', 'indexing', 'volumes', 'volume', 'binding'),
    ('role06', 'diagnostic', 'junior', 'curators', 'sorting', 'relics', 'relic', 'displaying'),
]


def build_examples():
    records = []

    def add(family, split, kind, contrast, first, second, label, reason, content):
        records.append({'example_id': family + '_' + contrast, 'family_id': family,
                        'split': split, 'family_type': kind, 'contrast_type': contrast,
                        'text1': first, 'text2': second, 'label': label, 'label_reason': reason,
                        'content_words': ' '.join(content)})

    for family, split, brand1, brand2, attribute, noun in NAME_FAMILIES:
        base = f'{attribute} {noun}'
        branded1, branded2 = f'{brand1}-branded {base}', f'{brand2}-branded {base}'
        content = [brand1, brand2, attribute, noun]
        emit = lambda c, a, b, y, r: add(family, split, 'name_boundary', c, a, b, y, r, content)
        emit('restriction1', branded1, base, 'F', '带指定品牌的物品属于该类物品；一般物品不保证品牌。')
        emit('restriction2', branded2, noun, 'F', '属性和品牌是正向限制，不能从一般物品推出。')
        emit('rephrase', branded1, f'{base} branded {brand1}', 'E', '两侧均为带同一品牌的同类物品，前后置改写不增加条件。')
        emit('independent1', f'{brand1}-branded {noun}', f'{brand2}-branded {noun}', 'N',
             '两个不同拼写的虚构品牌没有已知同一或包含关系；共享名称子串不建立蕴含。')
        emit('independent2', branded1, f'{brand2}-branded {noun}', 'N',
             '增加属性仍不能推出另一品牌；另一品牌也不保证原品牌和属性。')

    for family, split, verb, purpose, noun, prep, place, month in PURPOSE_FAMILIES:
        base, intended = f'{verb} {noun}', f'{verb} {purpose} {noun}'
        located = f'{base} {prep} {place}'
        full = f'{intended} {prep} {place}'
        content = [verb, purpose, noun, place, month]
        emit = lambda c, a, b, y, r: add(family, split, 'purpose_place', c, a, b, y, r, content)
        emit('restriction1', intended, base, 'F', '用途限定的物品属于一般物品；一般动作不保证该用途。')
        emit('restriction2', located, base, 'F', '增加地点条件可推出同一动作，但动作不保证该地点。')
        emit('rephrase', full, f'{prep} {place}, {intended}', 'E', '同一动作的位置状语前置，施事和作用范围保持相同。')
        emit('independent1', intended, located, 'N', '用途限定和实际地点是独立条件，两侧互不保证。')
        emit('independent2', full, f'{base} during {month}', 'N',
             '用途和地点不保证具体月份；具体月份也不保证用途和地点。两侧没有蕴含保证。')

    for family, split, action, prep, environmental, noun, month in PATH_FAMILIES:
        base = f'{action} {prep} {noun}'
        full = f'{action} {prep} {environmental} {noun}'
        changed_actor = f'{environmental} {prep} {noun}'
        content = [action, environmental, noun, month]
        emit = lambda c, a, b, y, r: add(family, split, 'path_actor', c, a, b, y, r, content)
        emit('restriction1', full, base, 'F', '描述环境对象正在做某事，是对环境的限定，不改变主体动作。')
        emit('restriction2', f'{full} during {month}', full, 'F', '增加活动时间，只能推出一般活动，反向不保证月份。')
        emit('rephrase', full, f'{prep} {environmental} {noun}, {action}', 'E', '环境状语前置，环境中的动作仍修饰原名词，不改成主体动作。')
        emit('independent1', full, changed_actor, 'N', '身处某种活动环境不代表主体执行该活动；执行该活动也不保证原主体动作。')
        emit('independent2', f'{full} during {month}', f'{changed_actor} during {month}', 'N',
             '固定同一月份后，环境动作与主体动作的区别仍然存在。')

    for family, split, rank, job, action, objects, singular, other_action in ROLE_FAMILIES:
        ranked, working = f'{rank} {job}', f'{job} {action} {objects}'
        content = [job, action, objects, singular, other_action]
        emit = lambda c, a, b, y, r: add(family, split, 'role_restriction', c, a, b, y, r, content)
        emit('restriction1', ranked, job, 'F', '在给定职业义下，带职级的职业成员属于该职业，反向不保证职级。')
        emit('restriction2', f'{ranked} {action} {objects}', working, 'F', '保持动作和对象，仅增加职业职级限制。')
        emit('rephrase', working, f'{singular}-{action} {job}', 'E', '动作后置修饰改为复合修饰语，仍是执行同一动作的同类人员。')
        emit('independent1', working, f'{job} {other_action} {objects}', 'N', '两种具体职业活动不能互推，不能只按职业相同判E。')
        emit('independent2', ranked, working, 'N', '职级不保证正在执行某项具体活动，执行活动也不保证职级。')
    return pd.DataFrame(records)


def expand_rows(examples):
    records = []
    for row in examples.itertuples(index=False):
        group = 'local_' + row.example_id
        pairs = [(row.text1, row.text2, row.label)]
        if row.label != 'N':
            pairs.append((row.text2, row.text1, INVERSE[row.label]))
        if row.contrast_type == 'restriction2':
            pairs.reverse()  # FB和BF各占一半，不让新数据只增加一种组内方向。
        for index, (first, second, label) in enumerate(pairs):
            records.append({'instance_id': group + f'__{index}', 'pair_id': group,
                            'text1_lang': 'en', 'text2_lang': 'en', 'text1': first, 'text2': second,
                            'label': FULL[label], 'family_id': row.family_id, 'split': row.split,
                            'family_type': row.family_type, 'contrast_type': row.contrast_type,
                            'label_reason': row.label_reason})
    return pd.DataFrame(records)


def make_copy_control(baseline):
    # 数量、类别和可逆组构成都匹配新增128行；复制整组，N仍是单例。
    rng = random.Random(42)
    groups = {'FB': [], 'BF': [], 'EE': [], 'N': []}
    for _, group in baseline.groupby('pair_id', sort=False):
        labels = set(group.label)
        if labels == {FULL['F'], FULL['B']}:
            kind = 'FB' if group.label.iloc[0] == FULL['F'] else 'BF'
            groups[kind].append(group)
        elif labels == {FULL['E']}:
            groups['EE'].append(group)
        elif labels == {FULL['N']}:
            groups['N'].append(group)
    copies, sources = [], []
    for kind, count in [('FB', 16), ('BF', 16), ('EE', 16), ('N', 32)]:
        for index, group in enumerate(rng.sample(groups[kind], count)):
            copied = group.copy()
            new_pair = f'local_copy_{kind}_{index:03d}'
            copied['pair_id'] = new_pair
            copied['instance_id'] = [new_pair + f'__{i}' for i in range(len(group))]
            copies.append(copied[COLUMNS])
            for old_id, new_id in zip(group.instance_id, copied.instance_id):
                sources.append({'original_instance_id': old_id, 'new_instance_id': new_id,
                                'original_pair_id': group.pair_id.iloc[0], 'new_pair_id': new_pair})
    return pd.concat(copies, ignore_index=True), pd.DataFrame(sources)


def prepare():
    baseline = pd.read_csv(BASELINE_DIR / 'training_data.csv', keep_default_na=False)
    dev_text = pd.read_csv(DEV_PATH, usecols=['text1', 'text2'], keep_default_na=False)
    examples = build_examples()
    expanded = expand_rows(examples)
    extra = expanded[expanded.split.eq('train')].reset_index(drop=True)
    diagnostic = expanded[expanded.split.eq('diagnostic')].reset_index(drop=True)
    assert len(baseline) == 3290 and len(extra) == 128 and len(diagnostic) == 64
    assert set(extra.family_id).isdisjoint(diagnostic.family_id)
    for frame in [extra, diagnostic]:
        validate_groups(frame)
        assert frame.label.value_counts().to_dict() == {label: len(frame) // 4 for label in FULL.values()}

    # 新增Train与诊断均不碰Dev词形、词干、整短语或无序文本对。
    audits = {}
    for name, frame in [('extra_train', extra), ('diagnostic', diagnostic)]:
        audit = overlap_report(frame, dev_text)
        assert all(audit[field] == 0 for field in [
            'word_intersection_count', 'porter_stem_intersection_count',
            'whole_phrase_intersection_count', 'unordered_pair_intersection_count']) , (name, audit)
        audits[name + '_vs_dev'] = audit

    # 词族隔离只比较内容词；构造算子/语法可以共用。旧3290行的既有词汇另行披露。
    content = {}
    for split in ['train', 'diagnostic']:
        content[split] = {STEMMER.stem(word) for text in examples.loc[examples.split.eq(split), 'content_words']
                          for word in words(text)}
    assert not content['train'] & content['diagnostic'], content['train'] & content['diagnostic']
    old_pairs = {pair_key(a, b) for a, b in zip(baseline.text1, baseline.text2)}
    train_pairs = {pair_key(a, b) for a, b in zip(extra.text1, extra.text2)}
    diagnostic_pairs = {pair_key(a, b) for a, b in zip(diagnostic.text1, diagnostic.text2)}
    assert not train_pairs & diagnostic_pairs and not (train_pairs | diagnostic_pairs) & old_pairs
    old_phrases = {normalize(text) for column in ['text1', 'text2'] for text in baseline[column]}
    new_phrases = {normalize(text) for column in ['text1', 'text2'] for text in expanded[column]}
    assert not old_phrases & new_phrases, old_phrases & new_phrases
    keys = pd.DataFrame({'first': expanded.text1.map(normalize), 'second': expanded.text2.map(normalize),
                         'label': expanded.label})
    assert keys.groupby(['first', 'second']).label.nunique().max() == 1

    copy, copy_sources = make_copy_control(baseline)
    assert copy.label.value_counts().to_dict() == extra.label.value_counts().to_dict()
    validate_groups(copy)
    manual_groups = extra.groupby('pair_id', sort=False).label.agg(tuple).value_counts().to_dict()
    copy_groups = copy.groupby('pair_id', sort=False).label.agg(tuple).value_counts().to_dict()
    assert manual_groups == copy_groups, (manual_groups, copy_groups)
    combined = pd.concat([baseline[COLUMNS], extra[COLUMNS]], ignore_index=True)
    copied = pd.concat([baseline[COLUMNS], copy], ignore_index=True)
    validate_groups(combined)
    validate_groups(copied)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [('examples_source.csv', examples), ('extra_train_rows.csv', extra),
                        ('diagnostic_rows.csv', diagnostic), ('training_data.csv', combined),
                        ('training_data_copy.csv', copied), ('copy_control_rows.csv', copy),
                        ('copy_control_sources.csv', copy_sources)]:
        frame.to_csv(PREPARED_DIR / name, index=False, encoding='utf-8-sig')
    report = {'baseline_rows': len(baseline), 'extra_train_rows': len(extra),
              'train_rows': len(combined), 'diagnostic_rows': len(diagnostic),
              'train_families': extra.family_id.nunique(), 'diagnostic_families': diagnostic.family_id.nunique(),
              'class_counts_new_train': extra.label.value_counts().to_dict(),
              'class_counts_diagnostic': diagnostic.label.value_counts().to_dict(),
              'new_train_group_counts': {'FB': 16, 'BF': 16, 'EE': 16, 'N_singleton': 32},
              **audits, 'train_diagnostic_content_stem_overlap': [],
              'baseline_new_whole_phrase_overlap': [], 'baseline_new_unordered_pair_overlap': [],
              'annotation': 'Assistant-designed examples with explicit readings; not independently human-adjudicated.',
              'provenance': 'Four patterns selected from Train-only OOF review; templates and labels are newly constructed. No source context copied.',
              'dev_use': 'Dev text only as a rejection blacklist. No Dev label loaded for construction.',
              'diagnostic_scope': 'Held-out content families within the new batch; shared templates/operators are allowed. Not a new official test set.',
              'baseline_scope': 'Old3290 retains existing vocabulary, one known label-conflicting input and earlier Dev-guided development history.',
              'diagnostic_in_training': False, 'diagnostic_used_for_checkpoint_selection': False,
              'training_performed_by_prepare': False,
              'baseline_csv_sha256': hashlib.sha256((BASELINE_DIR / 'training_data.csv').read_bytes()).hexdigest(),
              'prepare_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (PREPARED_DIR / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('训练：旧3290 + 新128 =', len(combined))
    print('独立诊断：64行，8个留出词族；不参加训练、不参加选模。')
    print('新增数据与Dev的词形、词干、整短语和文本对交集：均为0。')
    print('数据与标签理由：', PREPARED_DIR / 'examples_source.csv')
    return combined, diagnostic, report


if __name__ == '__main__':
    prepare()
