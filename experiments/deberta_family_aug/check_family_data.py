"""数据、语义结构、整家族筛选和实际分词检查；不加载模型或训练。"""
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_family_aug.family_examples import build_examples, QUANTITY_FAMILIES, ATTRIBUTE_FAMILIES
from experiments.deberta_family_aug.prepare_family_data import (
    prepare, PREPARED_DIR, BASELINE_DIR, screen_families, conflict_rows, FULL,
)
from experiments.deberta_joint_pair.data import GroupDataset
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset


def check_scope_logic():
    def label(first, second):
        forward, backward = first <= second, second <= first
        return 'E' if forward and backward else 'F' if forward else 'B' if backward else 'N'

    # 有两种无上下位关系的对象，分别可有/无两个独立属性。
    universe = set(itertools.product(['head', 'foil'], [False, True], [False, True]))
    head = {x for x in universe if x[0] == 'head'}
    foil = universe - head
    attr1 = {x for x in universe if x[1]}
    attr2 = {x for x in universe if x[2]}
    assert label(head & attr1, head) == 'F'
    assert label(head & attr1 & attr2, head & attr1) == 'F'
    assert label((head & attr1) | (foil & attr1), head | foil) == 'F'
    assert label((head & attr1) | foil, foil | (head & attr1)) == 'E'
    assert label(head & attr1, head & attr2) == 'N'
    assert label((head & attr1) | foil, head | (foil & attr1)) == 'N'
    # 对象包含关系不同，中心词不同不能机械当N：厨房是房间的一种。
    assert label({'kitchen'}, {'kitchen', 'lounge'}) == 'F'

    # 防止把明确的上下位/同义词塞进“改变对象=N”的模板。
    known_inclusions = [({'dolls', 'toys'}), ({'kitchens', 'rooms'}), ({'motorbikes', 'motorcycles'})]
    for _, _, head, foil, _ in ATTRIBUTE_FAMILIES:
        assert {head, foil} not in known_inclusions

    mapping = {'14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
               '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '40': 'forty', '50': 'fifty'}
    for number, word, other, *_ in QUANTITY_FAMILIES:
        assert mapping[number] == word and int(number) != int(other) and int(number) > 0
    source = build_examples()
    quantity_f = source[source.family_type.eq('quantity_description') & source.label.eq('F')]
    assert not (quantity_f.text1.str.startswith('exactly') & quantity_f.text2.str.startswith('exactly')).any()
    print('结构检查通过：限制、析取作用范围、数字英文对应；没有exactly修饰作用域的伪F。')


def check_atomic_screen():
    source = build_examples().iloc[:20].copy()  # 两个完整属性家族。
    empty = pd.DataFrame(columns=['text1', 'text2'])
    old = source.iloc[:1][['text1', 'text2']]
    accepted, rejected = screen_families(source, empty, old, empty)
    assert len(accepted) == 10 and len(rejected) == 1
    assert set(accepted.family_id) == {source.family_id.iloc[-1]}
    fake_dev = pd.DataFrame({'text1': ['beige'], 'text2': ['unrelated']})
    accepted, rejected = screen_families(source, fake_dev, empty, empty)
    assert len(accepted) == 10 and len(rejected) == 1
    print('整家族筛选通过：单个对照重复或命中Dev会排除整家族，不留下残缺配对。')


def main():
    check_scope_logic()
    check_atomic_screen()
    combined, extra, diagnostic, report = prepare()
    copied = pd.read_csv(PREPARED_DIR / 'training_data_copy.csv', keep_default_na=False)
    control = pd.read_csv(PREPARED_DIR / 'copy_control_rows.csv', keep_default_na=False)
    baseline = pd.read_csv(BASELINE_DIR / 'training_data.csv', keep_default_na=False)
    assert len(extra) == 768 and len(diagnostic) == 192 and len(combined) == len(copied) == 4058
    assert report['train_families'] == 48 and report['diagnostic_families'] == 12
    assert report['rejected_families'] == 0
    assert combined.iloc[:3290].reset_index(drop=True).equals(baseline)
    assert set(conflict_rows(combined).instance_id) == set(conflict_rows(baseline).instance_id)
    assert set(conflict_rows(copied).instance_id) == set(conflict_rows(baseline).instance_id)
    for frame in [extra, control]:
        assert frame.label.value_counts().to_dict() == {value: 192 for value in FULL.values()}
        order = Counter(tuple(g.label) for _, g in frame.groupby('pair_id', sort=False))
        assert order == {(FULL['F'], FULL['B']): 96, (FULL['B'], FULL['F']): 96,
                         (FULL['E'], FULL['E']): 96, (FULL['N'],): 192}
    reference = pd.read_csv(PREPARED_DIR / 'diagnostic_reference.csv', keep_default_na=False)
    assert len(reference) == 192 and reference.instance_id.is_unique
    assert reference.loc[reference.label.eq(FULL['N']), 'reverse_pair_id'].eq('').all()
    assert reference.loc[reference.label.ne(FULL['N']), 'reverse_pair_id'].isin(reference.instance_id).all()
    print('预算通过：训练768、诊断192；F/B/E/N均衡；复制对照标签及FB/BF/EE/N组数一致。')

    tokenizer = AutoTokenizer.from_pretrained(
        'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    for name, frame in [('manual', combined), ('copy', copied)]:
        groups = GroupDataset(frame, tokenizer, 128)
        pairs = RealPairDataset(groups.rows)
        assert len(pairs) == 1668 and len(groups) == 2390
        assert groups.rows.labels.bincount(minlength=4).tolist() == [1125, 1125, 1086, 722]
        # 检查新增和诊断的每条完整输入，不截掉文本；不能只检查平均长度。
        print(name, 'Row步数', math.ceil(len(frame) / 16), 'Pair步数', math.ceil(len(pairs) / 16))
    for frame in [extra, diagnostic]:
        lengths = tokenizer(frame.text1.tolist(), frame.text2.tolist(), truncation=False)['input_ids']
        assert max(map(len, lengths)) <= 128
    print('分词通过：新增和诊断均无需截断。全部检查完成，没有加载模型或训练。')


if __name__ == '__main__':
    main()
