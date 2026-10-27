"""从 Train 的关系图做传递闭包，生成标签由构造保证的新样本。

原理
 等价：A ≡ B 且 B ≡ C  =>  A ≡ C
 蕴含：A ⊨ B 且 B ⊨ C  =>  A ⊨ C
只做两跳；中间节点必须也在 Train 里出现过。

输出
 closure_pairs.csv  新对清单（含中间节点，便于人工复核）
 closure_rows.csv   可直接拼进训练集的 两行组
"""
import pandas as pd
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs/closure_generated'
OUT_PAIRS = OUTPUT_DIR / 'closure_pairs.csv'
OUT_ROWS = OUTPUT_DIR / 'closure_rows.csv'

# 蕴含闭包默认关闭。
# 原因：等价是双向的，两条边都成立才推得出来；蕴含是单向的，
# 中间任何一条边标错，两跳结果就是错的。实测里
#   'karzai' -> 'afghan legislator'  (总统不是立法者)
#   'in obama' -> 'us government'
#   'through water' -> 'the water'
# 都是这么来的。Train 的 F/B 边噪声明显大于 E 边。
USE_ENTAILMENT_CLOSURE = False

L2F = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
       'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}
F2L = {v: k for k, v in L2F.items()}


def norm(s):
    return ' '.join(str(s).casefold().split())


train = pd.read_csv(TRAIN, keep_default_na=False)
dev = pd.read_csv(DEV, keep_default_na=False)

# 已存在的关系（双向都记），新对不能落在这些上面
known = set()
for df in (train, dev):
    for a, b in zip(df.text1.map(norm), df.text2.map(norm)):
        known.add((a, b))
        known.add((b, a))

# 等价图（无向）、蕴含图（有向 a -> b 表示 a entails b）
E = defaultdict(set)
ENT = defaultdict(set)

for r in train.itertuples():
    a, b, lab = norm(r.text1), norm(r.text2), L2F[r.label]
    if lab == 'E':
        E[a].add(b)
        E[b].add(a)
    elif lab == 'F':
        ENT[a].add(b)
    elif lab == 'B':
        ENT[b].add(a)          # 标 B 表示 text2 entails text1


def two_hop(graph):
    """x -> y -> z，返回 {(x,z): {中间节点}}"""
    out = defaultdict(set)
    for x, mids in graph.items():
        for y in mids:
            for z in graph.get(y, ()):
                if z == x:
                    continue
                if (x, z) in known:
                    continue
                out[(x, z)].add(y)
    return out


e2 = two_hop(E)
f2 = two_hop(ENT)

# 等价是对称的，two_hop 会把 (a,b) 和 (b,a) 各算一次；
# 去重后每对只保留一条，否则展开成两行组会完全重复。
_uniq = {}
for (a, b), mids in e2.items():
    key = tuple(sorted((a, b)))
    if key not in _uniq:
        _uniq[key] = (a, b, mids)
e2_raw_n = len(e2)
e2 = {(a, b): m for a, b, m in _uniq.values()}

# 人工复核后确认有问题的（种子边本身标错，两跳结果必然错）。可自行增删。
EXCLUDE = {
    ("'s killing", 'among dead'),               # 种子边错误
    ('has seized', 'has been captured'),        # 主动/被动，施受角色反了
    ('the male child', 'the young male child'), # young 不被蕴含
    ('in the white snow', 'amid snow'),         # white 不被蕴含
    ('in the white snow', 'within the snow'),
    ('amid snow', 'in the white snow'),
    ('within the snow', 'in the white snow'),
}
def _excluded(a, b):
    return tuple(sorted((a, b))) in {tuple(sorted(x)) for x in EXCLUDE}
e2 = {k: v for k, v in e2.items() if not _excluded(*k)}

# 同一对可能同时被等价和蕴含推出；等价优先，蕴含那边去掉
both = set(e2) & set(f2)
for k in both:
    f2.pop(k, None)

print('等价闭包：', e2_raw_n, '→ 去重后', len(e2), '→ 排除后', len(e2), '对')
print('蕴含闭包：', len(f2), '对')
print('两边都推出（按等价处理）：', len(both))
print('合计启用：', len(e2) + (len(f2) if USE_ENTAILMENT_CLOSURE else 0), '对')

# ---------- 输出新对清单 ----------
records = []
for (a, b), mids in sorted(e2.items()):
    records.append({'text1': a, 'text2': b, 'label': 'E',
                    'kind': 'equivalence_closure',
                    'middle': ' | '.join(sorted(mids)), 'support': len(mids)})
if USE_ENTAILMENT_CLOSURE:
    for (a, b), mids in sorted(f2.items()):
        records.append({'text1': a, 'text2': b, 'label': 'F',
                        'kind': 'entailment_closure',
                        'middle': ' | '.join(sorted(mids)), 'support': len(mids)})

pairs = pd.DataFrame(records)
pairs['len1'] = pairs.text1.str.split().str.len()
pairs['len2'] = pairs.text2.str.split().str.len()
pairs = pairs.sort_values(['kind', 'len1', 'len2']).reset_index(drop=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pairs.to_csv(OUT_PAIRS, index=False, encoding='utf-8-sig')
print('已写出：', OUT_PAIRS)

# ---------- 展开成两行组 ----------
INV = {'F': 'B', 'B': 'F', 'E': 'E'}
rows = []
for i, r in pairs.iterrows():
    pid = f'aug_closure_{i:03d}_{r.label}'
    views = [(r.text1, r.text2, r.label)]
    if r.label != 'N':
        views.append((r.text2, r.text1, INV[r.label]))
    for d, (a, b, lab) in enumerate(views):
        rows.append({'instance_id': f'{pid}__{d}', 'pair_id': pid,
                     'text1_lang': 'en', 'text2_lang': 'en',
                     'text1': a, 'text2': b, 'label': F2L[lab]})

out = pd.DataFrame(rows)
out.to_csv(OUT_ROWS, index=False, encoding='utf-8-sig')
print('训练行：', len(out), '  新对：', len(pairs))
print(out.label.value_counts().to_string())
print()
print('--- 最短的 25 对，先人工过这些 ---')
for _, r in pairs.head(25).iterrows():
    print(f"  {r.label}  {r.text1!r:<38} -> {r.text2!r:<38}  via {r.middle[:40]}")
