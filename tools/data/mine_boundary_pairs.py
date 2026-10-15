
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

train_file = (PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv')
output_file = (PROJECT_ROOT / 'results/boundary_mining/boundary_pair_candidates_unreviewed.csv')

data = pd.read_csv(train_file)
by_anchor = defaultdict(list)

for pair_id, group in data.groupby("pair_id"):
    if len(group) != 2:
        continue

    labels = set(group["label"])
    if labels == {"EQUIVALENCE"}:
        relation = "E"
    elif labels == {"FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"}:
        relation = "D"
    else:
        continue

    row = group.iloc[0]
    for anchor, other in [(row["text1"], row["text2"]),
                          (row["text2"], row["text1"])]:
        anchor = str(anchor).strip().casefold()
        other = str(other).strip().casefold()
        by_anchor[anchor].append((pair_id, relation, other))

candidates = []
for anchor, items in by_anchor.items():
    e_items = [x for x in items if x[1] == "E"]
    d_items = [x for x in items if x[1] == "D"]

    for e_id, _, e_other in e_items:
        for d_id, _, d_other in d_items:
            similarity = SequenceMatcher(None, e_other, d_other).ratio()
            if e_id != d_id and similarity >= 0.5:
                candidates.append({
                    "共同短语": anchor,
                    "E的另一侧": e_other,
                    "方向关系的另一侧": d_other,
                    "文字相似度": round(similarity, 3),
                    "E_pair_id": e_id,
                    "方向_pair_id": d_id,
                })

result = (
    pd.DataFrame(candidates)
    .drop_duplicates(["共同短语", "E_pair_id", "方向_pair_id"])
    .sort_values("文字相似度", ascending=False)
)
output_file.parent.mkdir(parents=True, exist_ok=True)
result["approved"] = 0
result.to_csv(output_file, index=False, encoding="utf-8-sig")

print("候选对比组数：", len(result))
print("文件：", output_file)
print(result.head(10).to_string(index=False))