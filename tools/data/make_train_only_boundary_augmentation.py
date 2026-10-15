"""Create a small, auditable F/B-versus-E augmentation from training labels only.

DEV labels and DEV errors are never loaded. Held-out text/IDs are read solely
to reject overlap; they do not decide the content or labels of examples.
"""

# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()


from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = PROJECT_ROOT
TRAIN_PATH = PROJECT_DIR / "data/internal_split_clean/train.csv"
VALID_PATH = PROJECT_DIR / "data/internal_split_clean/valid.csv"
OFFICIAL_DEV_PATH = (
    PROJECT_DIR / "data/dev/dico_nli_dev_track1_participant_labeled.csv"
)
OFFICIAL_TRAIN_PATH = (
    PROJECT_DIR / "data/train/dico_nli_train_track1_participant_labeled.csv"
)
OUTPUT_PATH = (
    PROJECT_DIR / "data/augmentation/train_only_boundary_24pairs.csv"
)

# kind: EE = two training E edges; ED = E plus directional training edge;
# manual = authored paraphrase/restriction based on the listed training edge.
# Only clean internal training pair_ids may appear here. No held-out example
# was inspected to write or label these candidates.
CURATED = [
    # E: synonym/paraphrase changes that should preserve both directions.
    ("E", "amid snow", "within the snow", "EE", "0000749", "0001199"),
    ("E", "in the snow", "in snow", "EE", "0000749", "0000824"),
    ("E", "Nobel author Doris Lessing", "Nobel Prize winning author Doris Lessing", "EE", "0000691", "0000961"),
    ("E", "entering a transport terminal", "moving into a transport terminal", "EE", "0001249", "0001296"),
    ("E", "the canine with black-and-white fur", "a canine with black-and-white fur", "EE", "0000797", "0000800"),
    ("E", "eats", "consuming food", "EE", "0000819", "0001135"),
    ("E", "Volcanic eruption", "an eruption of a volcano", "manual", "0000690", ""),
    ("E", "a decade of imprisonment", "ten years in prison", "manual", "0001064", ""),
    ("E", "in the car", "inside the car", "manual", "0000771", ""),
    ("E", "The old lady", "the elderly woman", "manual", "0001092", ""),
    ("E", "the male child", "the young male child", "EE", "0000840", "0001299"),
    ("E", "is responding", "is reacting", "manual", "0000652", ""),
    # F: the left phrase adds information; flipping it produces B.
    ("F", "A black , brown and white dog", "a canine with black-and-white fur", "ED", "0000800", "0001524"),
    ("F", "Grey and white cat", "a feline", "ED", "0001200", "0000634"),
    ("F", "British Nobel-Winning Author Doris Lessing", "Nobel laureate Doris Lessing", "ED", "0000961", "0001366"),
    ("F", "in a snowy field", "in the snow", "ED", "0000798", "0000156"),
    ("F", "through a green grassy field", "over grass", "ED", "0000812", "0001453"),
    ("F", "Paul Walker Fast and Furious star", "Hollywood star Paul Walker", "ED", "0000697", "0001367"),
    ("F", "a decade of imprisonment", "a period of imprisonment", "manual", "0001064", ""),
    ("F", "the canine with black-and-white fur", "a canine", "manual", "0000797", ""),
    ("F", "a decade of imprisonment", "imprisonment", "manual", "0001064", ""),
    ("F", "Volcanic eruption", "an eruption", "manual", "0000690", ""),
    ("F", "A black , brown and white dog", "a multicolored canine", "manual", "0001524", ""),
    ("F", "Grey and white cat", "a multicolored feline", "manual", "0000634", ""),
]


def norm(value):
    """Normalize case, whitespace, and common punctuation for overlap checks."""
    return re.sub(r"[^\w]+", "", str(value).casefold(), flags=re.UNICODE)


def pair_key(a, b):
    return tuple(sorted((norm(a), norm(b))))


def source_rows(training, short_id):
    pair_id = f"dico_phrasis_{short_id}"
    rows = training[training["pair_id"] == pair_id]
    if len(rows) != 2:
        raise ValueError(f"Source {pair_id} is absent or is not a complete pair")
    return rows


def source_relation(rows):
    first = rows.iloc[0]
    if set(rows["label"]) == {"EQUIVALENCE"}:
        return "E", first["text1"], first["text2"]
    if set(rows["label"]) != {"FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"}:
        raise ValueError(f"Bad source labels for {first['pair_id']}")
    forward = rows.loc[rows["label"] == "FORWARD_ENTAILMENT"].iloc[0]
    return "D", forward["text1"], forward["text2"]


def check_derivation(kind, left, right, first, second):
    """Verify every mechanically derived label against its training edges."""
    type1, a, b = source_relation(first)
    if kind == "manual":
        return
    type2, c, d = source_relation(second)
    if kind == "EE":
        if type1 != "E" or type2 != "E":
            raise ValueError("EE must have two equivalence sources")
        shared = {norm(a), norm(b)} & {norm(c), norm(d)}
        possible = {
            pair_key(x, y)
            for x in (a, b) for y in (c, d)
            if norm(x) not in shared and norm(y) not in shared
        }
        if len(shared) != 1 or pair_key(left, right) not in possible:
            raise ValueError(f"Invalid EE derivation: {left} / {right}")
    elif kind == "ED":
        if type1 != "E" or type2 != "D":
            raise ValueError("ED requires E source first and directional source second")
        shared = {norm(a), norm(b)} & {norm(c), norm(d)}
        if len(shared) != 1:
            raise ValueError(f"Invalid ED anchor: {left} / {right}")
        alternate = b if norm(a) in shared else a
        derived = (alternate, d) if norm(c) in shared else (c, alternate)
        if (norm(left), norm(right)) != tuple(map(norm, derived)):
            raise ValueError(f"Invalid ED direction: {left} -> {right}")
    else:
        raise ValueError(f"Unknown derivation kind: {kind}")


def build_dataset():
    train = pd.read_csv(TRAIN_PATH)
    original_train = pd.read_csv(
        OFFICIAL_TRAIN_PATH, usecols=["pair_id", "text1", "text2"]
    )
    # Exclusion audit only. In particular, no DEV label column is read.
    held = [
        pd.read_csv(path, usecols=["pair_id", "text1", "text2"])
        for path in (VALID_PATH, OFFICIAL_DEV_PATH)
    ]
    held_ids = set().union(*(set(df["pair_id"]) for df in held))
    held_texts = {
        norm(value)
        for df in held for col in ("text1", "text2") for value in df[col]
    }
    original_keys = {
        pair_key(row.text1, row.text2)
        for row in original_train.itertuples(index=False)
    }
    original_ids = set(original_train["pair_id"])
    original_instance_ids = set(train["instance_id"])
    seen_keys = set()
    output = []
    counts = {"E": 0, "F": 0}

    for number, (relation, left, right, kind, id1, id2) in enumerate(CURATED, 1):
        if relation not in counts or not left.strip() or not right.strip():
            raise ValueError(f"Bad curated item {number}")
        first = source_rows(train, id1)
        second = source_rows(train, id2) if id2 else None
        check_derivation(kind, left, right, first, second)
        source_ids = [first.iloc[0]["pair_id"]]
        if second is not None:
            source_ids.append(second.iloc[0]["pair_id"])
        source_texts = [
            value for df in (first, second) if df is not None
            for col in ("text1", "text2") for value in df[col]
        ]
        if any(norm(value) in held_texts for value in [left, right, *source_texts]):
            raise ValueError(f"Held-out text occurs in item {number} or its sources")
        if any(pair_id in held_ids for pair_id in source_ids):
            raise ValueError(f"Held-out pair_id occurs in sources for item {number}")

        key = pair_key(left, right)
        if norm(left) == norm(right) or key in original_keys or key in seen_keys:
            raise ValueError(f"Duplicate or already-trained pair at item {number}")
        seen_keys.add(key)
        pair_id = f"train_only_boundary_{number:04d}"
        if pair_id in original_ids or pair_id in held_ids:
            raise ValueError(f"pair_id collision: {pair_id}")
        forward_label = (
            "EQUIVALENCE" if relation == "E" else "FORWARD_ENTAILMENT"
        )
        reverse_label = (
            "EQUIVALENCE" if relation == "E" else "BACKWARD_ENTAILMENT"
        )
        for direction, a, b, label in (
            ("forward", left, right, forward_label),
            ("reverse", right, left, reverse_label),
        ):
            instance_id = f"{pair_id}__en-en__{direction}"
            if instance_id in original_instance_ids:
                raise ValueError(f"instance_id collision: {instance_id}")
            output.append({
                "instance_id": instance_id,
                "pair_id": pair_id,
                "text1_lang": "en",
                "text2_lang": "en",
                "text1": a,
                "text2": b,
                "label": label,
                "source_pair_ids": ";".join(source_ids),
                "derivation": kind,
            })
        counts[relation] += 1

    if counts != {"E": 12, "F": 12}:
        raise ValueError(f"Unexpected balance: {counts}")
    result = pd.DataFrame(output)
    if result["instance_id"].duplicated().any():
        raise ValueError("Duplicate generated instance IDs")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("增强样本：24 组 / 48 行（E 12 组，F/B 12 组）")
    print("来源：内部训练集金标关系；DEV 标签和错误样本均未读取")
    print("检查：生成文本、来源文本、pair_id 与留出数据均无规范化完全重合")
    print("增强文件：", OUTPUT_PATH)
    return result


if __name__ == "__main__":
    build_dataset()
