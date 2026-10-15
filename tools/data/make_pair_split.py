
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

# todo 1. 按文本对分组，创建不含重复文本对的内部训练集和验证集

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_DIR = (PROJECT_ROOT)


def create_clean_pair_split():
    # 读取官方训练集
    data = pd.read_csv(
        PROJECT_DIR
        / "data/train/dico_nli_train_track1_participant_labeled.csv"
    )

    # 把 (A, B) 和 (B, A) 变成同一个分组键。
    # 大小写和两端空格的差别也忽略。
    pair_keys = pd.Series(
        [
            tuple(sorted((
                str(text1).strip().casefold(),
                str(text2).strip().casefold(),
            )))
            for text1, text2 in zip(data["text1"], data["text2"])
        ],
        index=data.index,
    )

    # GroupShuffleSplit 使用数字编号作为分组
    group_ids = pd.factorize(pair_keys)[0]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_indices, valid_indices = next(
        splitter.split(data, groups=group_ids)
    )

    internal_train = data.iloc[train_indices].copy()
    internal_valid = data.iloc[valid_indices].copy()

    # 检查 1：同一个 pair_id 不能同时出现在两边
    pair_id_overlap = (
        set(internal_train["pair_id"])
        & set(internal_valid["pair_id"])
    )

    # 检查 2：相同文本对也不能同时出现在两边
    train_keys = set(pair_keys.iloc[train_indices])
    valid_keys = set(pair_keys.iloc[valid_indices])
    text_pair_overlap = train_keys & valid_keys

    print("内部训练集行数：", len(internal_train))
    print("内部验证集行数：", len(internal_valid))
    print("重复的 pair_id 数：", len(pair_id_overlap))
    print("重复文本对数：", len(text_pair_overlap))
    print("内部验证集标签数量：")
    print(internal_valid["label"].value_counts())

    # 检查不通过就停止，不写入文件
    assert len(pair_id_overlap) == 0
    assert len(text_pair_overlap) == 0

    # 使用新目录，保留原来的 data/internal_split
    output_dir = PROJECT_DIR / "data/internal_split_clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_train.to_csv(
        output_dir / "train.csv",
        index=False,
    )
    internal_valid.to_csv(
        output_dir / "valid.csv",
        index=False,
    )

    print("新划分已保存到：", output_dir)


if __name__ == "__main__":
    create_clean_pair_split()