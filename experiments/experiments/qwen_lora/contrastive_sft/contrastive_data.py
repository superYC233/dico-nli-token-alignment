"""只增加同源分组编号；不增加、删除训练行，也不改变标签。"""


def add_contrast_groups(train_dataset):
    groups = {}
    group_ids = []

    for row in train_dataset:
        # 忽略顺序后相同的文本对，视为同源。
        # 这样原始 N 和 C 中新增的反向 N，也会得到同一个编号。
        text1 = ' '.join(row['text1'].casefold().split())
        text2 = ' '.join(row['text2'].casefold().split())
        key = tuple(sorted([text1, text2]))

        if key not in groups:
            groups[key] = len(groups)

        group_ids.append(groups[key])

    return train_dataset.add_column('contrast_group', group_ids)

