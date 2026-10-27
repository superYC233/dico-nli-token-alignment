from datasets import concatenate_datasets

def build_train_dataset(train_dataset, mode):
    if mode == 'A':
        return train_dataset
    elif mode in ('B', 'C'):
        # extra = 原标签为 NEGATIVE_OTHER
        extra = train_dataset.filter(
            lambda row : row['label'] == 'NEGATIVE_OTHER'
        )

        # 把两列中同一行的文本组合
        train_pairs = set(
            zip(
                train_dataset['text1'],
                train_dataset['text2']
            )
        )

        # 检查每条 N 的反向文本对是否已经存在
        # 排除已存在的反向文本
        extra = extra.filter(
            lambda row : (row['text2'], row['text1']) not in train_pairs
        )

        # 去掉extra内部重复的对
        seen_pairs = set()  # 记录已经遇到的文本对
        keep_indices = []  # 记录需要保留的行号

        for index, row in enumerate(extra):
            pair = (row['text1'], row['text2'])

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                keep_indices.append(index)

        extra = extra.select(keep_indices)

        if mode == 'C':
            extra = extra.map(
                lambda row: {
                    'text1': row['text2'],
                    'text2': row['text1'],
                    'instance_id': row['instance_id'] + '__swap',
                    'pair_id': row['pair_id'] + '__swap'
                }
            )

        return concatenate_datasets([train_dataset, extra])
