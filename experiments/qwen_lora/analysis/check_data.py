
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import pandas as pd

train_df = pd.read_csv('../../data/train/dico_nli_train_track1_participant_labeled.csv')
dev_df = pd.read_csv('../../data/dev/dico_nli_dev_track1_participant_labeled.csv')


def check_basic_info(df, name):
    print(f'\n===== {name} =====')
    # 打印形状、列名、前3行、标签数量
    print(df.shape)
    print(df.columns)
    print(df.head(3))

    # 查看标签数量和比例
    print(df['label'].value_counts(dropna=False)) # dropna=False：缺失值也要统计。
    print(df['label'].value_counts(normalize=True,dropna=False)) # normalize=True：把数量换成比例

    # 统计每一列的缺失值数量
    print(df.isnull().sum())

def check_pair(df, name):
    print(f'\n===== {name} =====')

    pair_size = df.groupby('pair_id').size()
    print(pair_size.value_counts().sort_values())

def check_single(df, name):
    print(f'\n===== {name} =====')

    pair_size = df.groupby('pair_id').size()
    single_pairs = pair_size[pair_size == 1]
    pair_ids = single_pairs.index
    df = df[df['pair_id'].isin(pair_ids)]
    print(df.shape)
    print(df['label'].value_counts(dropna=False))

def check_double(df, name):
    print(f'\n===== {name} =====')
    pair_size = df.groupby('pair_id').size()
    single_pairs = pair_size[pair_size == 2]
    pair_ids = single_pairs.index
    df = df[df['pair_id'].isin(pair_ids)]

    for pair_id in pair_ids:
        group = df[df['pair_id'] == pair_id]
        row0 = group.iloc[0]
        row1 = group.iloc[1]
        if not (
                row0['text1'] == row1['text2']
                and row0['text2'] == row1['text1']
        ):
            print('不符合：', pair_id)

        reverse_map = {
            'FORWARD_ENTAILMENT': 'BACKWARD_ENTAILMENT',
            'BACKWARD_ENTAILMENT': 'FORWARD_ENTAILMENT',
            'EQUIVALENCE': 'EQUIVALENCE'
        }

        excepted_label = reverse_map[row0['label']]

        if row1['label'] != excepted_label:
            print('不符合：', pair_id)

def check_split_overlap(train_df, dev_df):
    print('\n===== TRAIN / DEV 重合检查 =====')

    train_pair_ids = set(train_df['pair_id'])
    dev_pair_ids = set(dev_df['pair_id'])

    common_pair_ids = train_pair_ids & dev_pair_ids
    print('重合的 pair_id 数量：', len(common_pair_ids))

    if common_pair_ids:
        print('重合 ID 示例：', sorted(common_pair_ids)[:10])


def check_text_overlap(train_df, dev_df):
    print('\n===== TRAIN / DEV 文本对重合检查 =====')

    train_pairs = set(zip(train_df['text1'], train_df['text2']))
    dev_pairs = set(zip(dev_df['text1'], dev_df['text2']))

    train_reverse_pairs = set(zip(train_df['text2'], train_df['text1']))

    same_overlap = dev_pairs & train_pairs
    reverse_overlap = dev_pairs & train_reverse_pairs
    any_overlap = same_overlap | reverse_overlap

    print('同向重合数量：', len(same_overlap))
    print('反向重合数量：', len(reverse_overlap))
    print('任一方向重合数量：', len(any_overlap))
    print('重合示例：', sorted(any_overlap)[:5])

if __name__ == '__main__':
    # check_basic_info(train_df, 'TRAIN')
    # check_basic_info(dev_df, 'DEV')

    # check_pair(train_df, 'train')
    # check_pair(dev_df, 'dev')

    # check_single(train_df, 'train')
    # check_single(dev_df, 'dev')

    # check_double(train_df, 'train')
    # check_double(dev_df, 'dev')

    # check_split_overlap(train_df, dev_df)

    check_text_overlap(train_df, dev_df)