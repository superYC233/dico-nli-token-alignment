
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

import pandas as pd

from prepare_sixclass_data import (
    build_source_index,
    text_pair_key,
    source_dir
)

script_dir = QWEN_DIR
project_root = script_dir.parent.parent

label_map = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N'
}


def compare_predictions():
    four_path = (
        script_dir
        / 'outputs/qwen3_0p6b_abc/run04_r16/C'
        / 'predictions_with_scores.csv'
    )

    six_path = (
        script_dir
        / 'outputs/qwen3_0p6b_sixclass/run01/C'
        / 'predictions_with_scores.csv'
    )

    dev_path = (
        project_root
        / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
    )

    four = pd.read_csv(four_path, keep_default_na=False)
    six = pd.read_csv(six_path, keep_default_na=False)
    dev = pd.read_csv(dev_path, keep_default_na=False)

    # 确认两个文件都覆盖了同一批 Dev 样本，
    # 并且文件里的真实标签与官方 Dev 一致。
    gold_by_id = dev.set_index('instance_id')['label']

    for predictions in [four, six]:
        if predictions['instance_id'].duplicated().any():
            raise ValueError('预测文件中有重复 instance_id')

        if set(predictions['instance_id']) != set(dev['instance_id']):
            raise ValueError('预测文件与 Dev 的样本名单不一致')

        expected_gold = predictions['instance_id'].map(gold_by_id)

        if not predictions['gold_label'].eq(expected_gold).all():
            raise ValueError('预测文件中的真实标签与 Dev 不一致')

    # 比较两版经过 pair-aware 后的最终答案。
    four = four[
        ['instance_id', 'pair_prediction']
    ].rename(columns={
        'pair_prediction': 'four_prediction'
    })

    six = six[
        ['instance_id', 'pair_prediction']
    ].rename(columns={
        'pair_prediction': 'six_prediction'
    })

    df = dev.merge(
        four,
        on='instance_id',
        validate='one_to_one'
    )

    df = df.merge(
        six,
        on='instance_id',
        validate='one_to_one'
    )

    df['gold'] = df['label'].map(label_map)

    # 仅为事后分析补充 Dev N 的来源类别。
    # 这些诊断标签不加入训练数据。
    source_index = build_source_index([
        source_dir / 'PhrasIS.train.all.txt',
        source_dir / 'PhrasIS.test.all.txt'
    ])

    df['n_relation'] = ''

    for index, row in df[df['gold'] == 'N'].iterrows():
        key = text_pair_key(row['text1'], row['text2'])

        matches = source_index.get(key, [])
        relations = {
            item['relation']
            for item in matches
        }

        if len(relations) != 1:
            raise ValueError(
                f"N 来源匹配不唯一：{row['instance_id']}"
            )

        df.loc[index, 'n_relation'] = next(iter(relations))

    # 分别判断两版是否预测正确。
    df['four_correct'] = df['four_prediction'] == df['gold']
    df['six_correct'] = df['six_prediction'] == df['gold']

    # 原四类错，六类对。
    fixed = df[
        (~df['four_correct']) & df['six_correct']
    ].copy()

    # 原四类对，六类错。
    broken = df[
        df['four_correct'] & (~df['six_correct'])
    ].copy()

    # 统计各 N 子类中，有多少条被正确判成 N。
    summary = []

    for relation in ['SIMI', 'REL', 'OPPO']:
        group = df[df['n_relation'] == relation]

        summary.append({
            'relation': relation,
            'total': len(group),
            'four_correct': int(group['four_correct'].sum()),
            'six_correct': int(group['six_correct'].sum())
        })

    n_summary = pd.DataFrame(summary)

    output_dir = script_dir / 'outputs/sixclass_comparison/run01'
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        'instance_id',
        'pair_id',
        'text1',
        'text2',
        'gold',
        'n_relation',
        'four_prediction',
        'six_prediction'
    ]

    # 保存明细。
    df.to_csv(
        output_dir / 'all_comparison.csv',
        index=False,
        encoding='utf-8-sig'
    )

    fixed[columns].to_csv(
        output_dir / 'fixed.csv',
        index=False,
        encoding='utf-8-sig'
    )

    broken[columns].to_csv(
        output_dir / 'broken.csv',
        index=False,
        encoding='utf-8-sig'
    )

    n_summary.to_csv(
        output_dir / 'n_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print('总样本数：', len(df))
    print('修好的错误：', len(fixed))
    print('新增的错误：', len(broken))

    print('\n===== N 子类：正确判为 N 的数量 =====')
    print(n_summary.to_string(index=False))

    display_columns = [
        'text1',
        'text2',
        'gold',
        'n_relation',
        'four_prediction',
        'six_prediction'
    ]

    print('\n===== 修好的错误 =====')
    print(fixed[display_columns].to_string(index=False))

    print('\n===== 新增的错误 =====')
    print(broken[display_columns].to_string(index=False))

    print('\n结果目录：', output_dir)


if __name__ == '__main__':
    compare_predictions()