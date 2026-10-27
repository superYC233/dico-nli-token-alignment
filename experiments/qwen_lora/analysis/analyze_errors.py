
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

import pandas as pd


label_map = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}


def analyze_errors(prediction_path, dev_path, output_dir):
    # 1. 读取预测记录和 DEV 原文
    predictions = pd.read_csv(
        prediction_path,
        keep_default_na=False
    )
    dev = pd.read_csv(
        dev_path,
        keep_default_na=False
    )

    # 2. 根据 instance_id，把 text1、text2 补充到预测记录中
    # validate 保证两张表中每个 instance_id 都只有一行
    df = predictions.merge(
        dev[['instance_id', 'text1', 'text2']],
        on='instance_id',
        how='left',
        validate='one_to_one',
    )

    # 3. 将真实标签转成短标签，方便和预测直接比较
    df['gold_short'] = df['gold_label'].map(label_map)
    df['is_error'] = df['gold_short'] != df['prediction']

    # 全部分类错误
    all_errors = df[df['is_error']].copy()

    # 真实为 E、预测为 N：两个条件必须同时满足
    e_to_n_errors = df[
        (df['gold_short'] == 'E')
        & (df['prediction'] == 'N')
    ].copy()

    print('总样本数：', len(df))
    print('全部错误数：', len(all_errors))
    print('E → N 错误数：', len(e_to_n_errors))
    print('涉及的 pair_id 数：', e_to_n_errors['pair_id'].nunique())

    # 统计各种错误方向，例如真实 E、预测 N
    print('\n错误方向统计（左侧真实标签，右侧预测标签）：')
    print(
        all_errors.groupby(['gold_short', 'prediction'])
        .size()
        .sort_values(ascending=False)
    )

    # 4. 保存错误记录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        'instance_id',
        'pair_id',
        'text1',
        'text2',
        'gold_label',
        'gold_short',
        'prediction',
        'is_error',
    ]

    all_errors[columns].to_csv(
        output_dir / 'all_errors.csv',
        index=False,
        encoding='utf-8-sig',
    )

    e_to_n_errors[columns].to_csv(
        output_dir / 'e_to_n_errors.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # 5. 从发生 E → N 错误的组中，抽取 5 个不同的组
    pair_ids = e_to_n_errors['pair_id'].drop_duplicates()

    selected_pair_ids = pair_ids.sample(
        n=min(5, len(pair_ids)),
        random_state=42,
    )

    # 回到完整表 df，取出所选组的两个方向
    # 因此这里也可能包含同组中预测正确的那一行
    samples = df[
        df['pair_id'].isin(selected_pair_ids)
    ].sort_values(['pair_id', 'instance_id'])

    samples[columns].to_csv(
        output_dir / 'e_to_n_pair_examples.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # 6. 分组打印，避免 Pandas 把长文本省略
    for pair_id, group in samples.groupby('pair_id'):
        print(f'\n===== {pair_id} =====')

        for _, row in group.iterrows():
            print('\ninstance_id:', row['instance_id'])
            print('Text 1:', row['text1'])
            print('Text 2:', row['text2'])
            print('真实标签:', row['gold_short'])
            print('预测标签:', row['prediction'])

    print('\n分析文件已保存到：', output_dir)

    return e_to_n_errors

def compare_runs(old_prediction_path, new_prediction_path,
                 dev_path, output_dir):
    old = pd.read_csv(old_prediction_path, keep_default_na=False)
    new = pd.read_csv(new_prediction_path, keep_default_na=False)
    dev = pd.read_csv(dev_path, keep_default_na=False)

    # 确保比较的是同一批样本，且每个 ID 只出现一次
    expected_ids = set(dev['instance_id'])

    for name, data in [('DEV', dev), ('旧预测', old), ('新预测', new)]:
        if data['instance_id'].duplicated().any():
            raise ValueError(f'{name} 中存在重复 instance_id')

        if set(data['instance_id']) != expected_ids:
            raise ValueError(f'{name} 的样本 ID 与 DEV 不一致')

    # 分别给两次预测改名，避免合并后混淆
    old = old[['instance_id', 'prediction']].rename(
        columns={'prediction': 'old_prediction'}
    )
    new = new[['instance_id', 'prediction']].rename(
        columns={'prediction': 'new_prediction'}
    )

    # 按 ID 对齐，而不是假设两个文件的行顺序相同
    df = dev.merge(
        old, on='instance_id', validate='one_to_one'
    ).merge(
        new, on='instance_id', validate='one_to_one'
    )

    df['gold_short'] = df['label'].map(label_map)
    df['old_correct'] = df['old_prediction'] == df['gold_short']
    df['new_correct'] = df['new_prediction'] == df['gold_short']

    # 旧错新对：修好的错误
    fixed = df[
        (~df['old_correct']) & df['new_correct']
    ].copy()

    # 旧对新错：新增的错误
    new_errors = df[
        df['old_correct'] & (~df['new_correct'])
    ].copy()

    # 两次都错，预测标签可能相同，也可能不同
    persistent_errors = df[
        (~df['old_correct']) & (~df['new_correct'])
    ].copy()

    # 新出现的 E→B：旧模型没有预测 B，新模型预测成 B
    # 其中也可能包含旧模型原本就预测错的样本
    new_e_to_b = df[
        (df['gold_short'] == 'E')
        & (df['new_prediction'] == 'B')
        & (df['old_prediction'] != 'B')
    ].copy()

    print('\n===== 两次实验逐条比较 =====')
    print('总样本数：', len(df))
    print('旧模型错误数：', (~df['old_correct']).sum())
    print('新模型错误数：', (~df['new_correct']).sum())
    print('修好的错误：', len(fixed))
    print('新增的错误：', len(new_errors))
    print('两次都错：', len(persistent_errors))
    print('新出现的 E→B：', len(new_e_to_b))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / 'all_comparisons.csv',
        index=False,
        encoding='utf-8-sig'
    )

    groups = [
        ('修好的错误', fixed, 'fixed_errors.csv'),
        ('新增的错误', new_errors, 'new_errors.csv'),
        ('两次都错', persistent_errors, 'persistent_errors.csv'),
        ('新出现的 E→B', new_e_to_b, 'new_e_to_b.csv'),
    ]

    for name, group, filename in groups:
        group.to_csv(
            output_dir / filename,
            index=False,
            encoding='utf-8-sig'
        )

        # 两次都错的样本先保存，本轮重点打印发生变化的错误
        if name == '两次都错':
            continue

        print(f'\n===== {name}：抽取最多 5 条 =====')

        examples = group.sample(
            n=min(5, len(group)),
            random_state=42
        )

        for _, row in examples.iterrows():
            print('\ninstance_id:', row['instance_id'])
            print('Text 1:', row['text1'])
            print('Text 2:', row['text2'])
            print('真实标签:', row['gold_short'])
            print('旧预测:', row['old_prediction'])
            print('新预测:', row['new_prediction'])

    print('\n比较结果保存到：', output_dir)

    return df
if __name__ == '__main__':
    # 根据脚本位置定位路径，不依赖 PyCharm 的工作目录
    script_dir = QWEN_DIR
    project_root = script_dir.parent.parent

    run_dir = (
        script_dir / 'outputs' / 'qwen3_0p6b_lora' / 'run02_prompt'
    )

    analyze_errors(
        prediction_path=run_dir / 'predictions.csv',
        dev_path=(
            project_root
            / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
        ),
        output_dir=run_dir / 'error_analysis',
    )

    compare_runs(
        old_prediction_path=(
            script_dir
            / 'outputs/qwen3_0p6b_lora/run01/predictions.csv'
        ),
        new_prediction_path=(
            script_dir
            / 'outputs/qwen3_0p6b_lora/run02_prompt/predictions.csv'
        ),
        dev_path=(
            project_root
            / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
        ),
        output_dir=(
            script_dir
            / 'outputs/qwen3_0p6b_lora/run02_prompt/run_comparison'
        )
    )