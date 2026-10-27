
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path
import pandas as pd
from analyze_errors import analyze_errors
def inspect_n_errors(error_path):
    errors = pd.read_csv(error_path, keep_default_na=False)

    # 从全部错误中，筛出真实为 N、预测为 F/B/E 的样本
    n_errors = errors[
        (errors['gold_short'] == 'N')
        & errors['prediction'].isin(['F', 'B', 'E'])
    ]

    print('\n===== 训练集 N 类错误核查 =====')
    print('错误总数：', len(n_errors))

    for label in ['F', 'B', 'E']:
        group = n_errors[n_errors['prediction'] == label]

        examples = group.sample(
            n=min(5, len(group)),
            random_state=42
        )

        print(f'\n===== N → {label}：共 {len(group)} 条 =====')

        for _, row in examples.iterrows():
            print('\ninstance_id:', row['instance_id'])
            print('Text 1:', row['text1'])
            print('Text 2:', row['text2'])
            print('真实标签: N')
            print('预测标签:', row['prediction'])
def export_hard_candidates(error_path, output_path):
    errors = pd.read_csv(error_path, keep_default_na=False)

    candidates = errors[
        (errors['gold_short'] == 'N')
        & errors['prediction'].isin(['F', 'B', 'E'])
    ].copy()

    # 留给人工填写，不自动把所有错误认定为困难样本
    candidates['review'] = ''
    candidates['reason'] = ''

    columns = [
        'instance_id',
        'pair_id',
        'text1',
        'text2',
        'gold_short',
        'prediction',
        'review',
        'reason'
    ]

    output_path = Path(output_path)

    # 避免再次运行时覆盖已经填写的审核内容
    if output_path.exists():
        print('审核表已存在，跳过保存：', output_path)
        return

    candidates[columns].to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    print('候选样本数：', len(candidates))
    print('审核表：', output_path)

def export_review_sample(candidate_path, output_path):
    output_path = Path(output_path)

    # 避免覆盖已经填写的核查表
    if output_path.exists():
        print('核查表已存在：', output_path)
        return

    candidates = pd.read_csv(candidate_path, keep_default_na=False)

    # 三种错误都抽一些，方便比较：
    # N → F 抽4条，N → B 抽3条，N → E 抽3条
    parts = []
    for prediction, count in [('F', 4), ('B', 3), ('E', 3)]:
        group = candidates[candidates['prediction'] == prediction]

        sampled = group.sample(
            n=min(count, len(group)),
            random_state=42
        )
        parts.append(sampled)

    review_table = pd.concat(parts, ignore_index=True)

    review_table.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    print('\n===== 待核查样本 =====')
    print(review_table[
        ['text1', 'text2', 'gold_short', 'prediction']
    ].to_string(index=False))

    print('\n核查表：', output_path)
if __name__ == '__main__':
    script_dir = QWEN_DIR
    project_root = script_dir.parent.parent

    train_analysis_dir = (
        script_dir
        / 'outputs/qwen3_0p6b_lora/run01/train_analysis'
    )

    analyze_errors(
        # 刚刚保存的训练集预测
        prediction_path=train_analysis_dir / 'predictions.csv',

        # 训练集原文，用来补充 text1、text2
        dev_path=(
            project_root
            / 'data/train/dico_nli_train_track1_participant_labeled.csv'
        ),

        # 错误分析结果的保存位置
        output_dir=train_analysis_dir / 'error_analysis'
    )

    inspect_n_errors(
        train_analysis_dir / 'error_analysis/all_errors.csv'
    )

    export_hard_candidates(
        error_path=(
            train_analysis_dir / 'error_analysis/all_errors.csv'
        ),
        output_path=train_analysis_dir / 'hard_candidates.csv'
    )

    export_review_sample(
        candidate_path=train_analysis_dir / 'hard_candidates.csv',
        output_path=train_analysis_dir / 'n_review_10.csv'
    )