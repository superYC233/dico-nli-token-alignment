
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM

from baseline import predict_batch
from data_utils import build_prompt

from tqdm import tqdm

def diagnose_pairs(model):
    ginger_text = 'A ginger and white cat'
    orange_original = 'A orange and white cat'
    orange_corrected = 'An orange and white cat'

    # 每条测试记录包含：名称、text1、text2
    cases = [
        {
            'name': '原始表达：正向',
            'text1': ginger_text,
            'text2': orange_original,
        },
        {
            'name': '原始表达：反向',
            'text1': orange_original,
            'text2': ginger_text,
        },
        {
            'name': '修正冠词：正向',
            'text1': ginger_text,
            'text2': orange_corrected,
        },
        {
            'name': '修正冠词：反向',
            'text1': orange_corrected,
            'text2': ginger_text,
        },
        {
            'name': '相同文本对照',
            'text1': ginger_text,
            'text2': ginger_text,
        },
    ]

    # 使用和正式评估相同的 Prompt
    prompts = [
        build_prompt(case['text1'], case['text2'])
        for case in cases
    ]

    # 使用和正式评估相同的预测函数，一次处理五条
    predictions = predict_batch(prompts, model)

    for case, prediction in zip(cases, predictions):
        print(f"\n===== {case['name']} =====")
        print('Text 1:', case['text1'])
        print('Text 2:', case['text2'])
        print('预期标签：E')
        print('实际预测：', repr(prediction))

    return predictions

def analyze_n_errors(prediction_path, dev_path, output_dir):
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    dev = pd.read_csv(dev_path, keep_default_na=False)

    # 预测文件没有文本，因此通过 instance_id 找回原始文本
    df = predictions.merge(
        dev[['instance_id', 'text1', 'text2']],
        on='instance_id',
        how='left',
        validate='one_to_one'
    )

    # 真实标签为 N，但模型预测成 F、B 或 E
    errors = df[
        (df['gold_label'] == 'NEGATIVE_OTHER')
        & df['prediction'].isin(['F', 'B', 'E'])
    ].copy()

    print('\n===== 真实 N，预测 F/B/E 的错误 =====')
    print('错误总数：', len(errors))
    print('各预测标签的数量：')
    print(errors['prediction'].value_counts())

    # 保存全部目标错误，方便后续查看
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors.to_csv(
        output_dir / 'n_to_fbe_errors.csv',
        index=False,
        encoding='utf-8-sig'
    )

    for label in ['F', 'B', 'E']:
        group = errors[errors['prediction'] == label]

        # 固定随机种子，让每次查看的样本保持一致
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

    return errors

def diagnose_n_reversal(model, batch_size=32, output_path=None):
    script_dir = QWEN_DIR
    project_root = script_dir.parent.parent

    dev_path = (
        project_root
        / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
    )

    dev = pd.read_csv(dev_path, keep_default_na=False)

    # 只取真实标签为 N 的 Dev 样本
    df = dev[dev['label'] == 'NEGATIVE_OTHER'].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError('没有找到 N 类样本，请检查数据文件')

    forward_predictions = []
    reverse_predictions = []

    model.eval()

    for start in tqdm(
        range(0, len(df), batch_size),
        desc='N 类双向诊断'
    ):
        batch = df.iloc[start:start + batch_size]

        # 原方向：text1 → text2
        forward_prompts = [
            build_prompt(text1, text2)
            for text1, text2 in zip(batch['text1'], batch['text2'])
        ]

        # 反方向：交换两段文本
        reverse_prompts = [
            build_prompt(text2, text1)
            for text1, text2 in zip(batch['text1'], batch['text2'])
        ]

        forward_predictions.extend(
            predict_batch(forward_prompts, model)
        )
        reverse_predictions.extend(
            predict_batch(reverse_prompts, model)
        )

    df['forward_prediction'] = forward_predictions
    df['reverse_prediction'] = reverse_predictions

    labels = ['F', 'B', 'E', 'N']
    reverse_map = {
        'F': 'B',
        'B': 'F',
        'E': 'E',
        'N': 'N'
    }

    df['forward_valid'] = df['forward_prediction'].isin(labels)
    df['reverse_valid'] = df['reverse_prediction'].isin(labels)
    df['valid_pair'] = df['forward_valid'] & df['reverse_valid']

    # 本次全部样本的真实标签都是 N
    # 反向正确性采用 N ↔ N 的诊断假设
    df['forward_correct'] = df['forward_prediction'].eq('N')
    df['reverse_correct'] = df['reverse_prediction'].eq('N')

    # 将反向预测转换到原方向的标签含义
    # 例如：反向预测 B，转换到原方向就是 F
    df['reverse_aligned_prediction'] = (
        df['reverse_prediction'].map(reverse_map)
    )

    # 完整反转兼容性：
    # F/B、B/F、E/E、N/N 都兼容；非法输出计为不兼容
    df['reversal_consistent'] = (
        df['valid_pair']
        & df['forward_prediction'].eq(
            df['reverse_aligned_prediction']
        )
    )

    # 只检查“是否预测为 N”有没有变化
    # 非法输出不参与这项统计
    df['n_membership_changed'] = (
        df['valid_pair']
        & df['forward_correct'].ne(df['reverse_correct'])
    )

    if output_path is None:
        output_path = (
            script_dir
            / 'outputs/qwen3_0p6b_lora/run01'
            / 'n_reversal_diagnosis.csv'
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print('\n===== Dev N 类双向诊断 =====')
    print('辅助诊断，非官方 SoftCons')
    print('总样本数：', len(df))
    print('原方向预测为 N：', int(df['forward_correct'].sum()))
    print('反方向预测为 N：', int(df['reverse_correct'].sum()))

    print('\n===== 四类预测分布 =====')
    distribution = pd.DataFrame({
        '原方向': df['forward_prediction'].value_counts().reindex(
            labels, fill_value=0
        ),
        '反方向': df['reverse_prediction'].value_counts().reindex(
            labels, fill_value=0
        )
    })
    print(distribution)

    print('\n===== 非法输出 =====')
    print('原方向非法输出数：', int((~df['forward_valid']).sum()))
    print('反方向非法输出数：', int((~df['reverse_valid']).sum()))

    invalid_rows = df[~df['valid_pair']]
    if not invalid_rows.empty:
        print(invalid_rows[
            ['instance_id', 'forward_prediction', 'reverse_prediction']
        ].to_string(index=False))

    # 以下四种情况和分组统计，只使用双向输出均合法的样本
    valid_df = df[df['valid_pair']]
    forward_n = valid_df['forward_correct']
    reverse_n = valid_df['reverse_correct']

    print('\n===== 双向判断组合：仅双向合法样本 =====')
    print('合法样本对数：', len(valid_df))
    print('两向都 N：', int((forward_n & reverse_n).sum()))
    print('仅原方向 N：', int((forward_n & ~reverse_n).sum()))
    print('仅反方向 N：', int((~forward_n & reverse_n).sum()))
    print('两向都非 N：', int((~forward_n & ~reverse_n).sum()))

    print('\n===== 按原方向是否正确分组：仅双向合法样本 =====')
    for correct, name in [
        (True, '原方向预测正确'),
        (False, '原方向预测错误')
    ]:
        group = valid_df[valid_df['forward_correct'] == correct]

        print(f'\n{name}：{len(group)} 条')
        if group.empty:
            continue

        changed = int(group['n_membership_changed'].sum())
        incompatible = int((~group['reversal_consistent']).sum())

        print(
            f'是否为 N 发生变化：{changed}/{len(group)}'
            f' = {changed / len(group):.2%}'
        )
        print(
            f'完整标签反转不兼容：{incompatible}/{len(group)}'
            f' = {incompatible / len(group):.2%}'
        )

    print('\n===== 原方向 × 反方向预测交叉表 =====')
    print('行是原方向，列是交换后的原始预测标签')
    transition = pd.crosstab(
        valid_df['forward_prediction'],
        valid_df['reverse_prediction']
    ).reindex(index=labels, columns=labels, fill_value=0)
    print(transition)

    print(
        '\n完整反转兼容率（非法输出计不兼容）：',
        f"{df['reversal_consistent'].mean():.2%}"
    )
    print('结果文件：', output_path)

    return df

if __name__ == '__main__':
    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'
    adapter_path = (
        'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_lora/run01'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto'
    )

    model = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_path
    ).to('cuda')

    model.eval()

    diagnose_n_reversal(model, batch_size=32)


