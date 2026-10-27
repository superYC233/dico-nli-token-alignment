import sys
from pathlib import Path
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM
# 当前文件夹
script_dir = Path(__file__).resolve().parent
# 项目文件夹
project_root = script_dir.parent.parent
# 将项目根目录加入 Python 的模块搜索列表
sys.path.insert(0, str(project_root))
from compare_decoding import evaluate_with_scores
from baseline import score_predictions

model_path = 'D:/LLM/model/Qwen2.5-3B-Instruct'

adapter_path = (
    'D:/LLM/train_model/DiCo-NLI/'
    'qwen2p5_3b_abc/run01_best/C/best_adapter'
)

def load_lora_model(model_path, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto'
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path
    ).to('cuda')

    model.eval()
    return model

def pair_aware_decode(df):
    df = df.copy()

    df['pair_prediction'] = df['prediction']

    for pair_id, group in df.groupby('pair_id'):
        if len(group) != 2:
            continue

        row0 = group.iloc[0]
        row1 = group.iloc[1]

        scores = [
            row0['logit_F'] + row1['logit_B'],  # FB
            row0['logit_B'] + row1['logit_F'],  # BF
            row0['logit_E'] + row1['logit_E']  # EE
        ]

        # 顺序与 scores 的三个分数对应
        combinations = [
            ('F', 'B'),
            ('B', 'F'),
            ('E', 'E')
        ]

        best_index = scores.index(max(scores))

        label0, label1 = combinations[best_index]

        df.loc[group.index[0], 'pair_prediction'] = label0
        df.loc[group.index[1], 'pair_prediction'] = label1

    return df

def score_methods(df, output_dir):
    methods = [
        ('independent', 'prediction'),
        ('pair_aware', 'pair_prediction')
    ]

    for method, prediction_column in methods:
        score_records = []

        for _, row in df.iterrows():
            score_records.append({
                'instance_id': row['instance_id'],
                'prediction': row[prediction_column]
            })

        score_dir = output_dir / method
        score_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n===== {method} =====')

        report = score_predictions(
            score_records,
            str(score_dir)
        )

        if report is not None:
            print('SoftCons:', report.soft_cons)
            print('HardCons:', report.hard_cons)

            n_metrics = report.classification.per_label['NEGATIVE_OTHER']
            print('N-F1:', n_metrics.f1)


if __name__ == '__main__':
    model = load_lora_model(model_path, adapter_path)

    # 复用已有批量预测函数，固定 batch_size=16
    records = evaluate_with_scores(
        model,
        batch_size=8,
        split='dev'
    )

    df = pd.DataFrame(records)

    df = pair_aware_decode(df)

    output_dir = (
            script_dir / 'outputs/qwen2p5_3b_abc/run01_best/C'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / 'predictions_with_scores.csv',
        index=False,
        encoding='utf-8-sig'
    )

    print('保存样本数：', len(df))
    print('结果目录：', output_dir)

    score_methods(df, output_dir)