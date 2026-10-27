"""组合已有 Dev 预测，不重新训练、不加载 GPU 模型。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
qwen_dir = script_dir.parent
project_root = qwen_dir.parent.parent
sys.path.insert(0, str(project_root))

from evaluation_functions import score_files


qwen_path = (
    qwen_dir
    / 'outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
)
alignment_path = (
    qwen_dir
    / 'outputs/token_comparison/run01/alignment/epoch_4/predictions_with_scores.csv'
)
teacher_path = qwen_dir / 'outputs/teacher_scores/run01/dev_teacher_scores.csv'
teacher_pair_path = project_root / 'artifacts/final/dev_predictions_pair_aware.csv'
gold_path = project_root / 'data/dev/dico_nli_dev_track1_reference.csv'
output_dir = qwen_dir / 'outputs/hybrid_decode/run01'

labels = ['F', 'B', 'E', 'N']
long_labels = {
    'F': 'FORWARD_ENTAILMENT',
    'B': 'BACKWARD_ENTAILMENT',
    'E': 'EQUIVALENCE',
    'N': 'NEGATIVE_OTHER',
}


def load_aligned(path, instance_ids):
    frame = pd.read_csv(path, keep_default_na=False)
    if not frame['instance_id'].is_unique:
        raise ValueError(f'instance_id 存在重复：{path}')
    if set(frame['instance_id']) != set(instance_ids):
        raise ValueError(f'样本名单不一致：{path}')
    return frame.set_index('instance_id').loc[instance_ids].reset_index()


def four_probabilities(frame):
    scores = frame[['logit_F', 'logit_B', 'logit_E', 'logit_N']].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('候选分数包含 NaN 或无穷。')
    scores = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def decode_probabilities(frame, probabilities):
    """沿用 FB / BF / EE 三组合解码；单例先保留四类 argmax。"""
    prediction = np.array(labels)[probabilities.argmax(axis=1)]
    scores = np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0))
    combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]

    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise ValueError('本实验只处理 Track 1 的一行或两行分组。')
        first, second = group.index
        joint_scores = [
            scores[first, 0] + scores[second, 1],
            scores[first, 1] + scores[second, 0],
            scores[first, 2] + scores[second, 2],
        ]
        label0, label1 = combinations[int(np.argmax(joint_scores))]
        prediction[first] = label0
        prediction[second] = label1
    return prediction


def use_teacher_on_singletons(frame, pair_prediction, teacher_prediction):
    """仅按输入文件中的分组大小分流，不读取真实标签或错误名单。"""
    size = frame.groupby('pair_id')['instance_id'].transform('size')
    if not size.isin([1, 2]).all():
        raise ValueError('出现不支持的分组大小。')
    prediction = np.array(pair_prediction, dtype=object).copy()
    singleton = size.to_numpy() == 1
    prediction[singleton] = np.asarray(teacher_prediction)[singleton]
    return prediction


def main():
    original = pd.read_csv(qwen_path, keep_default_na=False)
    instance_ids = original['instance_id'].tolist()
    original = load_aligned(qwen_path, instance_ids)
    alignment = load_aligned(alignment_path, instance_ids)
    teacher = load_aligned(teacher_path, instance_ids)
    teacher_pair = load_aligned(teacher_pair_path, instance_ids)
    for other in (alignment, teacher):
        if not original['pair_id'].equals(other['pair_id']):
            raise ValueError('按 instance_id 对齐后，pair_id 不一致。')

    # 解码输入明确去掉金标；金标只由后面的官方评分函数读取。
    inputs = original[['instance_id', 'pair_id']].copy()
    qwen_probabilities = four_probabilities(original)
    alignment_probabilities = four_probabilities(alignment)
    qwen_prediction = decode_probabilities(inputs, qwen_probabilities)
    if not np.array_equal(qwen_prediction, original['pair_prediction'].to_numpy()):
        raise ValueError('原 Qwen 解码没有复现历史预测。')

    # 等权平均两套四类概率，再做同一个 pair-aware；没有搜索融合权重。
    fused_probabilities = (qwen_probabilities + alignment_probabilities) / 2
    fused_prediction = decode_probabilities(inputs, fused_probabilities)
    teacher_prediction = teacher['teacher_prediction'].to_numpy()

    methods = {
        'qwen_main': qwen_prediction,
        'deberta_reference': teacher_pair['label'].map(
            {value: key for key, value in long_labels.items()}
        ).to_numpy(),
        'qwen_alignment': fused_prediction,
        'qwen_deberta': use_teacher_on_singletons(
            inputs, qwen_prediction, teacher_prediction
        ),
        'qwen_alignment_deberta': use_teacher_on_singletons(
            inputs, fused_prediction, teacher_prediction
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = inputs.copy()
    comparison['group_size'] = inputs.groupby('pair_id')['instance_id'].transform('size')
    summary = []
    for method, prediction in methods.items():
        if not pd.Series(prediction).isin(labels).all():
            raise ValueError(f'{method} 存在非法输出。')
        comparison[method] = prediction
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({
            'instance_id': instance_ids,
            'label': [long_labels[label] for label in prediction],
        })
        prediction_path = method_dir / 'official_predictions.csv'
        submission.to_csv(prediction_path, index=False)
        report = score_files(str(gold_path), str(prediction_path), output_dir=str(method_dir))
        n_metrics = report.classification.per_label['NEGATIVE_OTHER']
        summary.append({
            'method': method,
            'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons,
            'hard_cons': report.hard_cons,
            'N_correct': n_metrics.true_positive,
            'FBE_correct': sum(
                report.classification.per_label[long_labels[label]].true_positive
                for label in ['F', 'B', 'E']
            ),
        })

    comparison.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    config = {
        'qwen_predictions': str(qwen_path),
        'alignment_predictions': str(alignment_path),
        'teacher_predictions': str(teacher_path),
        'deberta_reference_predictions': str(teacher_pair_path),
        'fusion_weights': [0.5, 0.5],
        'routing': 'group_size=2: Qwen pair-aware; group_size=1: raw DeBERTa',
        'status': 'Exploratory results on the existing, repeatedly used Dev; not independent test results.',
    }
    (output_dir / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(summary.to_string(index=False))
    print('结果目录：', output_dir)


if __name__ == '__main__':
    main()
