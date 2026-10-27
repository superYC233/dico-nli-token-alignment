"""分析原版 0.9248（seed 42、epoch 7），不训练、不加载模型。

右键运行即可。复用此前导出的原版 logits，并逐条核对官方历史预测。
报告保存在 artifacts/original_error_analysis，不修改原训练代码和权重。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from evaluation_functions import score_files


DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
REFERENCE_PATH = PROJECT_ROOT / 'artifacts/final/dev_predictions_pair_aware.csv'
SCORES_PATH = PROJECT_ROOT / 'experiments/qwen_lora/outputs/teacher_scores/run01/dev_teacher_scores.csv'
OUTPUT_DIR = PROJECT_ROOT / 'artifacts/original_error_analysis'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
TO_SHORT = dict(zip(FULL_LABELS, LABELS))
TO_FULL = dict(zip(LABELS, FULL_LABELS))
COMBINATIONS = [('F', 'B'), ('B', 'F'), ('E', 'E')]
EXPECTED_F1 = 0.9247935854136166


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def softmax(scores):
    # 先减最大值，防止 exp 溢出。支持单个分数向量或批量向量。
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def normalize(text):
    return ' '.join(str(text).casefold().split())


def load_scores(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'没有找到原版分数文件：{path}\n'
            '请使用 --scores 指定原版 epoch_7 的四类 logits CSV，不能用词汇版替代。'
        )
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    cached = pd.read_csv(path, keep_default_na=False)
    columns = ['instance_id', 'pair_id', 'text1', 'text2', 'label']
    for frame in [dev, cached]:
        if not frame['instance_id'].is_unique:
            raise ValueError('instance_id 重复，无法逐条对齐。')
    if set(dev['instance_id']) != set(cached['instance_id']):
        raise ValueError('分数文件与官方 Dev 的 instance_id 不一致。')

    # 始终按官方 Dev 的原始顺序对齐，不能依赖另一个 CSV 的行号。
    cached = cached.set_index('instance_id').loc[dev['instance_id']].reset_index()
    for column in columns:
        if not cached[column].equals(dev[column]):
            raise ValueError(f'分数文件的 {column} 与官方 Dev 不一致。')
    frame = dev[columns].copy()
    frame['gold'] = frame['label'].map(TO_SHORT)
    if frame['gold'].isna().any():
        raise ValueError('Dev 存在未知标签。')
    logits = cached[[f'logit_{label}' for label in LABELS]].to_numpy(dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError('logits 存在 NaN 或无穷。')
    for index, label in enumerate(LABELS):
        frame[f'logit_{label}'] = logits[:, index]
    frame['prediction'] = np.array(LABELS)[logits.argmax(axis=1)]
    if 'teacher_prediction' in cached:
        if not (frame['prediction'] == cached['teacher_prediction']).all():
            raise ValueError('缓存的独立标签与 logits 的 argmax 不一致。')
    probabilities = softmax(logits)
    frame['independent_max_probability'] = probabilities.max(axis=1)
    sorted_logits = np.sort(logits, axis=1)
    frame['independent_top1_minus_top2'] = sorted_logits[:, -1] - sorted_logits[:, -2]
    return frame


def pair_aware_decode(frame):
    frame = frame.copy().reset_index(drop=True)
    frame['pair_prediction'] = frame['prediction']
    pairs = []
    for pair_id, group in frame.groupby('pair_id', sort=False):
        if len(group) == 1:
            if group.iloc[0]['gold'] != 'N':
                raise ValueError(f'{pair_id}：发现非 N 单例。')
            continue
        if len(group) != 2:
            raise ValueError(f'{pair_id}：应该为两行。')
        row0, row1 = group.iloc[0], group.iloc[1]
        if row0['text1'] != row1['text2'] or row0['text2'] != row1['text1']:
            raise ValueError(f'{pair_id}：两行文本没有互换。')
        gold_combo = (row0['gold'], row1['gold'])
        if gold_combo not in COMBINATIONS:
            raise ValueError(f'{pair_id}：金标不是 FB/BF/EE。')
        scores = np.array([
            row0['logit_F'] + row1['logit_B'],
            row0['logit_B'] + row1['logit_F'],
            row0['logit_E'] + row1['logit_E'],
        ])
        choice = int(scores.argmax())
        prediction = COMBINATIONS[choice]
        frame.loc[group.index, 'pair_prediction'] = list(prediction)
        gold_index = COMBINATIONS.index(gold_combo)
        sorted_scores = np.sort(scores)
        pairs.append({
            'pair_id': pair_id,
            'instance_id_0': row0['instance_id'],
            'instance_id_1': row1['instance_id'],
            'text1': row0['text1'], 'text2': row0['text2'],
            # 金标与预测保持相同的行顺序，禁止只对金标排序。
            'gold_combination': ''.join(gold_combo),
            'independent_combination': row0['prediction'] + row1['prediction'],
            'predicted_combination': ''.join(prediction),
            'score_FB': scores[0], 'score_BF': scores[1], 'score_EE': scores[2],
            'gold_minus_chosen_joint_score': scores[gold_index] - scores[choice],
            'chosen_minus_runner_up_joint_score': sorted_scores[-1] - sorted_scores[-2],
            # 仅表示在三个合法组合内归一化后的值，不是校准后的正确概率。
            'conditional_legal_probability': softmax(scores)[choice],
            'correct': prediction == gold_combo,
        })
    return frame, pd.DataFrame(pairs)


def verify_history(frame):
    reference = pd.read_csv(REFERENCE_PATH, keep_default_na=False)
    if not reference['instance_id'].is_unique:
        raise ValueError('历史预测存在重复 ID。')
    if set(reference['instance_id']) != set(frame['instance_id']):
        raise ValueError('历史预测样本不完整。')
    expected = reference.set_index('instance_id')['label'].map(TO_SHORT)
    expected = expected.reindex(frame['instance_id']).to_numpy()
    different = int((expected != frame['pair_prediction'].to_numpy()).sum())
    if different:
        raise ValueError(f'与原版历史预测有 {different} 行不同，停止报告，避免分析错误版本。')


def error_kind(gold, predicted):
    if gold == predicted:
        return '正确'
    if gold == 'N':
        return 'N → F/B' if predicted in ('F', 'B') else 'N → E'
    if predicted == 'N':
        return 'F/B/E → N'
    if gold == 'E':
        return 'E → F/B'
    if predicted == 'E':
        return 'F/B → E'
    return 'F ↔ B'


def add_n_relations(frame):
    # 只分析对称的 N 子类；不把 BACK/FORW 机械复制到反方向。
    relations = {}
    for name in ['PhrasIS.train.all.txt', 'PhrasIS.test.all.txt']:
        path = PROJECT_ROOT / 'data/phrasis_sources' / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            fields = line.split('\t')
            if len(fields) < 4:
                continue
            relation = fields[1].strip()
            first, second = normalize(fields[2]), normalize(fields[3])
            # 保留所有来源标签，不能用 setdefault 隐藏冲突。
            for key in [(first, second), (second, first)]:
                relations.setdefault(key, set()).add(relation)
    frame = frame.copy()
    frame['n_relation'] = ''
    for index, row in frame[frame['gold'] == 'N'].iterrows():
        found = relations.get((normalize(row['text1']), normalize(row['text2'])), set())
        if len(found) == 1 and found <= {'SIMI', 'REL', 'OPPO'}:
            value = next(iter(found))
        elif found:
            value = '需核验:' + '/'.join(sorted(found))
        else:
            value = '未匹配'
        frame.loc[index, 'n_relation'] = value
    return frame


def score_methods(frame, output_dir):
    rows = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        folder = output_dir / method
        folder.mkdir(parents=True, exist_ok=True)
        prediction_file = folder / 'official_predictions.csv'
        save_csv(pd.DataFrame({
            'instance_id': frame['instance_id'], 'label': frame[column].map(TO_FULL),
        }), prediction_file)
        # SoftCons/HardCons 直接调用官方评分，N/N 不算官方反转兼容。
        report = score_files(GOLD_PATH, prediction_file, output_dir=folder)
        row = {
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
        }
        per_class = []
        for short, full in zip(LABELS, FULL_LABELS):
            metrics = report.classification.per_label[full]
            row[f'{short}_correct'] = metrics.true_positive
            per_class.append({
                'label': short, 'support': metrics.support,
                'correct': metrics.true_positive, 'precision': metrics.precision,
                'recall': metrics.recall, 'f1': metrics.f1,
            })
        rows.append(row)
        save_csv(pd.DataFrame(per_class), folder / 'per_class.csv')
        matrix = pd.crosstab(frame['gold'], frame[column]).reindex(
            index=LABELS, columns=LABELS, fill_value=0,
        )
        matrix.to_csv(folder / 'confusion_matrix.csv', encoding='utf-8-sig')
    summary = pd.DataFrame(rows)
    actual = summary.loc[summary['method'] == 'pair_aware', 'weighted_f1'].iloc[0]
    if not np.isclose(actual, EXPECTED_F1, atol=1e-10, rtol=0):
        raise ValueError(f'没有复现原版成绩：{actual}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, default=SCORES_PATH)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    frame = load_scores(args.scores)
    frame, pairs = pair_aware_decode(frame)
    verify_history(frame)
    frame = add_n_relations(frame)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = score_methods(frame, output_dir)

    for column in ['prediction', 'pair_prediction']:
        frame[f'{column}_error'] = [
            error_kind(gold, pred) for gold, pred in zip(frame['gold'], frame[column])
        ]
    counts = pd.DataFrame({
        method: frame[f'{column}_error'].value_counts()
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]
    }).fillna(0).astype(int).rename_axis('error_type').reset_index()
    errors = frame[frame['gold'] != frame['pair_prediction']].copy()
    pair_errors = pairs[~pairs['correct']].sort_values('gold_minus_chosen_joint_score')
    n_frame = frame[frame['gold'] == 'N'].copy()
    n_frame['correct'] = n_frame['gold'] == n_frame['pair_prediction']
    n_summary = n_frame.groupby('n_relation').agg(
        total=('instance_id', 'size'), correct=('correct', 'sum'),
    ).reset_index()
    n_summary['errors'] = n_summary['total'] - n_summary['correct']

    changed = frame['prediction'] != frame['pair_prediction']
    before = frame['prediction'] == frame['gold']
    after = frame['pair_prediction'] == frame['gold']
    changes = pd.DataFrame([{
        'changed': int(changed.sum()),
        'fixed': int((~before & after).sum()),
        'broken': int((before & ~after).sum()),
        'changed_wrong': int((changed & ~before & ~after).sum()),
    }])
    tables = {
        'predictions_with_scores': frame, 'summary': summary, 'error_counts': counts,
        'errors': errors, 'pairs': pairs, 'pair_errors': pair_errors,
        'n_errors': n_frame[~n_frame['correct']], 'n_subtypes': n_summary,
        'decoding_changes': changes,
    }
    for name, table in tables.items():
        save_csv(table, output_dir / f'{name}.csv')

    notes = [
        '原版 DeBERTa + Token Alignment + Real Pair CE：seed 42，epoch 7',
        '仅分析已导出的 Dev 分数；不训练、不加载模型。历史结构预测逐条一致：660/660。',
        'independent_max_probability 只属于独立预测，不能解释为结构预测置信度。',
        '配对错误请看 pair_errors：联合分数为 FB/BF/EE 两行 logits 之和。',
        'conditional_legal_probability 是合法组合内归一化值，未经校准。',
        '每个配对组只列一次，text1/text2 为 instance_id_0 的方向，两个组合保持相同行序。',
        'N-P=1 和 SoftCons=1 依赖当前数据结构和解码规则。',
        '本报告只对应已选中的 epoch 7，不能据此判断全部轮次或跨种子稳定性。',
    ]
    sections = [
        '\n'.join(notes),
        '\n===== 成绩 =====\n' + summary.to_string(index=False),
        '\n===== 错误类型（行数） =====\n' + counts.to_string(index=False),
        '\n===== 结构解码改动 =====\n' + changes.to_string(index=False),
        '\n===== 配对错误（每组一次） =====\n' + pair_errors.to_string(index=False),
        '\n===== N 子类 =====\n' + n_summary.to_string(index=False),
        '\n===== N 错误 =====\n' + tables['n_errors'][[
            'text1', 'text2', 'gold', 'pair_prediction', 'n_relation',
            'independent_max_probability',
        ]].to_string(index=False),
    ]
    text_report = '\n'.join(sections)
    (output_dir / 'report.txt').write_text(text_report, encoding='utf-8-sig')
    metadata = {
        'scores_path': str(args.scores.resolve()),
        'scores_sha256': hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        'reference_path': str(REFERENCE_PATH),
        'checkpoint': 'model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth',
        'historical_prediction_differences': 0,
        'original_export_batch_size': 16, 'max_length': 128, 'dtype': 'float32',
        'notes': notes,
    }
    (output_dir / 'analysis_config.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(text_report)
    print('\n报告目录：', output_dir)


if __name__ == '__main__':
    # Windows 控制台默认可能是 GBK，无法打印报告中的双向箭头。
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
