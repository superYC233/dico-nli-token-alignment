"""全面错误分析：扫描一个 run 的所有 epoch，选出最佳轮次，逐项拆解错误。

只读，不训练、不改任何模型。运行时自动定位 0.9306 那一版。

用法：
    python analyze_errors.py
    python analyze_errors.py run01        # 指定 run
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = SCRIPT_DIR / 'outputs'
ANALYSIS_ROOT = SCRIPT_DIR / 'analysis'

LABELS = ['F', 'B', 'E', 'N']
REV = {'F': 'B', 'B': 'F', 'E': 'E', 'N': 'N'}
NAME_TO_SHORT = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}

PHRASIS_FILES = [
    PROJECT_ROOT / 'data/phrasis_sources/PhrasIS.train.all.txt',
    PROJECT_ROOT / 'data/phrasis_sources/PhrasIS.test.all.txt',
]
BASELINE_PRED = PROJECT_ROOT / 'artifacts/final/dev_predictions_pair_aware.csv'


# ---------------------------------------------------------------- 基础工具

def norm(text):
    return ' '.join(str(text).casefold().split())


def load_epoch(csv_path):
    df = pd.read_csv(csv_path)
    df['gold'] = df['gold_label'].map(NAME_TO_SHORT)
    if df['gold'].isna().any():
        raise ValueError(f'{csv_path} 存在未知标签')
    for label in LABELS:
        if f'logit_{label}' not in df.columns:
            raise ValueError(f'{csv_path} 缺少 logit_{label}')
    return df.reset_index(drop=True)


def pair_aware_from_logits(df):
    """按 pair_id 在 FB/BF/EE 里联合选一组，singleton 不动。"""
    pred = df['prediction'].to_numpy().copy()
    logits = df[[f'logit_{l}' for l in LABELS]].to_numpy()

    combos = [('F', 'B'), ('B', 'F'), ('E', 'E')]
    for _, group in df.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        a, b = group.index[0], group.index[1]
        scores = [
            logits[a][0] + logits[b][1],   # FB
            logits[a][1] + logits[b][0],   # BF
            logits[a][2] + logits[b][2],   # EE
        ]
        pa, pb = combos[int(np.argmax(scores))]
        pred[a], pred[b] = pa, pb
    return pred


def hard_cons(df, pred):
    """成对行两行都预测正确。"""
    correct = pred == df['gold'].to_numpy()
    kept = []
    for _, group in df.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        kept.append(bool(correct[group.index[0]] and correct[group.index[1]]))
    return float(np.mean(kept)) if kept else float('nan')


def soft_cons(df, pred):
    """两行预测是否满足标签反转关系。"""
    kept = []
    for _, group in df.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        pa, pb = pred[group.index[0]], pred[group.index[1]]
        kept.append(REV[pa] == pb)
    return float(np.mean(kept)) if kept else float('nan')


def summarize(df, pred):
    gold = df['gold'].to_numpy()
    out = {
        'w_f1': f1_score(gold, pred, average='weighted'),
        'acc': float((gold == pred).mean()),
        'hard_cons': hard_cons(df, pred),
        'soft_cons': soft_cons(df, pred),
        'correct': int((gold == pred).sum()),
    }
    for label in LABELS:
        mask = gold == label
        out[f'{label}_correct'] = int((pred[mask] == label).sum())
        out[f'{label}_n'] = int(mask.sum())
    out['N_correct'] = out['N_correct']
    out['FBE_correct'] = out['F_correct'] + out['B_correct'] + out['E_correct']
    return out


def confusion(df, pred):
    gold = df['gold'].to_numpy()
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for g, p in zip(gold, pred):
        matrix.loc[g, p] += 1
    matrix.index.name = '真实 \\ 预测'
    return matrix


def error_kind(gold, pred):
    if gold == pred:
        return '正确'
    if gold in ('F', 'B', 'E') and pred == 'N':
        return 'F/B/E → N'
    if gold == 'N' and pred in ('F', 'B', 'E'):
        return 'N → 非N'
    if gold == 'E' and pred in ('F', 'B'):
        return 'E → F/B'
    if gold in ('F', 'B') and pred == 'E':
        return 'F/B → E'
    if gold in ('F', 'B') and pred in ('F', 'B'):
        return 'F ↔ B'
    return '其他'


# ---------------------------------------------------------------- N 子类

def load_phrasis_relations():
    """(text1, text2) -> 原始关系（SIMI / REL / OPPO / UNR ...）。"""
    relations = {}
    if not PHRASIS_FILES[0].parent.exists():
        return relations
    for path in PHRASIS_FILES:
        if not path.is_file():
            continue
        with path.open(encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                relation, text1, text2 = parts[1].strip(), parts[2], parts[3]
                key = (norm(text1), norm(text2))
                key_rev = (key[1], key[0])
                relations.setdefault(key, relation)
                relations.setdefault(key_rev, relation)
    return relations


def attach_relation(df, relations):
    df = df.copy()
    df['phrasis_relation'] = [
        relations.get((norm(a), norm(b)), '未匹配')
        for a, b in zip(df['text1'], df['text2'])
    ]
    return df


# ---------------------------------------------------------------- 主流程

def collect_runs():
    """扫描 outputs/*/*/epoch_*/predictions_with_scores.csv。"""
    runs = {}
    for csv_path in sorted(OUTPUT_ROOT.glob('*/*/epoch_*/predictions_with_scores.csv')):
        run = csv_path.parents[2].name
        epoch = int(csv_path.parent.name.split('_')[1])
        runs.setdefault(run, []).append((epoch, csv_path))
    for run in runs:
        runs[run].sort()
    return runs


def scan_curve(run, entries):
    rows = []
    for epoch, path in entries:
        df = load_epoch(path)
        pred_pa = pair_aware_from_logits(df)
        ind = summarize(df, df['prediction'].to_numpy())
        pa = summarize(df, pred_pa)
        rows.append({
            'run': run, 'epoch': epoch,
            'ind_w_f1': ind['w_f1'], 'ind_N': ind['N_correct'], 'ind_FBE': ind['FBE_correct'],
            'pa_w_f1': pa['w_f1'], 'pa_N': pa['N_correct'], 'pa_FBE': pa['FBE_correct'],
            'pa_hard': pa['hard_cons'], 'pa_correct': pa['correct'],
        })
    return pd.DataFrame(rows)


def analyze(run, best_epoch, path, relations):
    df = load_epoch(path)
    pred_ind = df['prediction'].to_numpy()
    pred_pa = pair_aware_from_logits(df)

    # 与文件里保存的 pair_prediction 对照，确认规则一致
    if 'pair_prediction' in df.columns:
        diff = int((df['pair_prediction'].to_numpy() != pred_pa).sum())
        if diff:
            print(f'[警告] 重算的 pair-aware 与文件里的 pair_prediction 有 {diff} 行不同')

    print('\n' + '=' * 78)
    print(f'分析：{run}  第 {best_epoch} 轮   ({path.parent.name})')
    print('=' * 78)

    for name, pred in [('独立预测', pred_ind), ('结构解码', pred_pa)]:
        metric = summarize(df, pred)
        print(f'\n----- {name} -----')
        print(f"Weighted F1 {metric['w_f1']:.6f} | 准确率 {metric['acc']:.6f} | "
              f"HardCons {metric['hard_cons']:.6f} | SoftCons {metric['soft_cons']:.6f}")
        print(f"总正确 {metric['correct']}/660 | "
              f"N {metric['N_correct']}/{metric['N_n']} | "
              f"F/B/E {metric['FBE_correct']}/554")
        per_class = []
        for label in LABELS:
            gold = df['gold'].to_numpy()
            mask = gold == label
            tp = int(((pred == label) & mask).sum())
            fp = int(((pred == label) & ~mask).sum())
            fn = int(((pred != label) & mask).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class.append({
                'label': label, 'n': int(mask.sum()), 'correct': tp,
                'precision': precision, 'recall': recall, 'f1': f1,
                '错误率': 1 - recall,
            })
        print(pd.DataFrame(per_class).round(4).to_string(index=False))
        print('\n混淆矩阵：')
        print(confusion(df, pred).to_string())

    # ---- 错误分类
    gold = df['gold'].to_numpy()
    print('\n----- 错误类型分布 -----')
    table = pd.DataFrame({
        'gold': gold, '独立': pred_ind, '结构': pred_pa,
    })
    table['独立类型'] = [error_kind(g, p) for g, p in zip(gold, pred_ind)]
    table['结构类型'] = [error_kind(g, p) for g, p in zip(gold, pred_pa)]
    kinds = ['F/B/E → N', 'N → 非N', 'E → F/B', 'F/B → E', 'F ↔ B', '其他']
    cross = pd.DataFrame({
        '独立': table['独立类型'].value_counts().reindex(kinds, fill_value=0),
        '结构': table['结构类型'].value_counts().reindex(kinds, fill_value=0),
    })
    print(cross.to_string())

    # ---- 逐条错误明细
    logits = df[[f'logit_{l}' for l in LABELS]].to_numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    detail = df[['instance_id', 'pair_id', 'text1', 'text2', 'gold']].copy()
    detail['独立预测'] = pred_ind
    detail['结构预测'] = pred_pa
    detail['独立错误'] = pred_ind != gold
    detail['结构错误'] = pred_pa != gold
    detail['gold_logit'] = [logits[i][LABELS.index(g)] for i, g in enumerate(gold)]
    detail['pred_logit'] = [logits[i][LABELS.index(p)] for i, p in enumerate(pred_ind)]
    detail['margin'] = detail['gold_logit'] - detail['pred_logit']
    detail['置信度'] = probs.max(axis=1)
    detail['错误类型'] = table['结构类型']
    detail = attach_relation(detail, relations)

    wrong = detail[detail['结构错误']].sort_values('margin')
    print(f'\n----- 结构解码后的错误明细（{len(wrong)} 条，按 margin 从大到小）-----')
    show = wrong[['pair_id', 'text1', 'text2', 'gold', '结构预测',
                  'gold_logit', 'pred_logit', 'margin', '置信度', 'phrasis_relation']]
    with pd.option_context('display.width', 200, 'display.max_colwidth', 40):
        print(show.round(3).to_string(index=False))

    # ---- N 子类
    print('\n----- N 子类（按 PhrasIS 原始关系）-----')
    n_rows = detail[detail['gold'] == 'N'].copy()
    matched = (n_rows['phrasis_relation'] != '未匹配').sum()
    print(f'匹配率：{matched}/{len(n_rows)}')
    for name, pred in [('独立', pred_ind), ('结构', pred_pa)]:
        n_rows[name] = pred[detail['gold'].to_numpy() == 'N']
    n_rows['独立错误N'] = n_rows['独立'] != 'N'
    n_rows['结构错误N'] = n_rows['结构'] != 'N'
    group = n_rows.groupby('phrasis_relation').agg(
        总数=('pair_id', 'size'),
        独立错误=('独立错误N', 'sum'),
        结构错误=('结构错误N', 'sum'),
    )
    group['独立错误率'] = (group['独立错误'] / group['总数']).round(3)
    group['结构错误率'] = (group['结构错误'] / group['总数']).round(3)
    print(group.to_string())
    print('\nN 错误的方向：')
    print(pd.crosstab(n_rows.loc[n_rows['结构错误N'], 'phrasis_relation'],
                      n_rows.loc[n_rows['结构错误N'], '结构']).to_string())

    # ---- 成对分析
    print('\n----- 成对分析（277 组）-----')
    pairs = []
    for pair_id, g in detail.groupby('pair_id'):
        if len(g) != 2:
            continue
        ok = (~g['结构错误']).sum()
        pairs.append({'pair_id': pair_id, '对的行数': int(ok),
                      '金标': '/'.join(sorted(g['gold'])),
                      '预测': '/'.join(g['结构预测'])})
    pairs = pd.DataFrame(pairs)
    print(pairs['对的行数'].value_counts().sort_index().to_string())
    print('\n错误对的组合：')
    bad = pairs[pairs['对的行数'] < 2]
    print(bad.groupby(['金标', '预测']).size().sort_values(ascending=False).head(12).to_string())

    # ---- 词面重合度
    print('\n----- 词面重合度（Jaccard）与错误率 -----')
    token_a = [set(norm(t).split()) for t in detail['text1']]
    token_b = [set(norm(t).split()) for t in detail['text2']]
    jac = [len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in zip(token_a, token_b)]
    detail['jaccard'] = jac
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    detail['重合度区间'] = pd.cut(detail['jaccard'], bins=bins, right=False)
    overlap = detail.groupby('重合度区间', observed=True).agg(
        样本数=('pair_id', 'size'),
        独立错误=('独立错误', 'sum'),
        结构错误=('结构错误', 'sum'),
    )
    overlap['独立错误率'] = (overlap['独立错误'] / overlap['样本数']).round(3)
    overlap['结构错误率'] = (overlap['结构错误'] / overlap['样本数']).round(3)
    print(overlap.to_string())

    # ---- 与原版 DeBERTa 的差异
    if BASELINE_PRED.is_file():
        base = pd.read_csv(BASELINE_PRED).set_index('instance_id')['label']
        base = base.reindex(detail['instance_id']).map(NAME_TO_SHORT).to_numpy()
        fixed = int(((base != gold) & (pred_pa == gold)).sum())
        broken = int(((base == gold) & (pred_pa != gold)).sum())
        both_wrong = int(((base != gold) & (pred_pa != gold)).sum())
        print(f'\n----- 与原版 DeBERTa 对比 -----')
        print(f'修好 {fixed} 条 | 破坏 {broken} 条 | 两者都错 {both_wrong} 条')

        target = (base != gold) & (pred_pa == gold)
        print('\n修好的样本：')
        print(detail.loc[target, ['text1', 'text2', 'gold', '结构预测']]
              .head(20).to_string(index=False))
        target = (base == gold) & (pred_pa != gold)
        print('\n破坏的样本：')
        print(detail.loc[target, ['text1', 'text2', 'gold', '结构预测']]
              .head(20).to_string(index=False))
    else:
        print('\n[跳过] 找不到原版预测文件：', BASELINE_PRED)

    # ---- 导出
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = ANALYSIS_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / f'errors_epoch_{best_epoch}.csv', index=False, encoding='utf-8-sig')
    pairs.to_csv(out_dir / f'pairs_epoch_{best_epoch}.csv', index=False, encoding='utf-8-sig')
    print(f'\n明细已保存：{out_dir}')


def main():
    target_run = sys.argv[1] if len(sys.argv) > 1 else None
    runs = collect_runs()
    if not runs:
        raise SystemExit(f'没找到任何预测文件：{OUTPUT_ROOT}')

    curves = {run: scan_curve(run, entries) for run, entries in runs.items()}

    print('=' * 78)
    print('全部运行的逐轮曲线（pair_aware）')
    print('=' * 78)
    for run, curve in curves.items():
        print(f'\n--- {run} ---')
        print(curve[['epoch', 'ind_w_f1', 'pa_w_f1', 'pa_N', 'pa_FBE',
                     'pa_hard']].round(4).to_string(index=False))
        spread = curve['pa_w_f1'].max() - curve['pa_w_f1'].min()
        print(f"最好 {curve['pa_w_f1'].max():.6f} | 最差 {curve['pa_w_f1'].min():.6f} | "
              f"均值 {curve['pa_w_f1'].mean():.6f} | 跨度 {spread:.4f}")

    if target_run is None:
        all_best = [(c['pa_w_f1'].max(), run) for run, c in curves.items()]
        all_best.sort(reverse=True)
        target_run = all_best[0][1]
        print(f'\n[自动] 全局最佳来自 {target_run}（{all_best[0][0]:.6f}）')

    if target_run not in curves:
        raise SystemExit(f'没有这个 run：{target_run}；可选 {list(curves)}')

    curve = curves[target_run]
    row = curve.loc[curve['pa_w_f1'].idxmax()]
    best_epoch = int(row['epoch'])
    path = dict(runs[target_run])[best_epoch]

    relations = load_phrasis_relations()
    print(f'\nPhrasIS 关系表：{len(relations)} 个键')

    analyze(target_run, best_epoch, path, relations)


if __name__ == '__main__':
    main()
