"""N 类改序消融 A / B / C。

三组共用同一套训练逻辑，只有训练集构造不同：

    A  base   原始 Train                                3042 条
    B  dup    原始 Train ＋ 同一批 474 条 N 的原样副本   3516 条
    C  swap   原始 Train ＋ 同一批 474 条 N 的反向版本   3516 条

主对照是 C − B：两组样本量、N 类别比例、训练步数完全一致，
差别只在“新增的是原样副本，还是交换后的版本”。

判据（预先固定，避免事后挑指标）：
    主  逐条反转兼容率：Dev 106 条 N 中，双向都预测 N 的比例
        当前 run01 = 55 / 106
    辅  Weighted F1 / SoftCons / HardCons、N 的 P/R/F1
    辅  不一致对 b（原向 N、反向非 N）、c（原向非 N、反向 N）

注意：
    * set_seed 必须在 get_peft_model 之前，否则 LoRA 初始化不可复现
    * 每个 epoch 都保留 checkpoint，不按 eval_loss 挑，避免选模与处理混淆
    * 反向副本使用新的 pair_id，避免以后做 pair-aware 时被当成可逆组

用法（在 experiments/qwen_lora 下执行）：
    python train_n_ablation.py --modes A B C
    python train_n_ablation.py --modes B C --eval-last-only
"""

from __future__ import annotations

# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')


import argparse
import json
import re
import sys
from pathlib import Path

# baseline.py 会 import evaluation_functions，那是个包级导入，
# 需要项目根目录在 sys.path 里。PyCharm 会自动加，命令行不会。
SCRIPT_DIR = QWEN_DIR
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from baseline import evaluate, score_predictions
from data_utils import get_datasets, load_data, tokenizer
from diagnose_pairs import diagnose_n_reversal


# --------------------------------------------------------------- 路径与常量

BASE_MODEL = 'D:/LLM/model/Qwen3-0.6B-Base'
ADAPTER_ROOT = Path('D:/LLM/train_model/DiCo-NLI/n_ablation')
OUTPUT_ROOT = SCRIPT_DIR / 'outputs' / 'n_ablation'

TRAIN_CSV = (
    PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
)
DEV_CSV = (
    PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
)

SEED = 42
EPOCHS = 5
LABEL_N = 'NEGATIVE_OTHER'

# 交换语序后的标签：F↔B，E 和 N 保持不变
REVERSE_LABEL = {
    'FORWARD_ENTAILMENT': 'BACKWARD_ENTAILMENT',
    'BACKWARD_ENTAILMENT': 'FORWARD_ENTAILMENT',
    'EQUIVALENCE': 'EQUIVALENCE',
    LABEL_N: LABEL_N,
}

LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules='all-linear',
    bias='none',
    task_type='CAUSAL_LM',
)

# 三组必须完全一致的部分
COMMON_SFT = dict(
    learning_rate=1e-4,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=32,
    bf16=True,
    completion_only_loss=True,
    packing=False,
    seed=SEED,
    data_seed=SEED,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=None,          # 5 个 checkpoint 全部保留
    load_best_model_at_end=False,   # 不按 eval_loss 选模
    logging_steps=20,
    report_to='none',
)


# ----------------------------------------------------------- 增强来源的筛选

def norm(text: object) -> str:
    """去掉大小写、空白和标点，用于精确重合检查。"""
    return re.sub(r'[^0-9a-z]+', '', str(text).casefold())


def pair_key(text1: object, text2: object) -> tuple[str, str]:
    return norm(text1), norm(text2)


def build_swap_sources() -> tuple[pd.DataFrame, dict]:
    """从 Train 的 478 条 N 中筛出可安全反向的来源。

    排除四类来源，其余作为 B / C 两组共同的新增来源：
        1. Train 中已存在反向行（互为反向的两条，都不再反向）
        2. 反向后的文本对在 Train 中已存在，且标签不是 N
        3. 来源本身或其反向与 Dev 重合
        4. 与已保留的反向文本对重复
    """
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    dev = pd.read_csv(DEV_CSV, keep_default_na=False)

    # 用“键 -> 标签集合”，因为同一个有序文本对可能对应多个标签。
    # 当前 Train 里确实存在这种冲突，用单个标签的字典会把非 N 行误判成 N 行。
    train_label_sets: dict[tuple[str, str], set[str]] = {}
    for text1, text2, label in zip(train['text1'], train['text2'], train['label']):
        train_label_sets.setdefault(pair_key(text1, text2), set()).add(label)

    dev_keys: set[tuple[str, str]] = set()
    for text1, text2 in zip(dev['text1'], dev['text2']):
        dev_keys.add(pair_key(text1, text2))
        dev_keys.add(pair_key(text2, text1))

    audit = {
        '已有反向行': 0,
        '反向与其他标签冲突': 0,
        '与 Dev 重合': 0,
        '反向文本对重复': 0,
    }

    kept: list[dict] = []
    seen_swap: set[tuple[str, str]] = set()

    # 只在真正的 N 行上迭代；不要用标签字典反查，否则标签冲突的行会被算进来
    n_rows = train[train['label'] == LABEL_N]

    for text1, text2, pair_id, instance_id in zip(
        n_rows['text1'], n_rows['text2'], n_rows['pair_id'], n_rows['instance_id']
    ):
        src = pair_key(text1, text2)
        rev = pair_key(text2, text1)

        if src in dev_keys or rev in dev_keys:
            audit['与 Dev 重合'] += 1
            continue

        reverse_labels = train_label_sets.get(rev)
        if reverse_labels:
            if LABEL_N in reverse_labels:
                audit['已有反向行'] += 1
            else:
                audit['反向与其他标签冲突'] += 1
            continue

        if rev in seen_swap:
            audit['反向文本对重复'] += 1
            continue

        seen_swap.add(rev)
        kept.append({
            'instance_id': instance_id,
            'pair_id': pair_id,
            'text1': text1,
            'text2': text2,
        })

    sources = pd.DataFrame(kept)
    audit['原始 N 行数'] = int((train['label'] == LABEL_N).sum())
    audit['保留来源数'] = len(sources)
    audit['合计排除'] = sum(
        audit[key] for key in
        ['已有反向行', '反向与其他标签冲突', '与 Dev 重合', '反向文本对重复']
    )
    return sources, audit

def build_full_augmented_rows(train_df, mode, safe_n_ids):
    """构造"全体新增"的第二份数据（D / E 两臂共用）。

    D：每一行原样复制              → 与 A 合并后 6084 条
    E：每一行交换语序、标签反转     → 与 A 合并后 6084 条
       其中 N 里反向不安全的行（不在 safe_n_ids 中的）改为原样复制，
       这样两臂行数、类别比例完全一致，唯一差别是 N 第二份的方向。
    """
    suffix = '__dup' if mode == 'D' else '__rev'

    out = {
        'instance_id': [], 'pair_id': [],
        'text1_lang': [], 'text2_lang': [],
        'text1': [], 'text2': [], 'label': [],
    }
    swapped = 0
    copied = 0
    copied_ids = []

    for row in train_df.itertuples(index=False):
        if mode == 'D':
            text1, text2, label = row.text1, row.text2, row.label
        else:
            # F/B/E 一律反向；N 只对安全的 472 条反向
            need_swap = (row.label != LABEL_N) or (row.instance_id in safe_n_ids)
            if need_swap:
                text1 = row.text2
                text2 = row.text1
                label = REVERSE_LABEL[row.label]
                swapped += 1
            else:
                text1, text2, label = row.text1, row.text2, row.label
                copied += 1
                copied_ids.append(row.instance_id)

        out['instance_id'].append(f'{row.instance_id}{suffix}')
        out['pair_id'].append(f'{row.pair_id}{suffix}')
        out['text1_lang'].append('en')
        out['text2_lang'].append('en')
        out['text1'].append(text1)
        out['text2'].append(text2)
        out['label'].append(label)

    audit = {'augmented_rows': len(out['instance_id'])}
    if mode == 'E':
        audit['swapped'] = swapped
        audit['copied_instead_of_swap'] = copied
        audit['copied_instance_ids'] = copied_ids

    return pd.DataFrame(out), audit
# --------------------------------------------------------------- 训练集构造

def build_train_dataset(train_ds, sources: pd.DataFrame, mode: str):
    """返回 (训练集, 说明, 审计信息)。"""
    if mode == 'A':
        return train_ds, '原始 Train', {}

    # ---- D / E：全体新增一份 ----
    if mode in ('D', 'E'):
        safe_n_ids = set(sources['instance_id'])
        extra_df, audit = build_full_augmented_rows(
            train_ds.to_pandas(), mode, safe_n_ids
        )
        extra = Dataset.from_pandas(extra_df, preserve_index=False)
        extra = extra.cast(train_ds.features)
        note = (
            '全体原样复制' if mode == 'D'
            else '全体反向（N 不安全的用复制代替）'
        )
        return concatenate_datasets([train_ds, extra]), note, audit

    # ---- B / C：只针对 472 条 N 来源 ----
    source_ids = set(sources['instance_id'])
    picked = train_ds.filter(lambda row: row['instance_id'] in source_ids)

    if len(picked) != len(source_ids):
        raise ValueError(f'来源未全部匹配：{len(picked)} / {len(source_ids)}')

    if mode == 'B':
        extra = picked
        note = '同一批 N 的原样副本'
    elif mode == 'C':
        swapped = pd.DataFrame({
            'instance_id': [f'{v}__swap' for v in picked['instance_id']],
            'pair_id': [f'{v}__swap' for v in picked['pair_id']],
            'text1_lang': 'en',
            'text2_lang': 'en',
            'text1': picked['text2'],
            'text2': picked['text1'],
            'label': LABEL_N,
        })
        extra = Dataset.from_pandas(swapped, preserve_index=False)
        extra = extra.cast(train_ds.features)
        note = '同一批 N 的反向版本'
    else:
        raise ValueError(f'未知模式：{mode}')

    return concatenate_datasets([train_ds, extra]), note, {}

# --------------------------------------------------------------- 单组训练

def train_one(mode: str) -> Path:
    set_seed(SEED)

    dataset = load_data()
    sources, _ = build_swap_sources()
    train_ds, note, build_audit = build_train_dataset(
        dataset['train'], sources, mode
    )

    print(f'\n===== 模式 {mode} =====')
    print('训练集构造：', note)
    print('训练集行数：', len(train_ds))
    if build_audit:
        print('构造审计：', build_audit)

    dataset['train'] = train_ds
    sft_dataset = get_datasets(dataset)

    run_dir = ADAPTER_ROOT / f'run_{mode}'
    run_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype='auto')

    # LoRA 的 A 矩阵是随机初始化的，必须在 get_peft_model 之前定种子
    set_seed(SEED)
    lora_model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

    trainer = SFTTrainer(
        model=lora_model,
        args=SFTConfig(output_dir=str(run_dir), **COMMON_SFT),
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(run_dir / 'final'))

    print(f'模式 {mode} 训练完成，Adapter 保存于：{run_dir}')
    return run_dir


# --------------------------------------------------------------- 评估

def list_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = []
    for path in run_dir.glob('checkpoint-*'):
        try:
            step = int(path.name.split('-')[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))

    checkpoints.sort(key=lambda item: item[0])
    return [path for _, path in checkpoints]


def describe_n_reversal(diagnosis: pd.DataFrame) -> dict:
    """把逐条双向诊断压成对比用的几个数。"""
    valid = diagnosis[diagnosis['valid_pair']]

    forward_n = valid['forward_correct']
    reverse_n = valid['reverse_correct']

    return {
        'valid_pairs': int(len(valid)),
        'both_n': int((forward_n & reverse_n).sum()),
        'only_forward_n': int((forward_n & ~reverse_n).sum()),
        'only_reverse_n': int((~forward_n & reverse_n).sum()),
        'neither_n': int((~forward_n & ~reverse_n).sum()),
        'both_n_rate': float((forward_n & reverse_n).mean()) if len(valid) else 0.0,
        'reversal_consistent_rate': float(
            diagnosis['reversal_consistent'].mean()
        ) if len(diagnosis) else 0.0,
    }


def evaluate_one(mode: str, adapter_path: Path, tag: str) -> dict:
    out_dir = OUTPUT_ROOT / f'run_{mode}' / 'eval' / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype='auto')
    model = PeftModel.from_pretrained(base, str(adapter_path)).to('cuda')
    model.eval()

    records = evaluate(model, batch_size=16, split='dev')
    report = score_predictions(records, str(out_dir))

    result = {'mode': mode, 'checkpoint': tag, 'adapter': str(adapter_path)}
    if report is not None:
        neg = report.classification.per_label['NEGATIVE_OTHER']
        result.update({
            'weighted_f1': report.weighted_f1,
            'soft_cons': report.soft_cons,
            'hard_cons': report.hard_cons,
            'n_precision': neg.precision,
            'n_recall': neg.recall,
            'n_f1': neg.f1,
        })

    diagnosis = diagnose_n_reversal(
        model,
        batch_size=32,
        output_path=out_dir / 'n_reversal_diagnosis.csv',
    )
    result.update(describe_n_reversal(diagnosis))

    del model
    del base
    torch.cuda.empty_cache()

    return result


def evaluate_all(modes: list[str], last_only: bool) -> pd.DataFrame:
    rows = []

    for mode in modes:
        run_dir = ADAPTER_ROOT / f'run_{mode}'
        if not run_dir.is_dir():
            print(f'跳过模式 {mode}：找不到 {run_dir}')
            continue

        targets = [('final', run_dir / 'final')]
        if not last_only:
            # 三组每轮的步数不同（A 是 191，B/C 是 220），
            # 用 epoch 序号对齐，不要用 checkpoint 名字
            targets = [
                (f'epoch_{index + 1}', path)
                for index, path in enumerate(list_checkpoints(run_dir))
            ] + targets

        for tag, path in targets:
            if not path.is_dir():
                continue
            print(f'\n----- 评估 {mode} / {tag} -----')
            rows.append(evaluate_one(mode, path, tag))

    summary = pd.DataFrame(rows)
    return summary


# --------------------------------------------------------------- 入口

def main() -> None:
    parser = argparse.ArgumentParser(description='N 类改序消融 A/B/C')
    parser.add_argument('--modes', nargs='+', default=['A', 'B', 'C'])
    parser.add_argument('--train-only', action='store_true', help='只训练，不评估')
    parser.add_argument('--eval-only', action='store_true', help='只评估已有 Adapter')
    parser.add_argument(
        '--eval-last-only',
        action='store_true',
        help='只评估 final，不评估每个 epoch 的 checkpoint',
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sources, audit = build_swap_sources()
    sources.to_csv(
        OUTPUT_ROOT / 'swap_sources.csv', index=False, encoding='utf-8-sig'
    )
    (OUTPUT_ROOT / 'selection_audit.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print('\n===== 新增来源筛选 =====')
    for key, value in audit.items():
        print(f'{key}：{value}')
    if audit['保留来源数'] != 474:
        print(
            f"\n注意：保留 {audit['保留来源数']} 条，与预期的 474 条不同，"
            '请对照 selection_audit.json 核对排除规则。'
        )

    if not args.eval_only:
        for mode in args.modes:
            train_one(mode)

    if args.train_only:
        return

    summary = evaluate_all(args.modes, args.eval_last_only)
    if summary.empty:
        print('没有可汇总的评估结果')
        return

    # 允许分多次跑（例如先 B C 再补 A），合并已有汇总而不是覆盖
    summary_path = OUTPUT_ROOT / 'ablation_summary.csv'
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        summary = pd.concat([previous, summary], ignore_index=True)
        summary = summary.drop_duplicates(
            subset=['mode', 'checkpoint'], keep='last'
        )
        summary = summary.sort_values(['mode', 'checkpoint']).reset_index(drop=True)

    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    (OUTPUT_ROOT / 'ablation_summary.json').write_text(
        json.dumps(summary.to_dict(orient='records'), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    columns = [
        'mode', 'checkpoint', 'weighted_f1', 'n_f1', 'n_precision', 'n_recall',
        'both_n', 'only_forward_n', 'only_reverse_n', 'neither_n',
        'both_n_rate', 'soft_cons', 'hard_cons',
    ]
    print('\n===== 汇总 =====')
    print(summary[[c for c in columns if c in summary.columns]].to_string(index=False))
    print('\n结果目录：', OUTPUT_ROOT)


if __name__ == '__main__':
    main()
