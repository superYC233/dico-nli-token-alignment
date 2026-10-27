
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path
import gc
import os
import sys

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


SCRIPT_DIR = QWEN_DIR
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 兼容现有代码的导入和数据相对路径
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(SCRIPT_DIR)

from compare_decoding import evaluate_with_scores
from analyze_decoding import pair_aware_decode, score_one_method


MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'

TRAIN_DIR = Path(
    'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_valid/run01'
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / 'outputs/qwen3_0p6b_valid/run01/epoch_comparison'
)

# 每轮对应的 checkpoint
CHECKPOINTS = [
    (1, 'checkpoint-220'),
    (2, 'checkpoint-440'),
    (3, 'checkpoint-660'),
    (4, 'checkpoint-880'),
    (5, 'checkpoint-1100'),
]


def main():
    # 开始前检查，避免跑到一半才发现缺少文件
    for epoch, checkpoint in CHECKPOINTS:
        adapter_path = TRAIN_DIR / checkpoint
        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(
                f'第 {epoch} 轮 Adapter 不存在：{adapter_path}'
            )

    # 避免覆盖之前的评估结果
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f'结果目录已有内容，请修改 OUTPUT_DIR：{OUTPUT_DIR}'
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for epoch, checkpoint in CHECKPOINTS:
        print(f'\n===== 评估第 {epoch} 轮 =====')

        # 每轮重新加载 Base，再挂载对应 Adapter
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype='auto',
            local_files_only=True,
        )

        model = PeftModel.from_pretrained(
            model,
            str(TRAIN_DIR / checkpoint),
            local_files_only=True,
        ).to('cuda')

        model.eval()

        # 固定 batch=16，与之前正式评估保持一致
        records = evaluate_with_scores(
            model,
            batch_size=16,
            split='dev',
        )

        # 同一份分数，同时用于独立预测和结构解码
        df = pair_aware_decode(pd.DataFrame(records))

        epoch_dir = OUTPUT_DIR / f'epoch_{epoch}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            epoch_dir / 'predictions_with_scores.csv',
            index=False,
            encoding='utf-8-sig',
        )

        for method, column in [
            ('independent', 'prediction'),
            ('pair_aware', 'pair_prediction'),
        ]:
            report = score_one_method(
                df,
                column,
                epoch_dir / method,
            )

            n = report.classification.per_label[
                'NEGATIVE_OTHER'
            ]

            summary_rows.append({
                'epoch': epoch,
                'method': method,
                'weighted_f1': report.weighted_f1,
                'soft_cons': report.soft_cons,
                'hard_cons': report.hard_cons,
                'N_precision': n.precision,
                'N_recall': n.recall,
                'N_f1': n.f1,
            })

            print(
                f'{method}: '
                f'W-F1={report.weighted_f1:.6f}, '
                f'N-F1={n.f1:.6f}'
            )

        # 每完成一轮就保存汇总，防止中断后丢失进度
        pd.DataFrame(summary_rows).to_csv(
            OUTPUT_DIR / 'epoch_summary.csv',
            index=False,
            encoding='utf-8-sig',
        )

        # 释放当前模型，避免下一轮加载时显存叠加
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)

    print('\n===== 各轮结果 =====')
    print(summary.round(6).to_string(index=False))

    print('\n===== 按 Dev Weighted F1 选择最佳轮次 =====')
    for method in ['independent', 'pair_aware']:
        group = summary[summary['method'] == method]
        best = group.loc[group['weighted_f1'].idxmax()]

        print(
            f"{method}：第 {int(best['epoch'])} 轮，"
            f"W-F1={best['weighted_f1']:.6f}，"
            f"N-F1={best['N_f1']:.6f}"
        )

    print('\n汇总文件：', OUTPUT_DIR / 'epoch_summary.csv')


if __name__ == '__main__':
    main()