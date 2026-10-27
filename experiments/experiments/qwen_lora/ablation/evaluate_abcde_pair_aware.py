
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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_decoding import evaluate_with_scores
from analyze_decoding import pair_aware_decode, score_one_method


MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'

ADAPTER_ROOT = Path(
    'D:/LLM/train_model/DiCo-NLI/n_ablation'
)

RUN_NAME = 'run01'

OUTPUT_ROOT = (
    SCRIPT_DIR / 'outputs/n_ablation_pair_aware' / RUN_NAME
)

MODES = ['A', 'B', 'C', 'D', 'E']


def main():
    # 兼容现有 load_data() 的相对路径
    os.chdir(SCRIPT_DIR)

    # 检查已有 Adapter
    for mode in MODES:
        adapter_path = ADAPTER_ROOT / f'run_{mode}' / 'final'

        if not (adapter_path / 'adapter_config.json').exists():
            raise FileNotFoundError(
                f'找不到 {mode} 的 Adapter：{adapter_path}'
            )

    # 避免覆盖之前的结果
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f'结果目录已有内容，请修改 RUN_NAME：{OUTPUT_ROOT}'
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for mode in MODES:
        print(f'\n===== 评估模型 {mode} =====')

        adapter_path = ADAPTER_ROOT / f'run_{mode}' / 'final'
        output_dir = OUTPUT_ROOT / f'run_{mode}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载 Base，再挂载当前组的 Adapter
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype='auto',
            local_files_only=True,
        )

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=True,
        ).to('cuda')

        model.eval()

        # 固定 batch=16，与之前正式评估一致
        records = evaluate_with_scores(
            model,
            batch_size=16,
            split='dev',
        )

        # 保留原 prediction，后处理结果写入 pair_prediction
        df = pair_aware_decode(pd.DataFrame(records))

        df.to_csv(
            output_dir / 'predictions_with_scores.csv',
            index=False,
            encoding='utf-8-sig',
        )

        # 同一模型，同时评估独立预测和后处理结果
        for method, column in [
            ('independent', 'prediction'),
            ('pair_aware', 'pair_prediction'),
        ]:
            report = score_one_method(
                df,
                column,
                output_dir / method,
            )

            n = report.classification.per_label['NEGATIVE_OTHER']

            summary_rows.append({
                'model': mode,
                'method': method,
                'weighted_f1': report.weighted_f1,
                'soft_cons': report.soft_cons,
                'hard_cons': report.hard_cons,
                'N_precision': n.precision,
                'N_recall': n.recall,
                'N_f1': n.f1,
            })

            print(
                f'{mode} / {method}：'
                f'W-F1={report.weighted_f1:.6f}，'
                f'N-F1={n.f1:.6f}'
            )

        # 每完成一组就保存汇总
        pd.DataFrame(summary_rows).to_csv(
            OUTPUT_ROOT / 'summary.csv',
            index=False,
            encoding='utf-8-sig',
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)

    print('\n===== A–E 完整结果 =====')
    print(summary.round(6).to_string(index=False))

    comparison = summary.pivot(
        index='model',
        columns='method',
        values='weighted_f1',
    )

    comparison['gain'] = (
        comparison['pair_aware'] - comparison['independent']
    )

    print('\n===== W-F1 对比 =====')
    print(comparison.round(6).to_string())

    print('\n汇总文件：', OUTPUT_ROOT / 'summary.csv')


if __name__ == '__main__':
    main()