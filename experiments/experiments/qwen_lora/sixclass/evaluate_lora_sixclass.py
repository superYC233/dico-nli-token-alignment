
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import os
import sys
from pathlib import Path

import pandas as pd

script_dir = QWEN_DIR
sys.path.insert(0, str(script_dir.parent.parent))

from sixclass_data_utils import evaluate_sixclass_with_scores
from evaluate_lora_pair import (
    load_lora_model,
    pair_aware_decode,
    score_methods
)


if __name__ == '__main__':
    os.chdir(script_dir)

    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'

    adapter_path = (
        'D:/LLM/train_model/DiCo-NLI/'
        'qwen3_0p6b_sixclass/run01/C/best_adapter'
    )

    model = load_lora_model(
        model_path,
        adapter_path
    )

    records = evaluate_sixclass_with_scores(
        model,
        batch_size=16
    )

    df = pair_aware_decode(pd.DataFrame(records))

    output_dir = (
        script_dir
        / 'outputs/qwen3_0p6b_sixclass/run01/C'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / 'predictions_with_scores.csv',
        index=False,
        encoding='utf-8-sig'
    )

    score_methods(df, output_dir)