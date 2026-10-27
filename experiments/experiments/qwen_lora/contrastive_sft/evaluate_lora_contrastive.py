import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import pandas as pd

from data_utils import build_prompt
from compare_decoding import evaluate_with_scores
from evaluate_lora_pair import load_lora_model, pair_aware_decode, score_methods


model_path = 'D:/LLM/model/Qwen3-0.6B-Base'
adapter_path = (
    'D:/LLM/train_model/DiCo-NLI/'
    'qwen3_0p6b_contrastive/run01/C/best_adapter'
)
output_dir = QWEN_DIR / 'outputs/qwen3_0p6b_contrastive/run01/C'


def main():
    saved = json.loads((Path(adapter_path) / 'contrastive_config.json').read_text(encoding='utf-8'))
    if saved['prompt_template'] != build_prompt('{text1}', '{text2}'):
        raise ValueError('当前 Prompt 与训练时不一致，请恢复训练使用的旧短 Prompt。')

    # 只加载标准 LoRA 即可；对比投影层只用于训练，不参与标签生成。
    model = load_lora_model(model_path, adapter_path)
    records = evaluate_with_scores(model, batch_size=16, split='dev')
    df = pair_aware_decode(pd.DataFrame(records))

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    print('保存样本数：', len(df))
    print('结果目录：', output_dir)
    score_methods(df, output_dir)


if __name__ == '__main__':
    main()

