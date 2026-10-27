"""右键运行：加载最佳Adapter，使用训练时保存的聊天Prompt预测Dev。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.qwen25_3b_reason import config as cfg
from experiments.qwen25_3b_reason.train_lora_reason import load_tokenizer
from experiments.qwen_lora.reason_sft.reason_evaluation import predict_dev, score_methods


def main(run_name=None):
    run_name = run_name or cfg.RUN_NAME
    model_dir = cfg.MODEL_ROOT / run_name / cfg.MODE
    settings = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    adapter_path = model_dir / 'best_adapter'
    tokenizer = load_tokenizer(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(settings['model_path'], dtype='auto', local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True).to('cuda')
    model.eval()
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    frame = predict_dev(model, tokenizer, dev, settings['label_prompt'], cfg.PREDICT_BATCH_SIZE)
    score_methods(frame, cfg.RESULTS_ROOT / run_name / cfg.MODE / 'evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name')
    args = parser.parse_args()
    main(args.run_name)
