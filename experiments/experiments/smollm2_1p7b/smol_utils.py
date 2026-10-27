"""SmolLM2 的入口适配；公共预测代码显式接收本模型 tokenizer。"""
import sys

import smol_config as cfg

for folder in [cfg.PROJECT_ROOT, cfg.QWEN_DIR]:
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

from data_variants import build_train_dataset
from experiments.generative_nli import (
    load_data as load_task_data,
    get_datasets as make_sft_datasets,
    score_methods as score_task_methods,
    build_sft_example, get_label_ids, load_tokenizer, check_examples,
    predict_with_scores, pair_aware_decode,
)


def load_data():
    return load_task_data(cfg.PROJECT_ROOT, cfg.DATA_CACHE_DIR)


def get_datasets(dataset, prompt_template=cfg.PROMPT_TEMPLATE):
    return make_sft_datasets(dataset, prompt_template)


def score_methods(frame, output_dir, gold_path=cfg.GOLD_PATH):
    return score_task_methods(frame, output_dir, gold_path)
