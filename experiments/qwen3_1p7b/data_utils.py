"""只使用本次 1.7B 的 tokenizer；C 数据、预测与评分复用已有公共函数。"""
import json
import sys
from pathlib import Path

from transformers import AutoConfig

import config as cfg

for folder in [cfg.PROJECT_ROOT, cfg.QWEN_DIR]:
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

from data_variants import build_train_dataset
from experiments.generative_nli import (
    load_data as load_task_data,
    get_datasets as make_sft_datasets,
    score_methods as score_task_methods,
    get_label_ids, load_tokenizer, check_examples,
    predict_with_scores, pair_aware_decode,
)


def load_data():
    return load_task_data(cfg.PROJECT_ROOT, cfg.DATA_CACHE_DIR)


def build_prompt_template(tokenizer):
    messages = [{'role': 'user', 'content': cfg.TASK_PROMPT.strip()}]
    if not tokenizer.chat_template:
        raise ValueError('本地 tokenizer 缺少 chat_template，请检查模型是否下载完整。')
    # 1.7B 支持思考模式，这个分类实验必须显式关闭。
    template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    # 官方模板可能在输入末尾预填空的 <think> 块，这是正常的。
    # 它属于 Prompt，不是模型要生成的思考内容；真正监督的答案仍只有标签 + EOS。
    assistant_prefix = template.rsplit('<|im_start|>assistant\n', 1)[-1]
    if '<think>' in assistant_prefix and assistant_prefix.strip() != '<think>\n\n</think>':
        raise ValueError('助手前缀不是关闭思考后的空前缀，请检查 tokenizer 模板。')
    return template


def get_datasets(dataset, prompt_template):
    return make_sft_datasets(dataset, prompt_template)


def score_methods(frame, output_dir, gold_path=cfg.GOLD_PATH):
    return score_task_methods(frame, output_dir, gold_path)


def check_model_files(model_path):
    """只检查小型配置和文件名单，不把 1.7B 权重加载进内存或显存。"""
    model_path = Path(model_path)
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        weight_files = sorted(set(index['weight_map'].values()))
    else:
        weight_files = ['model.safetensors']
    missing = [name for name in weight_files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            '模型分片尚未下载完成：' + ', '.join(missing)
            + '\n请等 download_model.py 下载完成后再训练。'
        )
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if config.model_type != 'qwen3' or config.hidden_size != 2048 or config.num_hidden_layers != 28:
        raise ValueError('模型配置不是预期的 Qwen3-1.7B，请检查 MODEL_PATH。')
    total = sum((model_path / name).stat().st_size for name in weight_files)
    print(f'权重分片已就绪，文件总量约 {total / 1024 ** 3:.2f} GiB。')
