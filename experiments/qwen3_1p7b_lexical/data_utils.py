import json
from pathlib import Path

import torch
from transformers import AutoConfig

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.lexical_features import build_lexical_features
from experiments.qwen3_1p7b_lexical.data_variants import build_train_dataset
from experiments.generative_nli import (
    load_data as load_task_data, get_datasets as format_datasets,
    load_tokenizer, get_label_ids, check_examples, pair_aware_decode,
    score_methods as score_task_methods,
)


def load_data():
    return load_task_data(cfg.PROJECT_ROOT, cfg.DATA_CACHE_DIR)


def build_prompt_template(tokenizer):
    messages = [{'role': 'user', 'content': cfg.TASK_PROMPT.strip()}]
    if not tokenizer.chat_template:
        raise ValueError('tokenizer 缺少 chat_template。')
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def get_datasets(dataset, prompt_template):
    return format_datasets(dataset, prompt_template)


def prepare_training_dataset(dataset, tokenizer):
    label_ids = get_label_ids(tokenizer)

    def encode(row):
        prompt_ids = tokenizer(row['prompt'])['input_ids']
        full_ids = tokenizer(row['prompt'] + row['completion'] + tokenizer.eos_token)['input_ids']
        expected = [label_ids[cfg.LABELS.index(row['completion'])], tokenizer.eos_token_id]
        if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[len(prompt_ids):] != expected:
            raise ValueError(f'Prompt/答案边界不正确：{row["instance_id"]}')
        if len(full_ids) > cfg.MAX_LENGTH:
            raise ValueError(f'输入被截断：{row["instance_id"]}')
        return {
            'input_ids': full_ids, 'attention_mask': [1] * len(full_ids),
            'labels': [-100] * len(prompt_ids) + expected,
            'lexical_features': build_lexical_features(row['text1'], row['text2']),
        }

    prepared = dataset.map(
        encode, remove_columns=dataset.column_names,
        load_from_cache_file=False, desc='准备标签/EOS监督及词汇特征',
    )
    if len(prepared) != len(dataset):
        raise ValueError('准备输入后样本数量发生了变化。')
    return prepared


class LexicalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(row['input_ids']) for row in examples)
        ids, masks, labels = [], [], []
        for row in examples:
            padding = length - len(row['input_ids'])
            ids.append(row['input_ids'] + [self.pad_token_id] * padding)
            masks.append(row['attention_mask'] + [0] * padding)
            labels.append(row['labels'] + [-100] * padding)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'lexical_features': torch.tensor([row['lexical_features'] for row in examples], dtype=torch.float32),
        }


def score_methods(frame, output_dir, gold_path=cfg.GOLD_PATH):
    return score_task_methods(frame, output_dir, gold_path)


def check_model_files(model_path):
    model_path = Path(model_path)
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        files = set(index['weight_map'].values())
    else:
        files = {'model.safetensors'}
    missing = [name for name in files if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f'本地模型分片缺失：{missing}')
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if model_config.model_type != 'qwen3' or model_config.hidden_size != 2048:
        raise ValueError('MODEL_PATH 不是预期的 Qwen3-1.7B。')
    print('Qwen3-1.7B 本地权重已就绪。')
