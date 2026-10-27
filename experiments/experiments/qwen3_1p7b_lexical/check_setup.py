"""检查真实 tokenizer、全部输入和词汇特征，不加载 1.7B 权重。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.data_utils import (
    load_data, load_tokenizer, build_prompt_template, build_train_dataset,
    get_datasets, check_examples, prepare_training_dataset, LexicalCollator, check_model_files,
)


def main():
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    template = build_prompt_template(tokenizer)
    dataset = load_data()
    dataset['train'] = build_train_dataset(dataset['train'], cfg.MODE)
    formatted = get_datasets(dataset, template)
    check_examples(formatted, tokenizer, cfg.MAX_LENGTH)
    encoded = prepare_training_dataset(formatted['train'], tokenizer)
    batch = LexicalCollator(tokenizer.pad_token_id)([encoded[0], encoded[-1]])
    assert batch['lexical_features'].shape == (2, 20)
    assert torch.isfinite(batch['lexical_features']).all()
    assert ((batch['labels'] != -100).sum(dim=1) == 2).all()
    check_model_files(cfg.MODEL_PATH)
    print('Train / Dev：', len(encoded), '/', len(formatted['dev']))
    print('特征维度：20；监督位置：标签 + EOS。')
    print('检查通过，可以运行 train_lora_lexical.py。')


if __name__ == '__main__':
    main()
