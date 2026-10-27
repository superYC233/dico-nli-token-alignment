
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from data_utils import load_data, get_datasets, tokenizer


if __name__ == '__main__':
    set_seed(42)

    script_dir = QWEN_DIR

    # 1. 读取原始训练集、验证集
    dataset = load_data()
    original_size = len(dataset['train'])

    # 2. 读取候选困难样本的 ID
    candidate_path = (
        script_dir
        / 'outputs/qwen3_0p6b_lora/run01/train_analysis'
        / 'hard_candidates.csv'
    )

    candidates = pd.read_csv(
        candidate_path,
        keep_default_na=False
    )
    hard_ids = set(candidates['instance_id'])

    # 从原始训练集中取回样本，沿用原文本和标签
    hard_dataset = dataset['train'].filter(
        lambda example: example['instance_id'] in hard_ids
    )

    if len(hard_dataset) != len(hard_ids) or not hard_ids:
        raise ValueError('候选 ID 为空，或不能全部匹配到训练集')

    # 3. 每条候选样本额外加入一次
    # 原训练集中已有一次，追加后共出现两次
    dataset['train'] = concatenate_datasets([
        dataset['train'],
        hard_dataset
    ])

    print('原始训练样本数：', original_size)
    print('额外加入样本数：', len(hard_dataset))
    print('重采样后样本数：', len(dataset['train']))

    # 转换成原来的 prompt / completion 格式
    sft_dataset = get_datasets(dataset)

    # 4. 从 Base 新建 LoRA，不加载旧 Adapter
    model = AutoModelForCausalLM.from_pretrained(
        'D:/LLM/model/Qwen3-0.6B-Base',
        dtype='auto'
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules='all-linear',
        bias='none',
        task_type='CAUSAL_LM'
    )

    lora_model = get_peft_model(model, lora_config)

    # 5. 保持现有训练设置，使用新的保存目录
    train_args = SFTConfig(
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        bf16=True,
        completion_only_loss=True,
        packing=False,

        output_dir='D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_lora/run03_hard',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
        seed=42
    )

    trainer = SFTTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer
    )

    trainer.train()

    # 保存验证集 Loss 最低的 Adapter
    trainer.save_model()