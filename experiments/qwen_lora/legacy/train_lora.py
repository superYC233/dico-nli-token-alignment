
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from data_utils import load_data, get_datasets, tokenizer

model_path = 'D:/LLM/model/Qwen3-0.6B-Base'

model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto')

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules='all-linear',
    bias='none',
    task_type='CAUSAL_LM'
)

lora_model = get_peft_model(model, lora_config)
# lora_model.print_trainable_parameters()

train_args = SFTConfig(
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    bf16=True,
    completion_only_loss=True,
    packing=False,

    output_dir='D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_lora/run02_prompt',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    per_device_eval_batch_size=32,
    greater_is_better=False,
    save_total_limit=2
)

dataset = load_data()
sft_dataset = get_datasets(dataset)


trainer = SFTTrainer(
    model=lora_model,
    args=train_args,
    train_dataset=sft_dataset['train'],
    eval_dataset=sft_dataset['dev'],
    processing_class=tokenizer
)

trainer.train()
trainer.save_model()