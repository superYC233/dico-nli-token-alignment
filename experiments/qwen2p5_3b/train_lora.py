from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from data_utils import load_data, get_datasets, tokenizer

model_path = 'D:/LLM/model/Qwen2.5-3B-Instruct'

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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    completion_only_loss=True,
    packing=False,

    output_dir='D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_lora/run01',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    per_device_eval_batch_size=4,
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