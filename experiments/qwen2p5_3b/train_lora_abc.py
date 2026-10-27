from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from data_utils import load_data, get_datasets, tokenizer
from data_variants import build_train_dataset
from best_f1_callback import BestF1Callback

# 每次选择一个版本训练
mode = 'C'
seed = 42

output_dir = (
    f'D:/LLM/train_model/DiCo-NLI/'
    f'qwen2p5_3b_abc/run01_best/{mode}'
)

# 放在模型加载和 LoRA 初始化之前
set_seed(seed)

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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,

    bf16=True,
    completion_only_loss=True,
    packing=False,

    seed=seed,
    data_seed=seed,

    output_dir=output_dir,
    eval_strategy='epoch',

    # 只有 F1 刷新纪录时，才保存 checkpoint
    save_strategy='best',

    load_best_model_at_end=True,
    metric_for_best_model='eval_pair_f1',
    greater_is_better=True,

    # 只保留最佳 checkpoint
    save_total_limit=1,

    # 只保存模型，不保存优化器等完整续训状态
    save_only_model=True,

    logging_steps=20,
    report_to='none'
)

dataset = load_data()

original_size = len(dataset['train'])

# 先构造训练数据，再生成 Prompt
dataset['train'] = build_train_dataset(
    train_dataset=dataset['train'],
    mode=mode
)

print('当前版本：', mode)
print('原始训练样本数：', original_size)
print('构造后训练样本数：', len(dataset['train']))
print('Dev 样本数：', len(dataset['dev']))

sft_dataset = get_datasets(dataset)

# 供训练批次补齐使用
tokenizer.padding_side = 'right'


trainer = SFTTrainer(
    model=lora_model,
    args=train_args,
    train_dataset=sft_dataset['train'],
    eval_dataset=sft_dataset['dev'],
    processing_class=tokenizer,

    callbacks=[BestF1Callback()]
)

trainer.train()

best_dir = f'{output_dir}/best_adapter'

trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)

print('最佳 Adapter 路径：', trainer.state.best_model_checkpoint)
print('最佳 pair-aware F1：', trainer.state.best_metric)