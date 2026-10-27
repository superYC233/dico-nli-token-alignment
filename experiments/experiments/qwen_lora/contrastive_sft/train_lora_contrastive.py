import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig

from data_utils import load_data, get_datasets, tokenizer, build_prompt
from data_variants import build_train_dataset
from best_f1_callback import BestF1Callback
from contrastive_data import add_contrast_groups
from contrastive_trainer import ContrastiveSFTTrainer


mode = 'C'
seed = 42

# 本次新增的参数：先固定这一组探索，不表示已经找到最优值。
contrastive_weight = 0.1
temperature = 0.1
projection_dim = 128

output_dir = (
    f'D:/LLM/train_model/DiCo-NLI/'
    f'qwen3_0p6b_contrastive/run01/{mode}'
)

model_path = 'D:/LLM/model/Qwen3-0.6B-Base'


def main():
    set_seed(seed)
    prompt_template = build_prompt('{text1}', '{text2}')
    if 'Check both directions separately:' in prompt_template:
        raise ValueError('请保持旧短 Prompt，避免同时改变 Prompt 和损失。')

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto')

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
        bias='none',
        task_type='CAUSAL_LM'
    )

    lora_model = get_peft_model(model, lora_config)

    # 小投影层只用于训练时的对比损失，预测时不需要。
    # 临时保存/恢复随机数状态，不让新增层的初始化改变后续训练随机序列。
    with torch.random.fork_rng(devices=[]):
        lora_model.contrastive_head = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_dim, projection_dim)
        )

    train_args = SFTConfig(
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,

        bf16=True,
        completion_only_loss=True,
        packing=False,
        padding_free=False,
        # 全词表 SFT 保持不变，分块计算避免额外占用大量显存。
        loss_type='chunked_nll',

        seed=seed,
        data_seed=seed,

        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='best',
        load_best_model_at_end=True,
        metric_for_best_model='eval_pair_f1',
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,

        # 保留 contrast_group，让自定义 collator 取出同源编号。
        remove_unused_columns=False,
        logging_steps=20,
        report_to='none'
    )

    dataset = load_data()
    original_size = len(dataset['train'])

    dataset['train'] = build_train_dataset(
        train_dataset=dataset['train'],
        mode=mode
    )
    dataset['train'] = add_contrast_groups(dataset['train'])

    print('当前版本：', mode)
    print('原始训练样本数：', original_size)
    print('构造后训练样本数：', len(dataset['train']))
    print('Dev 样本数：', len(dataset['dev']))
    print('对比损失权重：', contrastive_weight)
    print('温度：', temperature)
    print('模型保存目录：', output_dir)

    sft_dataset = get_datasets(dataset)
    tokenizer.padding_side = 'right'

    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ['F', 'B', 'E', 'N']]
    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('F/B/E/N 必须分别是单 token。')
    label_ids = [ids[0] for ids in token_ids]
    if len(set(label_ids)) != 4:
        raise ValueError('四个标签的 token ID 必须不同。')

    trainer = ContrastiveSFTTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
        callbacks=[BestF1Callback()],

        label_ids=label_ids,
        contrastive_weight=contrastive_weight,
        temperature=temperature,
        prompt_template=prompt_template
    )

    expected_ids = tokenizer.encode(sft_dataset['train'][0]['prompt'], add_special_tokens=False)
    actual_ids = trainer.train_dataset[0]['input_ids']
    if actual_ids[:len(expected_ids)] != expected_ids:
        raise ValueError('实际训练输入与当前 Prompt 不一致，请检查缓存。')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset['train'].to_pandas()[['instance_id', 'pair_id', 'contrast_group', 'label']].to_csv(
        Path(output_dir) / 'contrast_groups.csv', index=False, encoding='utf-8-sig'
    )
    config = {
        'mode': mode, 'seed': seed, 'train_rows': len(dataset['train']),
        'model_path': model_path, 'prompt_template': prompt_template,
        'contrastive_weight': contrastive_weight, 'temperature': temperature,
        'projection_dim': projection_dim, 'training': train_args.to_dict()
    }
    (Path(output_dir) / 'experiment_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )

    trainer.train()

    best_dir = f'{output_dir}/best_adapter'
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 Adapter：', best_dir)


if __name__ == '__main__':
    main()

