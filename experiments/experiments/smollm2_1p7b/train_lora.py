import json

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

import smol_config as cfg
from smol_utils import (
    load_data, load_tokenizer, get_datasets, build_train_dataset, check_examples,
)
from best_f1_callback import BestF1Callback


def main():
    # 先检查本地文件和数据，再加载大模型。
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    model_config = AutoConfig.from_pretrained(str(cfg.MODEL_PATH), local_files_only=True)
    if model_config.model_type != 'llama':
        raise ValueError('当前目录不是 SmolLM2 使用的 Llama 架构，请检查 MODEL_PATH。')

    dataset = load_data()
    original_size = len(dataset['train'])
    dataset['train'] = build_train_dataset(dataset['train'], mode=cfg.MODE)
    sft_dataset = get_datasets(dataset)
    print('当前版本：', cfg.MODE)
    print('原始训练样本数：', original_size)
    print('构造后训练样本数：', len(dataset['train']))
    print('Dev 样本数：', len(dataset['dev']))
    check_examples(sft_dataset, tokenizer, cfg.MAX_LENGTH)

    set_seed(cfg.SEED)
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.MODEL_PATH), dtype='auto', local_files_only=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
        bias='none',
        task_type='CAUSAL_LM',
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    train_args = SFTConfig(
        learning_rate=1e-4,
        num_train_epochs=5,
        # 先适配 12GB 显存；4 × 4 = 有效 batch 16。
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        padding_free=False,
        max_length=cfg.MAX_LENGTH,
        loss_type='chunked_nll',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg.SEED,
        data_seed=cfg.SEED,
        output_dir=str(cfg.OUTPUT_DIR),
        eval_strategy='epoch',
        save_strategy='best',
        load_best_model_at_end=True,
        metric_for_best_model='eval_pair_f1',
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        logging_steps=20,
        report_to='none',
    )
    trainer = SFTTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
        callbacks=[BestF1Callback(sft_dataset['dev'], tokenizer)],
    )

    # 检查实际交给 Trainer 的输入和 completion mask。
    first = trainer.train_dataset[0]
    prompt_ids = tokenizer(sft_dataset['train'][0]['prompt'])['input_ids']
    if first['input_ids'][:len(prompt_ids)] != prompt_ids:
        raise ValueError('Trainer 实际输入没有使用当前 Prompt。')
    supervised_ids = [token for token, label in zip(first['input_ids'], first['labels']) if label != -100]
    expected = tokenizer.encode(sft_dataset['train'][0]['completion'], add_special_tokens=False)
    if supervised_ids != expected + [tokenizer.eos_token_id]:
        raise ValueError('实际训练标签不是答案 token + EOS，请检查 tokenizer。')

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        'model_id': cfg.MODEL_ID, 'model_path': str(cfg.MODEL_PATH),
        'mode': cfg.MODE, 'train_rows': len(dataset['train']),
        'prompt_template': cfg.PROMPT_TEMPLATE, 'labels': cfg.LABELS,
        'prediction_batch_size': cfg.PREDICT_BATCH_SIZE,
        'readout': 'four-candidate argmax, with unrestricted generation recorded separately',
        'training': train_args.to_dict(),
    }
    config_path = cfg.OUTPUT_DIR / 'run_config.json'
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    trainer.train()
    trainer.save_model(str(cfg.BEST_DIR))
    tokenizer.save_pretrained(str(cfg.BEST_DIR))
    (cfg.BEST_DIR / 'run_config.json').write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 Adapter：', cfg.BEST_DIR)


if __name__ == '__main__':
    main()
