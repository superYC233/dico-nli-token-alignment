"""CPU 两步检查：真实 4B tokenizer + 随机小 Qwen，不加载 4B 权重。"""
import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, Qwen3Config, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

import config as cfg
from best_f1_callback import BestF1Callback
from data_utils import (
    load_data, get_datasets, build_train_dataset, load_tokenizer,
    build_prompt_template, check_examples, get_label_ids, predict_with_scores,
)


def main():
    PartialState(cpu=True)
    torch.set_num_threads(2)
    work = cfg.DATA_CACHE_DIR / 'cpu_smoke'
    work.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    template = build_prompt_template(tokenizer)
    raw = load_data()
    raw['train'] = build_train_dataset(raw['train'], 'C')
    formatted = get_datasets(raw, template)
    check_examples(formatted, tokenizer, cfg.MAX_LENGTH)
    selected = []
    for short in cfg.LABELS:
        selected.extend([i for i, row in enumerate(formatted['train']) if row['completion'] == short][:2])
    train = formatted['train'].select(selected)

    set_seed(42)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    base = AutoModelForCausalLM.from_config(config)
    base.save_pretrained(work / 'base')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, target_modules='all-linear', lora_dropout=0.05, task_type='CAUSAL_LM',
    ))
    args = SFTConfig(
        output_dir=str(work / 'run'), use_cpu=True, bf16=False, max_steps=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, per_device_eval_batch_size=1,
        completion_only_loss=True, packing=False, padding_free=False, max_length=512,
        loss_type='chunked_nll', gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=1, save_strategy='best',
        load_best_model_at_end=True, metric_for_best_model='eval_pair_f1',
        greater_is_better=True, save_total_limit=1, save_only_model=True,
        report_to='none', logging_steps=1, dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=train,
        processing_class=tokenizer,
        callbacks=[BestF1Callback(formatted['dev'], tokenizer, batch_size=16)],
    )
    first = trainer.train_dataset[0]
    prompt_ids = tokenizer(train[0]['prompt'])['input_ids']
    assert first['input_ids'][:len(prompt_ids)] == prompt_ids
    supervised = [token for token, label in zip(first['input_ids'], first['labels']) if label != -100]
    assert supervised == [get_label_ids(tokenizer)[0], tokenizer.eos_token_id]

    trainer.train()
    assert trainer.state.best_model_checkpoint is not None
    assert trainer.state.best_metric is not None
    assert tokenizer.padding_side == 'right'
    trainer.save_model(str(work / 'best'))
    tokenizer.save_pretrained(str(work / 'best'))
    before = predict_with_scores(trainer.model, tokenizer, formatted['dev'].select(range(2)), 2)
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(work / 'base', local_files_only=True), str(work / 'best'),
    )
    reloaded.eval()
    after = predict_with_scores(reloaded, tokenizer, formatted['dev'].select(range(2)), 2)
    for column in ['prediction', 'logit_F', 'logit_B', 'logit_E', 'logit_N']:
        assert before[column].equals(after[column]), column
    print('CPU 检查通过：真实 4B tokenizer、答案/EOS mask、分块 CE、最佳 F1 选模、Adapter 重载。')


if __name__ == '__main__':
    main()
