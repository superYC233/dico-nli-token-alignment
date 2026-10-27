"""两步 CPU 集成检查：使用随机小模型，不下载/训练真正的 SmolLM2。"""
import torch
from accelerate import PartialState
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

import smol_config as cfg
from best_f1_callback import BestF1Callback
from smol_utils import (
    load_data, get_datasets, build_train_dataset, check_examples,
    get_label_ids, predict_with_scores,
)


def main():
    PartialState(cpu=True)
    torch.set_num_threads(2)
    work = cfg.DATA_CACHE_DIR / 'cpu_smoke'
    work.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    raw['train'] = build_train_dataset(raw['train'], 'C')
    assert len(raw['train']) == 3517 and len(raw['dev']) == 660
    formatted = get_datasets(raw)

    backend = Tokenizer(WordLevel(
        {'<unk>': 0, '<eos>': 1, '<pad>': 2, 'F': 3, 'B': 4, 'E': 5, 'N': 6},
        unk_token='<unk>',
    ))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='<unk>', eos_token='<eos>', pad_token='<pad>',
    )
    check_examples(formatted, tokenizer, 512)
    selected = []
    for short in cfg.LABELS:
        selected.extend([i for i, row in enumerate(formatted['train']) if row['completion'] == short][:2])
    train = formatted['train'].select(selected)

    set_seed(42)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=None, eos_token_id=1, pad_token_id=2,
    )
    base = AutoModelForCausalLM.from_config(config)
    base.save_pretrained(work / 'base')
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, target_modules='all-linear', lora_dropout=0.05, task_type='CAUSAL_LM',
    ))
    args = SFTConfig(
        output_dir=str(work / 'run'), use_cpu=True, bf16=False, max_steps=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, per_device_eval_batch_size=2,
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
    control = predict_with_scores(trainer.model, tokenizer, formatted['dev'].select(range(2)), 2)
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(work / 'base', local_files_only=True), str(work / 'best'),
    )
    reloaded.eval()
    restored = predict_with_scores(reloaded, tokenizer, formatted['dev'].select(range(2)), 2)
    for column in ['prediction', 'logit_F', 'logit_B', 'logit_E', 'logit_N']:
        assert control[column].equals(restored[column]), column
    print('CPU 集成检查通过：C3517、completion mask、梯度累积、F1 选模和 Adapter 重载。')


if __name__ == '__main__':
    main()
