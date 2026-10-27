"""只检查分词、监督位置与小型Qwen2损失；不训练真实3B模型。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig

from experiments.qwen25_3b_reason import config as cfg
from experiments.qwen25_3b_reason.train_lora_reason import load_tokenizer
from experiments.qwen_lora.reason_sft.reason_data import load_train_frame, build_dataset, ReasonCollator
from experiments.qwen_lora.reason_sft.reason_trainer import ReasonSFTTrainer


def main():
    tokenizer = load_tokenizer()
    label_prompt, reason_prompt = cfg.build_prompt_templates(tokenizer)
    frame = load_train_frame()
    dataset = build_dataset(frame, tokenizer, True, label_prompt, reason_prompt, cfg.MAX_LENGTH)
    assert len(dataset) == 3517 and int(frame.use_reason.sum()) == 2040
    assert all(len(tokenizer.encode(label, add_special_tokens=False)) == 1 for label in cfg.LABELS)
    assert '<|im_start|>assistant\n' in label_prompt
    for row, example in zip(frame.itertuples(), dataset):
        count = len(tokenizer.encode(label_prompt.format(text1=row.text1, text2=row.text2), add_special_tokens=False))
        assert example['labels'][:count] == [-100] * count
        assert [x for x in example['labels'] if x != -100] == tokenizer.encode(cfg.LABEL_MAP[row.label], add_special_tokens=False) + [tokenizer.eos_token_id]
        if row.use_reason:
            count = len(tokenizer.encode(reason_prompt.format(text1=row.text1, text2=row.text2), add_special_tokens=False))
            assert example['reason_labels'][:count] == [-100] * count
            target = [x for x in example['reason_labels'] if x != -100]
            assert tokenizer.decode(target, skip_special_tokens=True) == row.reason
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    dev_dataset = build_dataset(dev, tokenizer, False, label_prompt=label_prompt, max_length=cfg.MAX_LENGTH)
    assert all(not example['reason_input_ids'] for example in dev_dataset)

    torch.set_num_threads(2)
    backbone = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=False,
    ))
    model = get_peft_model(backbone, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, target_modules='all-linear', task_type='CAUSAL_LM',
    ))
    args = SFTConfig(
        output_dir=str(cfg.RESULTS_ROOT / 'setup_check'), use_cpu=True, bf16=False,
        loss_type='chunked_nll', packing=False, padding_free=False,
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none', save_strategy='no', prediction_loss_only=True,
    )
    collator = ReasonCollator(tokenizer.pad_token_id)
    trainer = ReasonSFTTrainer(model=model, args=args, train_dataset=dataset.select(range(2)),
                              processing_class=tokenizer, data_collator=collator, reason_weight=cfg.REASON_WEIGHT)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    batch = collator([dataset[index] for index in frame.index[frame.use_reason][:2]])
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if 'lora_' in name)
    assert trainer.optimizer is None
    result = {'train_rows': len(dataset), 'reason_rows': int(frame.use_reason.sum()), 'dev_rows': len(dev),
              'chat_template_and_masks_checked': True, 'dev_has_no_reason': True,
              'tiny_qwen2_chunked_loss_and_lora_backward': True, 'gradient_checkpointing': True,
              'effective_batch': cfg.TRAIN_BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS,
              'real_3b_loaded': False, 'optimizer_created': False, 'real_model_training_performed': False}
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.PREPARED_DIR / 'setup_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
