"""第一步：运行本文件，在 MNLI 上训练并保存 LoRA，不改原模型。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from experiments.qwen_mnli_transfer import config as cfg
from experiments.qwen_mnli_transfer.common import (
    check_cuda, check_empty_output, check_supervision, load_tokenizer,
    load_training_model, save_adapter, write_json,
)
from experiments.qwen_mnli_transfer.data_utils import prepare_mnli


def main():
    check_empty_output(cfg.MNLI_OUTPUT_DIR)
    check_cuda()
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    dataset = prepare_mnli(tokenizer)
    print('MNLI 训练 / 内部验证：', len(dataset['train']), '/', len(dataset['validation']))
    print('单批样本 / 梯度累计：', cfg.MNLI_BATCH_SIZE, '/', cfg.MNLI_GRAD_ACCUMULATION)
    print('模型输出目录：', cfg.MNLI_OUTPUT_DIR)

    set_seed(cfg.SEED)
    model = load_training_model()
    tokenizer.padding_side = 'right'
    train_args = SFTConfig(
        output_dir=str(cfg.MNLI_OUTPUT_DIR),
        learning_rate=cfg.MNLI_LEARNING_RATE,
        num_train_epochs=cfg.MNLI_EPOCHS,
        per_device_train_batch_size=cfg.MNLI_BATCH_SIZE,
        gradient_accumulation_steps=cfg.MNLI_GRAD_ACCUMULATION,
        per_device_eval_batch_size=cfg.MNLI_EVAL_BATCH_SIZE,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        padding_free=False,
        max_length=cfg.MAX_LENGTH,
        # 仍是完整词表 CE；仅分块计算，减少长句的 logits 显存。
        loss_type='chunked_nll',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg.SEED,
        data_seed=cfg.SEED,
        eval_strategy='epoch',
        prediction_loss_only=True,
        # 第一阶段预先固定训练轮数，只在最后保存一个 Adapter。
        save_strategy='no',
        load_best_model_at_end=False,
        logging_steps=20,
        report_to='none',
    )
    trainer = SFTTrainer(
        model=model, args=train_args,
        train_dataset=dataset['train'], eval_dataset=dataset['validation'],
        processing_class=tokenizer,
    )
    check_supervision(trainer, dataset['train'], tokenizer)
    metadata = {
        'stage': 'mnli', 'prompt_template': cfg.NLI_PROMPT,
        'prepared_manifest': str(cfg.PREPARED_DIR / 'manifest.json'),
        'training': train_args.to_dict(),
    }
    write_json(cfg.MNLI_OUTPUT_DIR / 'run_config.json', metadata)
    trainer.train()
    save_adapter(trainer, tokenizer, cfg.MNLI_ADAPTER_DIR, metadata)
    print('第一阶段完成，MNLI Adapter：', cfg.MNLI_ADAPTER_DIR)
    print('下一步运行同目录的 train_dico.py。')


if __name__ == '__main__':
    main()
