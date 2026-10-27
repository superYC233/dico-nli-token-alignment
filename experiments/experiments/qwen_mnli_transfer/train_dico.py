"""第二步：接着训练第一阶段 Adapter；DiCo 部分仍是原 C+r16 SFT。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from experiments.qwen_mnli_transfer import config as cfg
from experiments.qwen_mnli_transfer.common import (
    check_cuda, check_empty_output, check_mnli_adapter, check_supervision,
    load_tokenizer, load_training_model, save_adapter, write_json,
)
from experiments.qwen_mnli_transfer.data_utils import prepare_dico, check_examples
from experiments.qwen_mnli_transfer.best_f1_callback import BestF1Callback


def main():
    check_empty_output(cfg.DICO_OUTPUT_DIR)
    check_cuda()
    if cfg.USE_MNLI_ADAPTER:
        parent_manifest = check_mnli_adapter()
        adapter_path = cfg.MNLI_ADAPTER_DIR
    else:
        # 可选的同代码对照：不补 MNLI，其余配置完全相同，输出到独立目录。
        parent_manifest = None
        adapter_path = None

    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    dataset = prepare_dico()
    check_examples(dataset, tokenizer, cfg.MAX_LENGTH)
    set_seed(cfg.SEED)
    model = load_training_model(adapter_path)
    tokenizer.padding_side = 'right'
    train_args = SFTConfig(
        output_dir=str(cfg.DICO_OUTPUT_DIR),
        learning_rate=cfg.DICO_LEARNING_RATE,
        num_train_epochs=cfg.DICO_EPOCHS,
        per_device_train_batch_size=cfg.DICO_BATCH_SIZE,
        gradient_accumulation_steps=cfg.DICO_GRAD_ACCUMULATION,
        per_device_eval_batch_size=32,
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
        eval_strategy='epoch',
        prediction_loss_only=True,
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
        model=model, args=train_args,
        train_dataset=dataset['train'], eval_dataset=dataset['dev'],
        processing_class=tokenizer,
        callbacks=[BestF1Callback(dataset['dev'], tokenizer)],
    )
    check_supervision(trainer, dataset['train'], tokenizer)
    metadata = {
        'stage': 'dico', 'mode': cfg.MODE, 'train_rows': len(dataset['train']),
        'prompt_template': cfg.DICO_PROMPT,
        'parent_adapter': str(adapter_path) if adapter_path is not None else None,
        'parent_adapter_sha256': parent_manifest['adapter_sha256'] if parent_manifest else None,
        'prediction_batch_size': cfg.PREDICT_BATCH_SIZE,
        'training': train_args.to_dict(),
    }
    write_json(cfg.DICO_OUTPUT_DIR / 'run_config.json', metadata)
    trainer.train()
    metadata['best_checkpoint'] = trainer.state.best_model_checkpoint
    metadata['best_pair_f1'] = trainer.state.best_metric
    save_adapter(trainer, tokenizer, cfg.BEST_ADAPTER_DIR, metadata)
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最终可直接加载的 Adapter：', cfg.BEST_ADAPTER_DIR)
    print('下一步运行同目录的 evaluate_dico.py。')


if __name__ == '__main__':
    main()
