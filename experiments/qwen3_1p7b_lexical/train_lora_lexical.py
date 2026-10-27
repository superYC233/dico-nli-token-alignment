import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.data_utils import (
    load_data, load_tokenizer, build_train_dataset, build_prompt_template, get_datasets,
    check_examples, check_model_files, prepare_training_dataset, LexicalCollator, get_label_ids,
)
from experiments.qwen3_1p7b_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.qwen3_1p7b_lexical.model import QwenLexicalModel
from experiments.qwen3_1p7b_lexical.lexical_trainer import LexicalTrainer
from experiments.qwen3_1p7b_lexical.best_f1_callback import BestF1Callback


def main():
    if cfg.OUTPUT_DIR.exists() or cfg.RESULTS_DIR.exists():
        raise FileExistsError('本次输出目录已存在，请修改 config.py 的 RUN_NAME。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练需要 CUDA；CPU 验证请运行 test_cpu.py。')
    check_model_files(cfg.MODEL_PATH)
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    template = build_prompt_template(tokenizer)
    dataset = load_data()
    original_size = len(dataset['train'])
    dataset['train'] = build_train_dataset(dataset['train'], mode=cfg.MODE)
    formatted = get_datasets(dataset, template)
    check_examples(formatted, tokenizer, cfg.MAX_LENGTH)
    train_dataset = prepare_training_dataset(formatted['train'], tokenizer)
    dev_dataset = prepare_training_dataset(formatted['dev'], tokenizer)
    print('当前版本：', cfg.MODE)
    print('原始 / 构造后 Train：', original_size, '/', len(train_dataset))
    print('Dev：', len(dev_dataset))
    print('词汇分支：', cfg.USE_LEXICAL_FEATURES)

    set_seed(cfg.SEED)
    base = AutoModelForCausalLM.from_pretrained(
        str(cfg.MODEL_PATH), dtype='auto', local_files_only=True,
    )
    base.config.use_cache = False
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
    )
    language_model = get_peft_model(base, lora_config)
    model = QwenLexicalModel(
        language_model, get_label_ids(tokenizer), tokenizer.eos_token_id,
        feature_size=len(FEATURE_NAMES), lexical_hidden=cfg.LEXICAL_HIDDEN_SIZE,
        chunk_size=cfg.LOSS_CHUNK_SIZE, use_lexical_features=cfg.USE_LEXICAL_FEATURES,
    )
    language_model.print_trainable_parameters()
    print('词汇分支参数量：', sum(parameter.numel() for parameter in model.lexical_head.parameters()))

    # 需要自定义分块损失和模型保存，因此使用 Trainer 子类；监督仍是原全词表 SFT。
    # 新分支只作用于分类标签位置，EOS 监督完整保留。
    args = TrainingArguments(
        output_dir=str(cfg.OUTPUT_DIR),
        learning_rate=1e-4, num_train_epochs=5,
        per_device_train_batch_size=16, gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg.SEED, data_seed=cfg.SEED,
        eval_strategy='epoch', save_strategy='best', load_best_model_at_end=True,
        metric_for_best_model='eval_pair_f1', greater_is_better=True,
        save_total_limit=1, save_only_model=True,
        remove_unused_columns=False, prediction_loss_only=True,
        label_names=['labels'],
        logging_steps=20, report_to='none',
    )
    feature_file = Path(__file__).resolve().parent / 'lexical_features.py'
    run_config = {
        'model_id': cfg.MODEL_ID, 'model_path': str(cfg.MODEL_PATH),
        'mode': cfg.MODE, 'train_rows': len(train_dataset), 'seed': cfg.SEED,
        'enable_thinking': False, 'prompt_template': template,
        'labels': cfg.LABELS, 'label_ids': get_label_ids(tokenizer),
        'use_lexical_features': cfg.USE_LEXICAL_FEATURES,
        'feature_names': FEATURE_NAMES, 'feature_version': FEATURE_VERSION,
        'feature_source_sha256': hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        'lexical_hidden_size': cfg.LEXICAL_HIDDEN_SIZE,
        'lexical_learning_rate': cfg.LEXICAL_LEARNING_RATE,
        'loss_chunk_size': cfg.LOSS_CHUNK_SIZE,
        'loss': 'full-vocabulary CE on label + EOS; lexical correction only on label position',
        'prediction_batch_size': cfg.PREDICT_BATCH_SIZE,
        'training': args.to_dict(),
    }
    trainer = LexicalTrainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=LexicalCollator(tokenizer.pad_token_id), processing_class=tokenizer,
        lexical_learning_rate=cfg.LEXICAL_LEARNING_RATE, run_config=run_config,
        callbacks=[BestF1Callback(formatted['dev'], tokenizer)],
    )
    trainer.train()
    # load_best_model_at_end 会同时恢复最佳 LoRA 和词汇分支，然后再导出。
    trainer.save_model(str(cfg.BEST_DIR))
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 LoRA + 词汇分支：', cfg.BEST_DIR)


if __name__ == '__main__':
    main()
