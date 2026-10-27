"""右键运行：Qwen2.5-3B-Instruct：C+r16标签SFT + 0.2 * 文字解释SFT。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig

from experiments.qwen25_3b_reason import config as cfg
from experiments.qwen_lora.reason_sft.reason_data import (
    load_train_frame, build_dataset, ReasonCollator, save_prepared, file_hash,
)
from experiments.qwen_lora.reason_sft.reason_trainer import ReasonSFTTrainer
from experiments.qwen_lora.reason_sft.reason_evaluation import BestF1Callback, predict_dev, score_methods


def load_tokenizer(model_path=cfg.MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def main(run_name=None, baseline=False, prepare_only=False):
    run_name = run_name or (cfg.RUN_NAME + '_baseline' if baseline else cfg.RUN_NAME)
    reason_weight = 0.0 if baseline else cfg.REASON_WEIGHT
    set_seed(cfg.SEED)
    tokenizer = load_tokenizer()
    label_prompt, reason_prompt = cfg.build_prompt_templates(tokenizer)
    train_frame = load_train_frame()
    train_dataset = build_dataset(train_frame, tokenizer, include_reason=not baseline, label_prompt=label_prompt, reason_prompt=reason_prompt, max_length=cfg.MAX_LENGTH)
    if prepare_only:
        info = save_prepared(train_frame, train_dataset, cfg.PREPARED_DIR, label_prompt, reason_prompt)
        fields = ['train_rows', 'original_rows', 'reverse_N_rows', 'reason_rows',
                  'reason_original_rows', 'reason_reverse_N_rows', 'maximum_label_length', 'maximum_reason_length']
        print(json.dumps({key: info[key] for key in fields}, ensure_ascii=False, indent=2))
        print('只准备数据，没有加载Qwen主干或训练。')
        print('CSV：', cfg.PREPARED_DIR / 'train_C_with_reasons.csv')
        return

    model_dir = cfg.MODEL_ROOT / run_name / cfg.MODE
    results_dir = cfg.RESULTS_ROOT / run_name / cfg.MODE
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError('此RUN_NAME已有模型或结果，请修改RUN_NAME，保留旧实验。')
    results_dir.mkdir(parents=True)
    info = save_prepared(train_frame, train_dataset, results_dir / 'prepared', label_prompt, reason_prompt)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    dev_dataset = build_dataset(dev, tokenizer, include_reason=False, label_prompt=label_prompt, max_length=cfg.MAX_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_PATH, dtype='auto', local_files_only=True)
    lora_config = LoraConfig(
        r=cfg.LORA_R, lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
    )
    model.config.use_cache = False
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    train_args = SFTConfig(
        learning_rate=cfg.LEARNING_RATE, num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE, gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=cfg.EVAL_LOSS_BATCH_SIZE,
        bf16=cfg.BF16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        completion_only_loss=True, packing=False, padding_free=False,
        loss_type='chunked_nll', max_length=cfg.MAX_LENGTH,
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        seed=cfg.SEED, data_seed=cfg.SEED,
        output_dir=str(model_dir), eval_strategy='epoch', save_strategy='best',
        load_best_model_at_end=True, metric_for_best_model='eval_pair_f1',
        greater_is_better=True, save_total_limit=1, save_only_model=True,
        prediction_loss_only=True, logging_steps=20, report_to='none',
    )
    callback = BestF1Callback(dev, results_dir, label_prompt, batch_size=cfg.PREDICT_BATCH_SIZE)
    trainer = ReasonSFTTrainer(
        model=lora_model, args=train_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        processing_class=tokenizer, data_collator=ReasonCollator(tokenizer.pad_token_id),
        callbacks=[callback], reason_weight=reason_weight,
    )
    settings = {
        **info, 'run_name': run_name, 'mode': cfg.MODE, 'seed': cfg.SEED,
        'model_path': cfg.MODEL_PATH, 'reason_weight': reason_weight,
        'lora': lora_config.to_dict(), 'training': train_args.to_dict(),
        'selection': 'pair_aware', 'steps_per_epoch': ((len(train_dataset) + cfg.TRAIN_BATCH_SIZE - 1) // cfg.TRAIN_BATCH_SIZE + cfg.GRADIENT_ACCUMULATION_STEPS - 1) // cfg.GRADIENT_ACCUMULATION_STEPS,
        'objective': 'mean CE(label + EOS) + reason_weight * mean CE(reason + EOS); separate prompts',
        'inference': 'Instruct chat template; first-token four-label argmax; raw generation audited; no Dev reason',
        'dev_sha256': file_hash(cfg.DEV_PATH),
        'source_sha256': {str(path.relative_to(cfg.PROJECT_ROOT)): file_hash(path) for path in [
            cfg.EXPERIMENT_DIR / 'config.py', cfg.EXPERIMENT_DIR / 'train_lora_reason.py',
            *[(cfg.PROJECT_ROOT / 'experiments/qwen_lora/reason_sft' / name) for name in ['reason_data.py', 'reason_trainer.py', 'reason_evaluation.py']],
        ]},
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    for directory in [model_dir, results_dir]:
        (directory / 'run_config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('模型：Qwen2.5-3B-Instruct；数据：原C；r / alpha：', cfg.LORA_R, '/', cfg.LORA_ALPHA)
    print('标签行数：', len(train_frame), '；可用解释：', int(train_frame.use_reason.sum()))
    print('损失：标签SFT +', reason_weight, '* 文字解释SFT；没有Pair CE。')
    print('每轮标签批次：', settings['steps_per_epoch'], '；标签和解释均使用Instruct聊天模板。')
    print('模型目录：', model_dir)
    trainer.train()

    best_dir = model_dir / 'best_adapter'
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)
    frame = predict_dev(trainer.model, tokenizer, dev, label_prompt, cfg.PREDICT_BATCH_SIZE)
    summary = score_methods(frame, results_dir / 'best')
    summary.to_csv(results_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
    best_step = int(Path(trainer.state.best_model_checkpoint).name.rsplit('-', 1)[1])
    selected = next(row for row in callback.history if row['global_step'] == best_step and row['method'] == 'pair_aware')
    info = {'checkpoint': trainer.state.best_model_checkpoint, 'pair_f1': trainer.state.best_metric,
            'best_epoch': selected['epoch'], 'best_global_step': best_step,
            'same_checkpoint_metrics': summary.to_dict('records')}
    for directory in [model_dir, results_dir]:
        (directory / 'best_metrics.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print('最佳checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳pair-aware F1：', trainer.state.best_metric)
    print('最佳Adapter：', best_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name')
    parser.add_argument('--baseline', action='store_true', help='reason权重为0，同代码原C标签SFT对照。')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.baseline, args.prepare_only)
