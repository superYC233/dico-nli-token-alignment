"""Qwen C+r16标准SFT + 200条范围增强；右键运行训练。"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from experiments.qwen_scope_aug.data_utils import (
    PROMPT_TEMPLATE, prepare_data, get_sft_datasets, check_supervision,
)
from experiments.qwen_scope_aug.evaluate_lora_aug import evaluate_model


RUN_NAME = 'run02_c_scope'  # 修复并列最高分校验后重新运行；保留run01结果。
ADD_SCOPE_AUGMENTATION = True
SEED = 42
MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
OUTPUT_DIR = Path('D:/model/SEAMEVAL/qwen_scope_aug') / RUN_NAME
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME


class BestF1Callback(TrainerCallback):
    def __init__(self, tokenizer, dev, prompt_template, results_dir):
        self.tokenizer = tokenizer
        self.dev = dev
        self.prompt_template = prompt_template
        self.results_dir = Path(results_dir)
        self.history = []

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # 预测不改变后续训练的随机序列、padding方向或模型状态。
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        padding, was_training = self.tokenizer.padding_side, model.training
        try:
            epoch = int(round(state.epoch or 0))
            summary = evaluate_model(model, self.tokenizer, self.dev, self.prompt_template,
                                     self.results_dir / f'epoch_{epoch}')
            for row in summary.to_dict('records'):
                row['epoch'] = epoch
                self.history.append(row)
            pd.DataFrame(self.history).to_csv(
                self.results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig',
            )
            scores = summary.set_index('method').weighted_f1
            metrics['eval_independent_f1'] = float(scores['independent'])
            metrics['eval_pair_f1'] = float(scores['pair_aware'])
            print(f'第{epoch}轮：independent={scores["independent"]:.6f}，pair-aware={scores["pair_aware"]:.6f}')
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            self.tokenizer.padding_side = padding
            model.train(was_training)
        return control


def main():
    if OUTPUT_DIR.exists() or (RESULTS_DIR / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已使用，请换新名字，保留旧模型和结果。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练请使用你的 CUDA 环境。')
    train, dev_dataset, dev, data_info = prepare_data(RESULTS_DIR, ADD_SCOPE_AUGMENTATION)
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype='auto', local_files_only=True)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
    )
    lora_model = get_peft_model(model, lora_config)
    train_args = SFTConfig(
        learning_rate=1e-4, num_train_epochs=5,
        per_device_train_batch_size=16, gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        bf16=True, completion_only_loss=True, packing=False,
        seed=SEED, data_seed=SEED,
        output_dir=str(OUTPUT_DIR), eval_strategy='epoch', save_strategy='best',
        load_best_model_at_end=True, metric_for_best_model='eval_pair_f1', greater_is_better=True,
        save_total_limit=1, save_only_model=True,
        logging_steps=20, report_to='none',
    )
    dataset = get_sft_datasets(train, dev_dataset, PROMPT_TEMPLATE)
    tokenizer.padding_side = 'right'
    callback = BestF1Callback(tokenizer, dev, PROMPT_TEMPLATE, RESULTS_DIR)
    trainer = SFTTrainer(
        model=lora_model, args=train_args,
        train_dataset=dataset['train'], eval_dataset=dataset['dev'],
        processing_class=tokenizer, callbacks=[callback],
    )
    check_supervision(trainer.train_dataset, dataset['train'], tokenizer)
    config = {
        'run_name': RUN_NAME, 'model_path': MODEL_PATH, 'seed': SEED,
        'data': data_info, 'prompt_template': PROMPT_TEMPLATE,
        'lora': lora_config.to_dict(), 'training': train_args.to_dict(),
        'inference_batch_size': 16, 'selection': 'best Dev pair-aware W-F1',
        'training_loss': 'standard full-vocabulary completion-only SFT, including answer and EOS',
        'note': 'C built before adding scope rows. Synthetic labels are not independently adjudicated.',
    }
    (RESULTS_DIR / 'run_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
    )
    print('方案：原C + r16 + 旧短Prompt + 标准SFT + 范围增强')
    print('每轮训练步数：', math.ceil(len(train) / 16))
    print('模型目录：', OUTPUT_DIR)
    print('结果目录：', RESULTS_DIR)
    trainer.train()
    best_dir = OUTPUT_DIR / 'best_adapter'
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)
    (best_dir / 'experiment_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
    )
    best = {'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_pair_f1': trainer.state.best_metric, 'best_adapter': str(best_dir)}
    (RESULTS_DIR / 'best_metrics.json').write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 Adapter：', best_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='仅准备数据，不加载模型、不训练')
    args = parser.parse_args()
    if args.prepare_only:
        train, dev, _, info = prepare_data(RESULTS_DIR, ADD_SCOPE_AUGMENTATION)
        print('每轮步数 / 5轮总步数：', math.ceil(len(train) / 16), '/', math.ceil(len(train) / 16) * 5)
        print('数据准备完成，未加载模型、未启动训练。')
    else:
        main()
