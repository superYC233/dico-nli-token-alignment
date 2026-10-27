
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import os
import sys
from pathlib import Path

import pandas as pd
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

script_dir = QWEN_DIR
sys.path.insert(0, str(script_dir.parent.parent))

from data_utils import load_data, tokenizer
from data_variants import build_train_dataset
from binary_data_utils import (
    get_binary_datasets,
    evaluate_binary_with_scores
)
from evaluate_lora_pair import pair_aware_decode
from baseline import score_predictions


class BestBinaryF1Callback(TrainerCallback):
    def on_evaluate(
        self, args, state, control,
        model=None, processing_class=None, metrics=None, **kwargs
    ):
        old_padding_side = tokenizer.padding_side
        was_training = model.training
        model.eval()

        try:
            records = evaluate_binary_with_scores(
                model,
                batch_size=16,
                split='dev'
            )

            df = pair_aware_decode(pd.DataFrame(records))

            epoch = int(round(state.epoch or 0))
            score_dir = (
                Path(args.output_dir)
                / 'dev_scores'
                / f'epoch_{epoch}'
            )
            score_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(
                score_dir / 'predictions_with_scores.csv',
                index=False,
                encoding='utf-8-sig'
            )

            score_records = []

            for _, row in df.iterrows():
                score_records.append({
                    'instance_id': row['instance_id'],
                    'prediction': row['pair_prediction']
                })

            report = score_predictions(
                score_records,
                str(score_dir)
            )

            if report is None:
                raise ValueError('无法计算二元模型还原后的四分类 F1')

            metrics['eval_pair_f1'] = report.weighted_f1

            print(
                f'第 {epoch} 轮 pair-aware F1：'
                f'{report.weighted_f1:.6f}'
            )

            print(
                'N-F1：',
                report.classification.per_label['NEGATIVE_OTHER'].f1
            )

        finally:
            # 恢复训练时的补齐方式和模型状态。
            tokenizer.padding_side = old_padding_side
            model.train(was_training)

        return control


if __name__ == '__main__':
    # 让原有的数据路径和评分路径正常工作。
    os.chdir(script_dir)

    mode = 'C'
    seed = 42

    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'

    output_dir = (
        f'D:/LLM/train_model/DiCo-NLI/'
        f'qwen3_0p6b_binary/run01/{mode}'
    )

    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto'
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
        bias='none',
        task_type='CAUSAL_LM'
    )

    lora_model = get_peft_model(model, lora_config)

    train_args = SFTConfig(
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,

        bf16=True,
        completion_only_loss=True,
        packing=False,

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

        logging_steps=20,
        report_to='none'
    )

    dataset = load_data()

    # 继续使用同一份 C 数据。
    dataset['train'] = build_train_dataset(
        dataset['train'],
        mode
    )

    print('训练版本：', mode)
    print('训练样本数：', len(dataset['train']))
    print('Dev 样本数：', len(dataset['dev']))

    # 将四类标签转换成二元答案。
    binary_dataset = get_binary_datasets(dataset)

    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=binary_dataset['train'],
        eval_dataset=binary_dataset['dev'],
        processing_class=tokenizer,
        callbacks=[BestBinaryF1Callback()]
    )

    trainer.train()

    # 此时内存中已经加载了最佳模型。
    best_dir = f'{output_dir}/best_adapter'

    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 Adapter 已导出：', best_dir)