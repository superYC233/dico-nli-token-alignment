
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

from data_utils import tokenizer
from sixclass_data_utils import (
    get_sixclass_train_dataset,
    get_sixclass_token_ids,
    evaluate_sixclass_with_scores
)
from evaluate_lora_pair import pair_aware_decode
from baseline import score_predictions


class BestSixClassF1Callback(TrainerCallback):
    def __init__(self):
        self.best_f1 = -1.0
        self.best_epoch = None

    def on_epoch_end(
        self, args, state, control, model=None, **kwargs
    ):
        # 1. 当前轮模型预测 Dev，六类概率合并为四类。
        records = evaluate_sixclass_with_scores(
            model,
            batch_size=16
        )

        # 2. 复用已有 pair-aware。
        df = pair_aware_decode(pd.DataFrame(records))

        epoch = int(round(state.epoch or 0))

        score_dir = (
            Path(args.output_dir)
            / 'dev_scores'
            / f'epoch_{epoch}'
        )
        score_dir.mkdir(parents=True, exist_ok=True)

        # 每轮保存预测文件，方便比较。
        df.to_csv(
            score_dir / 'predictions_with_scores.csv',
            index=False,
            encoding='utf-8-sig'
        )

        # 3. 使用官方四分类评分。
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
            raise ValueError('无法计算四分类 F1')

        print(
            f'第 {epoch} 轮 pair-aware F1：'
            f'{report.weighted_f1:.6f}'
        )
        print(
            'N-F1：',
            report.classification.per_label['NEGATIVE_OTHER'].f1
        )

        # 4. 只有刷新纪录，才覆盖 best_adapter。
        if report.weighted_f1 > self.best_f1:
            self.best_f1 = report.weighted_f1
            self.best_epoch = epoch

            best_dir = Path(args.output_dir) / 'best_adapter'

            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            print('已更新最佳 Adapter：', best_dir)

        return control


if __name__ == '__main__':
    # 兼容原有评分函数中的相对路径。
    os.chdir(script_dir)

    mode = 'C'
    seed = 42

    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'

    output_dir = (
        f'D:/LLM/train_model/DiCo-NLI/'
        f'qwen3_0p6b_sixclass/run01/{mode}'
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

        bf16=True,
        completion_only_loss=True,
        packing=False,

        seed=seed,
        data_seed=seed,
        output_dir=output_dir,

        # 本版由回调直接评估四类 F1、保存最佳 Adapter。
        # 因此关闭 Trainer 自带的评估和自动保存。
        eval_strategy='no',
        save_strategy='no',
        load_best_model_at_end=False,

        logging_steps=20,
        report_to='none'
    )

    train_dataset = get_sixclass_train_dataset(mode)

    print('六个标签的 token ID：', get_sixclass_token_ids())

    tokenizer.padding_side = 'right'

    best_callback = BestSixClassF1Callback()

    trainer = SFTTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[best_callback]
    )

    trainer.train()

    # 最佳 Adapter 已经由回调保存。
    # 此处不要再用最后一轮覆盖 best_adapter。
    print('最佳轮次：', best_callback.best_epoch)
    print('最佳 pair-aware F1：', best_callback.best_f1)
    print('最佳 Adapter：', f'{output_dir}/best_adapter')