from pathlib import Path

import pandas as pd
from transformers import TrainerCallback

from evaluate_lora_pair import pair_aware_decode
from compare_decoding import evaluate_with_scores
from baseline import score_predictions


class BestF1Callback(TrainerCallback):
    def on_evaluate(
        self, args, state, control,
        model=None, processing_class=None, metrics=None, **kwargs
    ):
        # 记录原来的补齐方式和模型状态
        old_padding_side = processing_class.padding_side
        was_training = model.training

        model.eval()

        try:
            # 1. 使用当前轮模型预测 Dev
            records = evaluate_with_scores(
                model,
                batch_size=16,
                split='dev'
            )

            # 2. 执行 pair-aware
            df = pair_aware_decode(pd.DataFrame(records))

            # 3. 准备官方评分所需的记录
            score_records = []

            for _, row in df.iterrows():
                score_records.append({
                    'instance_id': row['instance_id'],
                    'prediction': row['pair_prediction']
                })

            # 每轮只保留小型评分文件
            epoch = int(round(state.epoch or 0))
            score_dir = (
                Path(args.output_dir)
                / 'dev_scores'
                / f'epoch_{epoch}'
            )
            score_dir.mkdir(parents=True, exist_ok=True)

            report = score_predictions(
                score_records,
                str(score_dir)
            )

            if report is None:
                raise ValueError('预测中存在非法标签，无法计算 F1')

            # 4. 把 F1 交给 Trainer，用于选择最佳模型
            metrics['eval_pair_f1'] = report.weighted_f1

            print(
                f'第 {epoch} 轮 pair-aware F1：'
                f'{report.weighted_f1:.6f}'
            )

        finally:
            # 预测会改为左补齐，结束后恢复，供后续训练使用
            processing_class.padding_side = old_padding_side
            model.train(was_training)

        return control