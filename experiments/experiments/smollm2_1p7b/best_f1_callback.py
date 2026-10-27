from pathlib import Path

from transformers import TrainerCallback

import smol_config as cfg
from smol_utils import predict_with_scores, pair_aware_decode, score_methods


class BestF1Callback(TrainerCallback):
    def __init__(self, dev_dataset, tokenizer, batch_size=cfg.PREDICT_BATCH_SIZE):
        self.dev_dataset = dev_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None:
            raise ValueError('Trainer 没有传入评估指标。')
        frame = predict_with_scores(model, self.tokenizer, self.dev_dataset, self.batch_size)
        frame = pair_aware_decode(frame)
        epoch = int(round(state.epoch or 0))
        score_dir = Path(args.output_dir) / 'dev_scores' / f'epoch_{epoch}'
        reports = score_methods(frame, score_dir)
        metrics['eval_pair_f1'] = reports['pair_aware'].weighted_f1
        print(f'第 {epoch} 轮 pair-aware F1：{metrics["eval_pair_f1"]:.6f}')
        return control
