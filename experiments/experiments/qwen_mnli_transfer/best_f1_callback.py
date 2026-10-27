import random
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from experiments.qwen_mnli_transfer import config as cfg
from experiments.generative_nli import predict_with_scores, pair_aware_decode, score_methods


class BestF1Callback(TrainerCallback):
    def __init__(self, dev_dataset, tokenizer):
        self.dev_dataset = dev_dataset
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None:
            raise ValueError('Trainer 没有传入评估指标。')
        # 评估用的随机数不应改变之后的 dropout / shuffle。
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        old_padding = self.tokenizer.padding_side
        was_training = model.training
        try:
            frame = predict_with_scores(model, self.tokenizer, self.dev_dataset, cfg.PREDICT_BATCH_SIZE)
            frame = pair_aware_decode(frame)
            epoch = int(round(state.epoch or 0))
            score_dir = Path(args.output_dir) / 'dev_scores' / f'epoch_{epoch}'
            reports = score_methods(frame, score_dir, cfg.GOLD_PATH)
            metrics['eval_pair_f1'] = reports['pair_aware'].weighted_f1
            print(f'第 {epoch} 轮 pair-aware F1：{metrics["eval_pair_f1"]:.6f}')
        finally:
            self.tokenizer.padding_side = old_padding
            model.train(was_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return control
