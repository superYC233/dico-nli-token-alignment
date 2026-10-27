import random
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.prediction import predict_with_scores
from experiments.qwen3_1p7b_lexical.data_utils import pair_aware_decode, score_methods


class BestF1Callback(TrainerCallback):
    def __init__(self, dev_dataset, tokenizer, batch_size=cfg.PREDICT_BATCH_SIZE):
        self.dev_dataset = dev_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None:
            raise ValueError('Trainer 没有传入评估指标。')
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        device = next(model.parameters()).device
        cuda_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        try:
            frame = predict_with_scores(model, self.tokenizer, self.dev_dataset, self.batch_size)
            frame = pair_aware_decode(frame)
            epoch = int(round(state.epoch or 0))
            directory = Path(args.output_dir) / 'dev_scores' / f'epoch_{epoch}'
            reports = score_methods(frame, directory)
            metrics['eval_pair_f1'] = reports['pair_aware'].weighted_f1
            print(f'第 {epoch} 轮 pair-aware F1：{metrics["eval_pair_f1"]:.6f}')
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return control
