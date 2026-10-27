"""重载BLOOP训练的最佳模型；预测不使用梯度EMA。"""
import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase
from experiments.deberta_manual_aug.train_relation_bloop import (
    RUN_NAME, MODE, MODEL_ROOT, DEV_PATH, GOLD_PATH, load_saved_model, file_hash, collate_rows,
)


MODEL_DIR = MODEL_ROOT / f'{RUN_NAME}_{MODE}'


def main(model_dir=MODEL_DIR):
    device = torch.device('cuda')
    model, tokenizer, config = load_saved_model(model_dir, device)
    assert file_hash(DEV_PATH) == config['dev_sha256']
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    loader = DataLoader(RowDataset(dev, tokenizer, config['max_length']),
                        batch_size=config['dev_batch_size'], shuffle=False,
                        collate_fn=partial(collate_rows, dynamic_padding=config['dynamic_padding']))
    output_dir = Path(__file__).resolve().parent / 'outputs' / config['run_name'] / 'reloaded/dev'
    evaluate_phase(model, loader, dev, device, output_dir, GOLD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
