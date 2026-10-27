"""可选：重新加载最佳双向联合模型；训练结束已自动评估。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_bidirectional.data import BidirectionalDataset, collate_units
from experiments.deberta_bidirectional.model import load_saved_model
from experiments.deberta_bidirectional.evaluation import evaluate
from experiments.deberta_bidirectional.train_deberta_bidirectional import (
    RUN_NAME, MODEL_ROOT, DEV_PATH, GOLD_PATH, file_hash,
)


MODEL_DIR = MODEL_ROOT / RUN_NAME


def main(model_dir):
    device = torch.device('cuda')
    model, tokenizer, config = load_saved_model(model_dir, device)
    assert file_hash(DEV_PATH) == config['dev_sha256']
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    loader = DataLoader(
        BidirectionalDataset(dev, tokenizer, config['max_length']),
        batch_size=config['eval_unit_batch_size'], shuffle=False, collate_fn=collate_units,
    )
    output_dir = Path(__file__).resolve().parent / 'outputs' / config['run_name'] / 'reloaded/dev'
    print('恢复模型：', model_dir)
    evaluate(model, loader, dev, device, output_dir, GOLD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
