"""重载参数锚定实验的最佳完整Lexical模型；推理不需要锚点。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_lexical.train_deberta_lexical_anchor import (
    RUN_NAME, MODEL_ROOT, DEV_PATH, GOLD_PATH, HERE, load_saved_model,
    file_hash, build_feature_tensor, DiCoDataset, evaluate_phase,
)


MODEL_DIR = MODEL_ROOT / RUN_NAME


def main(model_dir=MODEL_DIR):
    device = torch.device('cuda')
    model, tokenizer, config = load_saved_model(model_dir, device)
    assert file_hash(DEV_PATH) == config['dev_sha256']
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    features = build_feature_tensor(dev)
    loader = DataLoader(DiCoDataset(dev, tokenizer, features, config['max_length']),
                        batch_size=config['dev_batch_size'], shuffle=False)
    evaluate_phase(model, loader, dev, device, HERE / 'outputs' / config['run_name'] / 'reloaded/dev', GOLD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
