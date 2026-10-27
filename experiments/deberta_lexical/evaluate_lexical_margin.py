"""重载margin保护实验的最佳完整模型；预测不需要保护参照或乘子。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_lexical import train_deberta_lexical_margin as cfg


MODEL_DIR = cfg.MODEL_ROOT / cfg.RUN_NAME


def main(model_dir=MODEL_DIR):
    device = torch.device('cuda')
    model, tokenizer, config = cfg.load_saved_model(model_dir, device)
    assert cfg.file_hash(cfg.DEV_PATH) == config['dev_sha256']
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    features = cfg.build_feature_tensor(dev)
    loader = DataLoader(cfg.DiCoDataset(dev, tokenizer, features, config['max_length']),
                        batch_size=config['dev_batch_size'], shuffle=False)
    cfg.score_predictions(cfg.predict_preserving_rng(model, loader, device), dev,
                          cfg.HERE / 'outputs' / config['run_name'] / 'reloaded/dev', cfg.GOLD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
