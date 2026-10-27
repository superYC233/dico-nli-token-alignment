"""重载本目录训练的最佳模型，仍使用原来的independent及pair-aware预测。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.deberta_pair_hinge import config as cfg
from experiments.deberta_pair_hinge.train_deberta_pair_hinge import build_model, file_hash
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase


def main(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    assert file_hash(cfg.DEV_PATH) == config['data']['dev_sha256']
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer, device, model_name=config['model_name'])
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    loader = DataLoader(
        RowDataset(dev, tokenizer, config['max_length']),
        batch_size=config['dev_batch_size'], shuffle=False,
        generator=torch.Generator().manual_seed(config['seed'] + 1),
    )
    output_dir = cfg.HERE / 'outputs' / config['run_name'] / 'reloaded/dev'
    evaluate_phase(model, loader, dev, device, output_dir, cfg.GOLD_PATH)
    print('重载后的结果：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-loss', choices=['ce3', 'hinge16'], default=cfg.PAIR_LOSS)
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir or cfg.MODEL_ROOT / cfg.get_run_name(args.run_name, args.pair_loss)
    main(model_dir)
