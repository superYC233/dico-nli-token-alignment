"""可选：重新加载Row CE架构对照的最佳模型；训练结束已经自动评估。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase
from experiments.deberta_manual_aug.relation_row_model import load_saved_model


MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug/run23_alignment_row_seed7')
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'


def main(model_dir):
    device = torch.device('cuda')
    model, tokenizer, config = load_saved_model(model_dir, device)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    loader = DataLoader(
        RowDataset(dev, tokenizer, config['max_length']),
        batch_size=config['dev_batch_size'], shuffle=False,
    )
    output_dir = Path(__file__).resolve().parent / 'outputs' / config['run_name'] / 'reloaded/dev'
    print('恢复模型：', model_dir, '；架构：', config['head_type'])
    evaluate_phase(model, loader, dev, device, output_dir, GOLD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
