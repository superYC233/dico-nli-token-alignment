"""右键运行：恢复NLI底座+最佳LoRA+任务头，评估同一模型的两种读出。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase
from experiments.deberta_manual_aug.relation_lora_model import load_saved_model


MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug/run21_lora_r16_seed7_dropout01_linear_fast')
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'


def main():
    device = torch.device('cuda')
    model, tokenizer, config = load_saved_model(MODEL_DIR, device)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    loader = DataLoader(
        RowDataset(dev, tokenizer, config['max_length']),
        batch_size=config['dev_batch_size'], shuffle=False,
    )
    output_dir = Path(__file__).resolve().parent / 'outputs' / config['run_name'] / 'reloaded/dev'
    print('恢复模型：', MODEL_DIR)
    print('结果目录：', output_dir)
    evaluate_phase(model, loader, dev, device, output_dir, GOLD_PATH)


if __name__ == '__main__':
    main()
