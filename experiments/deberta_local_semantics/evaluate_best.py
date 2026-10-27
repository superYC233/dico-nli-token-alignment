"""右键运行：加载本实验唯一最佳权重，输出独立和原pair-aware评分。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_local_semantics import config as cfg
from experiments.deberta_local_semantics.model import load_saved_model
from experiments.deberta_local_semantics.data import LocalRowDataset, diagnostic_frame
from experiments.deberta_local_semantics.evaluation import evaluate, evaluate_local, compare_with_reference


def main(run_name=cfg.RUN_NAME):
    model_dir, results_dir = cfg.get_output_dirs(run_name)
    device = torch.device('cuda')
    model, tokenizer, settings = load_saved_model(model_dir, device)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    loader = DataLoader(RowDataset(dev, tokenizer, settings['max_length']),
                        batch_size=settings['dev_batch_size'], shuffle=False)
    output_dir = results_dir / 'reevaluation'
    evaluate(model, loader, dev, device, output_dir)
    compare_with_reference(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_0p938')
    diagnostic = pd.read_csv(model_dir / 'local_diagnostic_annotations.csv', keep_default_na=False)
    local_rows = LocalRowDataset(diagnostic_frame(diagnostic), tokenizer, diagnostic, settings['max_length'])
    evaluate_local(model, DataLoader(local_rows, batch_size=settings['dev_batch_size']), diagnostic,
                   device, output_dir / 'local_diagnostic')
    print('本次读取模型：', model_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    args = parser.parse_args()
    main(args.run_name)
