"""右键运行：重新评估Lexical + 3290行实验保存的唯一最佳模型。"""
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

from experiments.deberta_lexical.data import build_feature_tensor, DiCoDataset
from experiments.deberta_lexical_aug import train_deberta_lexical_relation as current


def main(run_name=current.RUN_NAME):
    model_dir = current.MODEL_ROOT / run_name
    results_dir = current.HERE / 'outputs' / run_name / 'reevaluation'
    settings = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = current.build_model(
        tokenizer, settings['model_name'], settings['lexical_hidden_size'], settings['base_revision'],
    )
    model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True))
    device = torch.device('cuda')
    model.to(device).eval()
    dev = pd.read_csv(current.DEV_PATH, keep_default_na=False)
    features = build_feature_tensor(dev)
    loader = DataLoader(DiCoDataset(dev, tokenizer, features, settings['max_length']),
                        batch_size=settings['dev_batch_size'], shuffle=False)
    current.evaluate(model, loader, dev, device, results_dir)
    current.compare_with_reference(results_dir / 'predictions_with_scores.csv', results_dir / 'comparison_with_0p938')
    print('读取权重：', model_dir / 'best_model.pth')
    print('原选模指标：', settings['selection_metric'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=current.RUN_NAME)
    args = parser.parse_args()
    main(args.run_name)
