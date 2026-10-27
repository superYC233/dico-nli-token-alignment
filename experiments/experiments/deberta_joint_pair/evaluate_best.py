"""右键运行：重新加载本实验保存的最佳模型，报告两种解码成绩。"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from transformers import AutoTokenizer

from experiments.deberta_joint_pair import config as cfg
from experiments.deberta_joint_pair.train_deberta_joint_pair import build_model, make_dev_loader, evaluate


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer')
    model = build_model(tokenizer)
    weights = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device)

    info = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    print('最佳轮次：', info['best_epoch'], '；选模依据：', info['selection_method'])
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    summary = evaluate(model, make_dev_loader(dev, tokenizer), dev, device, cfg.RESULTS_DIR / 'best_evaluation')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
