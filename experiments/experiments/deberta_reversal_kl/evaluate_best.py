"""从保存的原生四分类权重重载评估，不训练。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config

from experiments.deberta_reversal_kl import config as cfg
from experiments.deberta_reversal_kl.data import read_data, make_dev_loader
from experiments.deberta_reversal_kl.evaluation import predict_logits, score_logits


def main():
    saved = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    best = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(
        DebertaV2Config.from_dict(saved['model_config']), dtype=torch.float32,
    )
    state = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device('cuda')
    model.to(device)
    _, dev = read_data()
    loader = make_dev_loader(dev, tokenizer, saved['max_length'], saved['dev_batch_size'])
    logits = predict_logits(model, loader, device)
    score_logits(logits, dev, cfg.RESULTS_DIR / 'best_reevaluation', gold_path=cfg.GOLD_PATH)
    historical = pd.read_csv(cfg.RESULTS_DIR / f"step_{best['best_step']}" / 'predictions_with_scores.csv')
    columns = [f'logit_{label}' for label in cfg.LABELS]
    torch.testing.assert_close(logits, torch.tensor(historical[columns].values, dtype=torch.float32))
    print('重载分数与最佳保存step一致。')


if __name__ == '__main__':
    main()
