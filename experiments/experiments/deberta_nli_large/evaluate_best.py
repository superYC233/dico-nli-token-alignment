"""只加载本次最佳权重评估 Dev，不训练。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.deberta_nli_large import config as cfg
from experiments.deberta_nli_large.data import read_data
from experiments.deberta_nli_large.train_deberta_large import make_dev_loader
from experiments.deberta_nli_large.data import build_feature_tensor
from experiments.deberta_nli_large.model import DebertaDiCoModel
from experiments.deberta_nli_large.evaluation import predict_logits, score_logits


def main():
    saved = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_path'], local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'], use_lexical_features=saved['use_lexical_features'],
    )
    state = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    info = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    print('加载最佳第', info['best_epoch'], '轮，选模依据：', info['selection'])
    del state
    device = torch.device('cuda')
    model.to(device)
    _, dev = read_data()
    loader = make_dev_loader(
        dev, tokenizer, build_feature_tensor(dev), saved['max_length'], saved['dev_batch_size'],
    )
    score_logits(
        predict_logits(model, loader, device, bf16=saved['bf16']), dev,
        cfg.RESULTS_DIR / 'best_reevaluation', gold_path=cfg.GOLD_PATH,
    )


if __name__ == '__main__':
    main()
