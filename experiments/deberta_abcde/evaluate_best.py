"""可选：重新加载某组最佳权重，只做预测，不训练。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_abcde import config as cfg
from experiments.deberta_abcde.data_variants import RowDataset, validate_groups
from experiments.deberta_abcde.model import DebertaDiCoModel
from experiments.deberta_abcde.evaluation import predict_logits, score_logits


MODE = 'C'


def main(mode):
    model_dir, results_dir = cfg.get_output_dirs(mode)
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    if hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest() != saved['dev_sha256']:
        raise ValueError('Dev 文件和训练时不同，请检查。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA。')
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer)
    state = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(dev)
    loader = DataLoader(
        RowDataset(dev, tokenizer, saved['max_length']),
        batch_size=saved['dev_batch_size'], shuffle=False,
    )
    logits = predict_logits(model, loader, device)
    score_logits(logits, dev, results_dir / 'best_reevaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=list('ABCDE'), default=MODE)
    main(parser.parse_args().mode)
