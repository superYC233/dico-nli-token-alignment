"""重新加载原始base初始化实验的最佳完整模型，预测Dev；不训练。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from experiments.deberta_base_lexical import config as cfg
from experiments.deberta_base_lexical.train_deberta_base_lexical import (
    read_data, load_tokenizer, build_model, file_hash,
)
from experiments.deberta_lexical.data import build_feature_tensor, DiCoDataset
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES
from experiments.deberta_lexical.evaluation import predict_logits, score_logits


def main(model_dir):
    model_dir = Path(model_dir)
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    assert file_hash(cfg.DEV_PATH) == saved['dev_sha256']
    assert file_hash(ROOT / 'experiments/deberta_lexical/lexical_features.py') == saved['feature_source_sha256']
    assert saved['feature_names'] == FEATURE_NAMES
    device = torch.device('cuda')
    tokenizer = load_tokenizer(model_dir / 'tokenizer')
    model = build_model(
        tokenizer, device, pretrained_path=saved['pretrained_path'],
        lexical_hidden=saved['lexical_hidden_size'], use_lexical_features=saved['use_lexical_features'],
    )
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    _, dev = read_data()
    loader = DataLoader(
        DiCoDataset(dev, tokenizer, build_feature_tensor(dev), saved['max_length']),
        batch_size=saved['dev_batch_size'], shuffle=False,
    )
    output_dir = cfg.EXPERIMENT_DIR / 'outputs' / model_dir.parent.name / model_dir.name / 'best_reevaluation'
    logits = predict_logits(model, loader, device)
    score_logits(logits, dev, output_dir, gold_path=cfg.GOLD_PATH)
    print('重新预测完成：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=cfg.RUN_NAME)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir or cfg.get_output_dirs(args.run_name)[0]
    main(model_dir)
