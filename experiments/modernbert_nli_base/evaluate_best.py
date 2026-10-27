"""加载本实验保存的最佳完整权重，重新预测 Dev，不训练。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.modernbert_nli_base import config as cfg
from experiments.modernbert_nli_base.data import read_data, build_feature_tensor, DiCoDataset
from experiments.modernbert_nli_base.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.modernbert_nli_base.model import ModernBertDiCoModel
from experiments.modernbert_nli_base.evaluation import predict_logits, score_logits, compare_with_original


def main(use_lexical_features=cfg.USE_LEXICAL_FEATURES):
    model_dir, results_dir = cfg.get_output_dirs(use_lexical_features)
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    if saved['feature_names'] != FEATURE_NAMES or saved['feature_version'] != FEATURE_VERSION:
        raise ValueError('词汇特征定义或顺序与训练时不一致。')
    feature_file = Path(__file__).resolve().parent / 'lexical_features.py'
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != saved['feature_source_sha256']:
        raise ValueError('lexical_features.py 已改变，请先恢复训练时的版本。')
    if hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest() != saved['dev_sha256']:
        raise ValueError('Dev 文件与训练时不同。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA。')
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_path'], local_files_only=True,
        dtype=torch.float32, use_safetensors=True, attn_implementation='sdpa',
    )
    model = ModernBertDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'],
        use_lexical_features=saved['use_lexical_features'],
    )
    state = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    _, dev = read_data()
    features = build_feature_tensor(dev)
    loader = DataLoader(
        DiCoDataset(dev, tokenizer, features, saved['max_length']),
        batch_size=saved['dev_batch_size'], shuffle=False,
    )
    logits = predict_logits(model, loader, device)
    output_dir = results_dir / 'best_reevaluation'
    score_logits(logits, dev, output_dir)
    compare_with_original(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_original')
    print('重新预测完成：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--lexical', action='store_true')
    args = parser.parse_args()
    main(use_lexical_features=(cfg.USE_LEXICAL_FEATURES or args.lexical) and not args.baseline)
