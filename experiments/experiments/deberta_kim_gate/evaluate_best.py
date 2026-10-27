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
from experiments.deberta_kim_gate import config as cfg
from experiments.deberta_kim_gate.data import read_data, build_feature_tensor, DiCoDataset
from experiments.deberta_kim_gate.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_kim_gate.relation_features import RELATION_NAMES, RELATION_VERSION
from experiments.deberta_kim_gate.wordnet_resource import resource_info
from experiments.deberta_kim_gate.model import DebertaDiCoModel
from experiments.deberta_kim_gate.evaluation import (
    predict_logits, score_logits, compare_with_original, compare_with_lexical,
    save_gate_statistics, compare_with_kim,
)


def main(use_kim_features=cfg.USE_KIM_FEATURES, gate_mode=cfg.GATE_MODE):
    model_dir, results_dir = cfg.get_output_dirs(use_kim_features, gate_mode)
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    if saved['feature_names'] != FEATURE_NAMES or saved['feature_version'] != FEATURE_VERSION:
        raise ValueError('词汇特征定义或顺序与训练时不一致。')
    folder = Path(__file__).resolve().parent
    for name, digest in saved['feature_source_sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} 已改变，请先恢复训练时的版本。')
    if hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest() != saved['dev_sha256']:
        raise ValueError('Dev 文件与训练时不同。')
    if saved['relation_names'] != RELATION_NAMES or saved['relation_version'] != RELATION_VERSION:
        raise ValueError('局部关系定义与训练时不同。')
    if saved['wordnet_depth'] != cfg.WORDNET_MAX_HYPERNYM_DEPTH:
        raise ValueError('WordNet 查询深度与训练时不同。')
    if saved['use_kim_features'] and saved['wordnet_resource'] != resource_info():
        raise ValueError('WordNet 资源与训练时不同。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA。')
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'],
        use_kim_features=saved['use_kim_features'],
        gate_mode=saved['gate_mode'],
    )
    state = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    _, dev = read_data()
    features = build_feature_tensor(dev)
    loader = DataLoader(
        DiCoDataset(dev, tokenizer, features, saved['max_length'], saved['use_kim_features']),
        batch_size=saved['dev_batch_size'], shuffle=False,
    )
    logits = predict_logits(model, loader, device)
    output_dir = results_dir / 'best_reevaluation'
    score_logits(logits, dev, output_dir)
    save_gate_statistics(model, output_dir)
    compare_with_original(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_original')
    compare_with_lexical(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_lexical')
    compare_with_kim(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_kim')
    print('重新预测完成：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexical-only', action='store_true')
    parser.add_argument('--gate-mode', choices=['learned', 'fixed', 'open'], default=cfg.GATE_MODE)
    args = parser.parse_args()
    main(use_kim_features=cfg.USE_KIM_FEATURES and not args.lexical_only, gate_mode=args.gate_mode)
