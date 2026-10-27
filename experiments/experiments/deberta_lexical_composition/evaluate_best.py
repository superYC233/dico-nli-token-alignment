"""右键运行：加载本实验最佳完整权重，重新预测Dev，不训练。"""
import argparse
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
from experiments.deberta_lexical_composition import config as cfg
from experiments.deberta_lexical_composition.data import read_data, build_feature_tensor, DiCoDataset
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_lexical_composition.model import DebertaDiCoModel
from experiments.deberta_lexical_composition.evaluation import predict_logits, score_logits, compare_with_lexical
from experiments.deberta_lexical_composition.train_deberta_composition import file_hash


def main(use_composition=cfg.USE_COMPOSITION):
    model_dir, results_dir = cfg.get_output_dirs(use_composition)
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    feature_file = PROJECT_ROOT / 'experiments/deberta_lexical/lexical_features.py'
    if (saved['feature_names'] != FEATURE_NAMES or saved['feature_version'] != FEATURE_VERSION
            or saved['feature_source_sha256'] != file_hash(feature_file)):
        raise ValueError('词汇特征定义与训练时不同。')
    if file_hash(cfg.DEV_PATH) != saved['dev_sha256']:
        raise ValueError('Dev文件与训练时不同。')
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'],
        use_composition=saved['use_composition'],
        composition_layers=saved['composition_layers'],
        composition_kernel_size=saved['composition_kernel_size'],
    )
    state = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    _, dev = read_data()
    loader = DataLoader(
        DiCoDataset(dev, tokenizer, build_feature_tensor(dev), saved['max_length']),
        batch_size=saved['dev_batch_size'], shuffle=False,
    )
    output_dir = results_dir / 'best_reevaluation'
    logits = predict_logits(model, loader, device)
    score_logits(logits, dev, output_dir)
    compare_with_lexical(output_dir / 'predictions_with_scores.csv', output_dir / 'comparison_with_lexical')
    print('重新预测完成：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-composition', action='store_true')
    args = parser.parse_args()
    main(use_composition=cfg.USE_COMPOSITION and not args.no_composition)
