"""加载OOF加权训练的最终单模型，评估Dev，不训练。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config_oof as cfg
from experiments.deberta_lexical_mixed_pair.data import read_data
from experiments.deberta_lexical_mixed_pair.train_deberta_oof_hard import make_dev_loader
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, score_logits


def main():
    saved = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'], use_lexical_features=saved['use_lexical_features'],
    )
    state = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    info = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    print('最佳轮次：', info['best_epoch'], '；选模依据：', info['selection'])
    device = torch.device('cuda')
    model.to(device)
    _, dev = read_data()
    loader = make_dev_loader(
        dev, tokenizer, build_feature_tensor(dev), saved['max_length'], saved['dev_batch_size'],
    )
    score_logits(
        predict_logits(model, loader, device), dev,
        cfg.RESULTS_DIR / 'best_reevaluation', gold_path=cfg.GOLD_PATH,
    )


if __name__ == '__main__':
    main()
