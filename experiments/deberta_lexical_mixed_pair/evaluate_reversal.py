"""重载原架构CE + 反向KL版本的完整最佳模型。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from experiments.deberta_lexical_mixed_pair import config_reversal as cfg
from experiments.deberta_lexical_mixed_pair.train_deberta_reversal import read_data, make_dev_loader
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, score_logits


def main():
    saved = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    best = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_config(DebertaV2Config.from_dict(saved['backbone_config']), dtype=torch.float32)
    model = DebertaDiCoModel(
        backbone, tokenizer, len(saved['feature_names']), saved['lexical_hidden_size'],
        saved['use_lexical_features'],
    )
    state = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device('cuda')
    model.to(device)
    _, dev = read_data()
    loader = make_dev_loader(dev, tokenizer, build_feature_tensor(dev), saved['max_length'], saved['dev_batch_size'])
    logits = predict_logits(model, loader, device)
    score_logits(logits, dev, cfg.RESULTS_DIR / 'best_reevaluation', gold_path=cfg.GOLD_PATH)
    historical = pd.read_csv(cfg.RESULTS_DIR / f"epoch_{best['best_epoch']}" / 'predictions_with_scores.csv')
    torch.testing.assert_close(logits, torch.tensor(historical[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32))
    print('重载后的逐条分数与保存的最佳轮次一致。')


if __name__ == '__main__':
    main()
