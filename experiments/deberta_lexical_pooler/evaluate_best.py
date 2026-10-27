"""加载包含pooler的完整最佳权重，重新评估Dev；不训练。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from experiments.deberta_lexical_pooler import config as cfg
from experiments.deberta_lexical_pooler.model import DebertaPoolerModel
from experiments.deberta_lexical_pooler.train_deberta_pooler import read_data, make_dev_loader
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, score_logits


def main():
    saved = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_config(
        DebertaV2Config.from_dict(saved['backbone_config']), dtype=torch.float32,
    )
    model = DebertaPoolerModel(
        backbone, tokenizer, len(saved['feature_names']), saved['lexical_hidden_size'],
        saved['use_lexical_features'], saved['use_nli_pooler'],
    )
    state = torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device('cuda')
    model.to(device)
    _, dev = read_data()
    loader = make_dev_loader(
        dev, tokenizer, build_feature_tensor(dev), saved['max_length'], saved['dev_batch_size'],
    )
    logits = predict_logits(model, loader, device)
    summary = score_logits(logits, dev, cfg.RESULTS_DIR / 'best_reevaluation', gold_path=cfg.GOLD_PATH)
    best = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    reference = cfg.RESULTS_DIR / f"epoch_{best['best_epoch']}" / 'predictions_with_scores.csv'
    historical = pd.read_csv(reference)
    columns = [f'logit_{label}' for label in cfg.LABELS]
    torch.testing.assert_close(logits, torch.tensor(historical[columns].values, dtype=torch.float32))
    print('重载后的逐条分数与保存轮次一致。')
    return summary


if __name__ == '__main__':
    main()
