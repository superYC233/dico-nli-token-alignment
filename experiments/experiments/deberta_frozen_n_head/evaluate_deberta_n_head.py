"""右键运行：加载本次一个完整checkpoint，复核两种预测。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer

from experiments.deberta_frozen_n_head import config as cfg
from experiments.deberta_frozen_n_head.model import build_model
from experiments.deberta_frozen_n_head.train_deberta_n_head import selection_record, file_hash
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase
from torch.utils.data import DataLoader


def main():
    config = json.loads((cfg.MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    best = json.loads((cfg.MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    assert file_hash(cfg.DEV_PATH) == config['dev_sha256']
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone_config = AutoConfig.from_pretrained(cfg.MODEL_DIR / 'backbone_config', local_files_only=True)
    model = build_model(backbone_config, tokenizer, config['head_hidden_size'], config['n_prior'])
    # 这个文件包含编码器、原分类器及新N头；此处不再读取旧0.9381权重。
    model.load_state_dict(torch.load(cfg.MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True))
    device = torch.device(cfg.DEVICE)
    model.to(device).eval()
    loader = DataLoader(
        RowDataset(dev, tokenizer, config['max_length']),
        batch_size=config['dev_batch_size'], shuffle=False,
        generator=torch.Generator().manual_seed(config['seed'] + 1),
    )
    summary = evaluate_phase(model, loader, dev, device, cfg.RESULTS_DIR / 'evaluation', cfg.GOLD_PATH)
    selected = selection_record(summary, best['epoch'], config['target_independent'], config['target_pair'])
    assert abs(selected['independent_f1'] - best['independent_f1']) < 1e-10
    assert abs(selected['pair_aware_f1'] - best['pair_aware_f1']) < 1e-10
    print('已复现同一个最佳checkpoint：', selected)


if __name__ == '__main__':
    main()
