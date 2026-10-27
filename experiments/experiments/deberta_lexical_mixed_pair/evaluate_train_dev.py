"""同一份最佳权重评估 Train 和 Dev，只推理，不训练。"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config as cfg
from experiments.deberta_lexical_mixed_pair.data import read_data
from experiments.deberta_lexical_mixed_pair.train_deberta_mixed_pair import make_dev_loader
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits, pair_aware


# 默认检查混合训练基准，不随 config.py 当前选择的 run03 改变。
RUN_NAME = 'run01_mixed_pair'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair') / RUN_NAME
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME / 'train_dev_evaluation'


def summarize(logits, frame, split, row_weights):
    gold = frame['label'].map(cfg.LABEL_TO_ID).to_numpy(dtype=int)
    independent = logits.argmax(dim=1).numpy()
    structured = pair_aware(logits, frame).numpy()
    eval_ce = F.cross_entropy(logits, torch.tensor(gold), weight=torch.tensor(row_weights)).item()
    n_score = (logits[:, 3] - torch.logsumexp(logits[:, :3], dim=1)).numpy()
    n_auc = roc_auc_score(gold == 3, n_score)
    records = []
    for method, prediction in [('independent', independent), ('pair_aware', structured)]:
        counts = confusion_matrix(gold, prediction, labels=range(4))
        record = {
            'split': split, 'method': method, 'rows': len(gold),
            'weighted_f1': f1_score(gold, prediction, average='weighted'),
            'accuracy': accuracy_score(gold, prediction),
            'N_correct': int(counts[3, 3]), 'N_total': int(counts[3].sum()),
            'nonN_to_N': int(counts[:3, 3].sum()),
            'N_to_nonN': int(counts[3, :3].sum()),
            'E_to_FB': int(counts[2, :2].sum()),
            'FB_to_E': int(counts[:2, 2].sum()),
            'F_B_reversed': int(counts[0, 1] + counts[1, 0]),
            'eval_row_CE_before_decode': eval_ce, 'N_AUROC_before_decode': n_auc,
        }
        for index, label in enumerate(cfg.LABELS):
            record[f'{label}_correct'] = int(counts[index, index])
        records.append(record)
    return records, independent, structured


def main():
    saved = json.loads((MODEL_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'], use_lexical_features=saved['use_lexical_features'],
    )
    state = torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device('cuda')
    model.to(device)
    train, dev = read_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    print('本次只评估同一份最佳权重，不更新模型：', MODEL_DIR)
    for split, frame in [('train', train), ('dev', dev)]:
        print(f'\n===== {split}：{len(frame)} 行 =====')
        loader = make_dev_loader(
            frame, tokenizer, build_feature_tensor(frame), saved['max_length'], saved['dev_batch_size'],
        )
        logits = predict_logits(model, loader, device)
        metrics, independent, structured = summarize(logits, frame, split, saved['row_weights'])
        records.extend(metrics)
        predictions = frame[['instance_id', 'pair_id', 'text1', 'text2', 'label']].copy()
        predictions['prediction'] = np.array(cfg.LABELS)[independent]
        predictions['pair_prediction'] = np.array(cfg.LABELS)[structured]
        for index, label in enumerate(cfg.LABELS):
            predictions[f'logit_{label}'] = logits[:, index].numpy()
        predictions.to_csv(OUTPUT_DIR / f'{split}_predictions_with_scores.csv', index=False, encoding='utf-8-sig')
        print(pd.DataFrame(metrics).to_string(index=False))
    pd.DataFrame(records).to_csv(OUTPUT_DIR / 'train_dev_summary.csv', index=False, encoding='utf-8-sig')
    print('\n结果：', OUTPUT_DIR / 'train_dev_summary.csv')
    print('Train 是拟合诊断，不是泛化成绩；这里的 eval CE 与训练过程的平均 loss 口径不同。')


if __name__ == '__main__':
    main()
