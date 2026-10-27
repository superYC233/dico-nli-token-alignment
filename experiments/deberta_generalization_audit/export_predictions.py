"""只加载已有最佳权重，导出 Train / Dev；不训练、不改标签。"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / 'outputs/row_ce_frozen_epoch8'
# 审计固定旧基线，不随当前训练 config.py 的 RUN_NAME 改变。
BASELINE_MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_row_ce_frozen/run01_freeze6')
BASELINE_RESULTS_DIR = ROOT / 'experiments/deberta_row_ce_frozen/outputs/run01_freeze6'

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.deberta_row_ce_frozen import config as cfg
from experiments.deberta_row_ce_frozen.train_deberta_row_ce import build_model
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.evaluation import pair_aware

LABELS = ['F', 'B', 'E', 'N']
SHORT = dict(zip(cfg.FULL_LABELS, LABELS))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def predict(frame, tokenizer, model, device):
    dataset = RowDataset(frame, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    values = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc='只读推理'):
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device), None)
            values.append(logits.float().cpu())
    logits = torch.cat(values)
    result = frame.copy()
    result['gold'] = frame['label'].map(SHORT)
    result['prediction'] = [LABELS[i] for i in logits.argmax(1).tolist()]
    result['pair_prediction'] = [LABELS[i] for i in pair_aware(logits, frame).tolist()]
    probabilities = logits.softmax(1).numpy()
    for index, label in enumerate(LABELS):
        result[f'logit_{label}'] = logits[:, index].numpy()
        result[f'prob_{label}'] = probabilities[:, index]
    result['correct'] = result['prediction'] == result['gold']
    result['pair_correct'] = result['pair_prediction'] == result['gold']
    result['max_probability'] = probabilities.max(1)
    result['probability_gold'] = [probabilities[i, LABELS.index(g)]
                                  for i, g in enumerate(result['gold'])]
    ordered = np.sort(probabilities, axis=1)
    result['probability_margin'] = ordered[:, -1] - ordered[:, -2]
    result['encoded_length'] = dataset.encoded['attention_mask'].sum(1).numpy()
    # 检查真实输入碰撞；不把 padding 计入 hash。
    result['input_hash'] = [hashlib.sha256(
        np.asarray(ids[mask.bool()].tolist(), dtype=np.int64).tobytes()).hexdigest()
        for ids, mask in zip(dataset.encoded['input_ids'], dataset.encoded['attention_mask'])]
    lengths = tokenizer(frame['text1'].tolist(), frame['text2'].tolist(),
                        truncation=False, padding=False)['input_ids']
    result['untruncated_length'] = [len(ids) for ids in lengths]
    result['truncated'] = result['untruncated_length'] > cfg.MAX_LENGTH
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = BASELINE_MODEL_DIR / 'best_model.pth'
    best = json.loads((BASELINE_MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    if best['best_epoch'] != 8:
        raise ValueError('此审计固定 epoch 8，当前权重信息已变化，请新建输出目录。')
    manifest = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'best_metrics': best, 'train_sha256': sha256(cfg.TRAIN_PATH),
        'dev_sha256': sha256(cfg.DEV_PATH), 'batch_size': 16,
        'dtype': 'float32', 'max_length': cfg.MAX_LENGTH,
        'selection': 'Dev independent weighted F1',
        'training': 'Row CE only; embeddings and lower 6 layers frozen',
    }
    manifest_path = OUT / 'inference_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old == manifest and all((OUT / f'{s}_predictions.csv').exists() for s in ['train', 'dev']):
            print('权重、数据及协议未变，复用已验证的推理结果。')
            return
        raise ValueError('已有审计输入已变化，请使用新的输出目录。')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_DIR / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device).eval()
    # 先复现 Dev，防止把另一轮 / 另一套模型当作当前模型。
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    result = predict(dev, tokenizer, model, device)
    historical = pd.read_csv(BASELINE_RESULTS_DIR / 'epoch_8/predictions_with_scores.csv', keep_default_na=False)
    checked = result.merge(historical[['instance_id', 'prediction', 'pair_prediction']],
                           on='instance_id', suffixes=('', '_historical'), validate='one_to_one')
    differences = {column: int((checked[column] != checked[column + '_historical']).sum())
                   for column in ['prediction', 'pair_prediction']}
    if any(differences.values()) or len(checked) != len(dev):
        raise ValueError(f'Dev 复现不一致：{differences}；先查协议，不继续出报告。')
    manifest['dev_prediction_differences'] = differences
    result.to_csv(OUT / 'dev_predictions.csv', index=False, encoding='utf-8-sig')
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    result = predict(train, tokenizer, model, device)
    result.to_csv(OUT / 'train_predictions.csv', index=False, encoding='utf-8-sig')
    # 差异检查是输出，不作为下一次缓存输入；单独保存。
    (OUT / 'dev_reproduction.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    manifest.pop('dev_prediction_differences')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('导出完成：', OUT)


if __name__ == '__main__':
    main()
