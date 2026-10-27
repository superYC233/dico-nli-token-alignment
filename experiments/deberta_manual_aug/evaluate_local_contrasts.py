"""先运行本文件：同一批64行诊断比较0.938模型和run03；只预测，不训练。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.evaluation import pair_aware
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_frozen_n_head.model_top2 import build_model as build_top2
from experiments.deberta_manual_aug.prepare_local_contrasts import (
    HERE, BASELINE_DIR, PREPARED_DIR, FULL, prepare,
)


RUN_NAME = 'run01_local_diagnostic'
BATCH_SIZE = 16
LABELS = ['F', 'B', 'E', 'N']
MODEL_PATHS = [
    ('original_0p938', 'original', Path('D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged')),
    ('top2_row_best', 'top2', Path('D:/model/SEAMEVAL/deberta_frozen_n_head/run03_top2_row_only')),
]


def load_model(kind, model_dir):
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    if kind == 'original':
        backbone_config = AutoConfig.from_pretrained(
            config['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        )
        model = DebertaDiCoModel(AutoModel.from_config(backbone_config, dtype=torch.float32),
                                tokenizer, feature_size=0, use_lexical_features=False)
    else:
        backbone_config = AutoConfig.from_pretrained(model_dir / 'backbone_config', local_files_only=True)
        model = build_top2(backbone_config, tokenizer, config['head_hidden_size'],
                           config['n_prior'], config['unfreeze_last_layers'])
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model, tokenizer, config


def predict(model, tokenizer, frame, device, max_length=128):
    rows = RowDataset(frame, tokenizer, max_length)
    loader = DataLoader(rows, batch_size=BATCH_SIZE, shuffle=False,
                        generator=torch.Generator().manual_seed(43))
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc='局部对照预测'):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
            values.append(logits.float().cpu())
    return torch.cat(values)


def summarize(logits, frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.copy().reset_index(drop=True)
    frame['gold'] = frame.label.map({full: short for short, full in FULL.items()})
    frame['prediction'] = [LABELS[i] for i in logits.argmax(1).tolist()]
    frame['pair_prediction'] = [LABELS[i] for i in pair_aware(logits, frame).tolist()]
    for index, label in enumerate(LABELS):
        frame['logit_' + label] = logits[:, index].tolist()
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    summary = []
    groups = [('all', frame)] + list(frame.groupby('family_type'))
    for scope, part in groups:
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            correct = part[column].eq(part.gold)
            record = {'scope': scope, 'method': method, 'rows': len(part),
                      'weighted_f1': f1_score(part.gold, part[column], average='weighted', zero_division=0),
                      'accuracy': accuracy_score(part.gold, part[column]), 'correct': int(correct.sum()),
                      'whole_families_correct': int(correct.groupby(part.family_id).all().sum())}
            for label in LABELS:
                record[label + '_correct'] = int((part.gold.eq(label) & correct).sum())
            summary.append(record)
            matrix = confusion_matrix(part.gold, part[column], labels=LABELS)
            pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
                output_dir / f'confusion_{scope}_{method}.csv', encoding='utf-8-sig')
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    frame[frame.prediction.ne(frame.gold)].to_csv(
        output_dir / 'independent_errors.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary, frame


def check_historical_dev(model, tokenizer, config, name, device, output_dir):
    # 只是检查加载、分词、batch和精度是否复现旧模型，不据此修改新数据。
    if name == 'original_0p938':
        old_path = BASELINE_DIR / 'epoch_2/predictions_with_scores.csv'
    else:
        old_path = ROOT / 'experiments/deberta_frozen_n_head/outputs/run03_top2_row_only/best/dev/predictions_with_scores.csv'
    old = pd.read_csv(old_path, keep_default_na=False)
    dev = pd.read_csv(ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv', keep_default_na=False)
    logits = predict(model, tokenizer, dev, device, config['max_length'])
    prediction = [LABELS[i] for i in logits.argmax(1).tolist()]
    paired = [LABELS[i] for i in pair_aware(logits, dev).tolist()]
    old = old.set_index('instance_id').loc[dev.instance_id]
    differences = {'independent': sum(a != b for a, b in zip(prediction, old.prediction)),
                   'pair_aware': sum(a != b for a, b in zip(paired, old.pair_prediction))}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'historical_reproduction.json').write_text(
        json.dumps(differences, ensure_ascii=False, indent=2), encoding='utf-8')
    print('与历史Dev预测不一致的行数：', differences)
    assert differences == {'independent': 0, 'pair_aware': 0}, '旧预测未复现，先核对模型路径，不解释新诊断差异。'


def main(check_only=False, device_name='cuda'):
    _, diagnostic, report = prepare()
    output_dir = HERE / 'outputs' / RUN_NAME
    all_summaries, all_frames = [], []
    for name, kind, model_dir in MODEL_PATHS:
        print('\n=====', name, '=====')
        print('读取权重：', model_dir / 'best_model.pth')
        assert (model_dir / 'best_model.pth').is_file()
        model, tokenizer, config = load_model(kind, model_dir)
        device = torch.device('cpu' if check_only else device_name)
        model.to(device).eval()
        if check_only:
            logits = predict(model, tokenizer, diagnostic.iloc[:2], device, config['max_length'])
            assert logits.shape == (2, 4) and torch.isfinite(logits).all()
            print('严格加载权重及CPU两行前向检查通过；没有训练。')
        else:
            check_historical_dev(model, tokenizer, config, name, device, output_dir / name)
            logits = predict(model, tokenizer, diagnostic, device, config['max_length'])
            summary, frame = summarize(logits, diagnostic, output_dir / name)
            summary.insert(0, 'model', name)
            all_summaries.append(summary)
            all_frames.append(frame)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if check_only:
        return
    pd.concat(all_summaries, ignore_index=True).to_csv(
        output_dir / 'comparison.csv', index=False, encoding='utf-8-sig')
    first, second = all_frames
    assert first.instance_id.tolist() == second.instance_id.tolist()
    comparison = first[['instance_id', 'family_id', 'family_type', 'text1', 'text2', 'gold', 'label_reason']].copy()
    comparison['original_prediction'] = first.prediction
    comparison['top2_prediction'] = second.prediction
    comparison['fixed'] = first.prediction.ne(first.gold) & second.prediction.eq(second.gold)
    comparison['broken'] = first.prediction.eq(first.gold) & second.prediction.ne(second.gold)
    comparison.to_csv(output_dir / 'row_comparison.csv', index=False, encoding='utf-8-sig')
    (output_dir / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n看independent和各类错误；pair-aware使用构造的F/B/E分组，不能代替独立诊断。')
    print('汇总：', output_dir / 'comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    main(args.check_only, args.device)
