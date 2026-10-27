"""同一批保留家族评估旧/新模型；只用于诊断，不选模型、不更新参数。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_family_aug.prepare_family_data import PREPARED_DIR
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_stage_average.train_deberta_stage_average import evaluate_phase


def evaluate_probes(model, tokenizer, device, output_dir, max_length=128):
    diagnostic = pd.read_csv(PREPARED_DIR / 'diagnostic_rows.csv', keep_default_na=False)
    notes = pd.read_csv(PREPARED_DIR / 'diagnostic_label_notes.csv', keep_default_na=False)
    loader = DataLoader(
        RowDataset(diagnostic, tokenizer, max_length), batch_size=16, shuffle=False,
        generator=torch.Generator().manual_seed(43),
    )
    output_dir = Path(output_dir)
    print('\n===== 未参与训练的保留家族：主要看independent =====')
    summary = evaluate_phase(model, loader, diagnostic, device, output_dir,
                             PREPARED_DIR / 'diagnostic_reference.csv')
    predictions = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    metadata = notes[['instance_id', 'family_id', 'family_type', 'contrast_type', 'target', 'label_reason']]
    detail = predictions.merge(metadata, on='instance_id', validate='one_to_one')
    detail['independent_correct'] = detail.prediction.eq(detail.target)
    detail['pair_aware_correct'] = detail.pair_prediction.eq(detail.target)
    detail.to_csv(output_dir / 'diagnostic_details.csv', index=False, encoding='utf-8-sig')
    breakdown = detail.groupby('family_type').agg(
        rows=('instance_id', 'size'), independent_correct=('independent_correct', 'sum'),
        pair_aware_correct=('pair_aware_correct', 'sum'),
    ).reset_index()
    breakdown['independent_accuracy'] = breakdown.independent_correct / breakdown.rows
    breakdown.to_csv(output_dir / 'family_type_summary.csv', index=False, encoding='utf-8-sig')
    print(breakdown.to_string(index=False))
    detail[~detail.independent_correct].to_csv(output_dir / 'independent_errors.csv', index=False, encoding='utf-8-sig')
    return summary


def main(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    assert not config.get('use_lexical_features', False), '此入口用于原0.938架构及本批数据训练的模型。'
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        config['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    device = torch.device('cuda')
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False).to(device)
    weights = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    output_dir = Path(__file__).resolve().parent / 'outputs/probe_comparison' / model_dir.name
    evaluate_probes(model, tokenizer, device, output_dir, config['max_length'])
    print('保留诊断结果：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path,
                        default=Path('D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged'))
    args = parser.parse_args()
    main(args.model_dir)
