"""只读0.9381权重：拆开CLS与Alignment分数，不训练、不选择融合权重。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import pair_aware

RUN_DIR = Path(__file__).resolve().parent / 'outputs/run06_relation48_staged'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged')
OUTPUT_DIR = RUN_DIR / 'branch_audit'
LABELS = ['F', 'B', 'E', 'N']


def main():
    config = json.loads((RUN_DIR / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        config['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    weights = torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.to('cuda').eval()

    captured = {}
    def save_cls(module, inputs, output):
        captured['cls'] = output.detach().cpu()
    def save_alignment(module, inputs, output):
        captured['alignment'] = output.detach().cpu()
    handles = [model.output.register_forward_hook(save_cls),
               model.match_output.register_forward_hook(save_alignment)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, comparisons = [], []
    paths = {'dev': ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv',
             'train': RUN_DIR / 'training_data.csv'}
    for split, path in paths.items():
        frame = pd.read_csv(path, keep_default_na=False)
        rows = RowDataset(frame, tokenizer, config['max_length'])
        loader = DataLoader(rows, batch_size=16, shuffle=False)
        values = {name: [] for name in ['full', 'cls', 'alignment']}
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda'), None)
                values['full'].append(logits.cpu())
                values['cls'].append(captured['cls'])
                values['alignment'].append(captured['alignment'])
        values = {name: torch.cat(parts) for name, parts in values.items()}
        assert torch.allclose(values['full'], values['cls'] + values['alignment'])
        result = frame.copy()
        gold = rows.labels.numpy()
        for branch, logits in values.items():
            for index, label in enumerate(LABELS):
                result[f'{branch}_logit_{label}'] = logits[:, index].tolist()
            for decoding, predictions in [('independent', logits.argmax(1)),
                                          ('pair_aware', pair_aware(logits, frame))]:
                predictions = predictions.numpy()
                result[f'{branch}_{decoding}'] = [LABELS[i] for i in predictions]
                summary = {'split': split, 'branch': branch, 'decoding': decoding,
                           'weighted_f1': f1_score(gold, predictions, average='weighted'),
                           'correct': int((predictions == gold).sum()),
                           'nonN_to_N': int(((gold != 3) & (predictions == 3)).sum())}
                for index, label in enumerate(LABELS):
                    summary[f'{label}_correct'] = int(((gold == index) & (predictions == index)).sum())
                summaries.append(summary)
        saved = pd.read_csv(RUN_DIR / 'error_analysis' / f'{split}_readout.csv')
        assert result.instance_id.tolist() == saved.instance_id.tolist()
        assert result.full_independent.eq(saved.prediction).all()
        assert result.full_pair_aware.eq(saved.pair_prediction).all()
        result['gold'] = [LABELS[i] for i in gold]
        for decoding in ['independent', 'pair_aware']:
            base_correct = result[f'full_{decoding}'].eq(result.gold)
            cls_correct = result[f'cls_{decoding}'].eq(result.gold)
            align_correct = result[f'alignment_{decoding}'].eq(result.gold)
            comparisons.append({'split': split, 'decoding': decoding,
                                'full_wrong_cls_right': int((~base_correct & cls_correct).sum()),
                                'full_wrong_alignment_right': int((~base_correct & align_correct).sum()),
                                'full_wrong_either_branch_right': int((~base_correct & (cls_correct | align_correct)).sum()),
                                'full_right_both_branches_wrong': int((base_correct & ~cls_correct & ~align_correct).sum())})
        result.to_csv(OUTPUT_DIR / f'{split}_branches.csv', index=False, encoding='utf-8-sig')
    for handle in handles:
        handle.remove()
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(OUTPUT_DIR / 'comparison.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('Full分支Train/Dev均与历史预测逐条一致；未训练。')


if __name__ == '__main__':
    main()
