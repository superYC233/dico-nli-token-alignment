"""只读0.938145最佳权重，复核Dev并导出Train；不训练、不修改权重。"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_generalization_audit.export_predictions import predict, sha256


RUN_DIR = Path(__file__).resolve().parent / 'outputs/run06_relation48_staged'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged')
OUTPUT_DIR = RUN_DIR / 'error_analysis'
LABELS = ['F', 'B', 'E', 'N']


def main():
    config = json.loads((RUN_DIR / 'run_config.json').read_text(encoding='utf-8'))
    best = json.loads((MODEL_DIR / 'best_metrics.json').read_text(encoding='utf-8'))
    assert best['best_epoch'] == 2 and abs(best['weighted_f1'] - .9381451572973859) < 1e-12
    assert config['max_length'] == 128 and config['dev_batch_size'] == 16 and config['dtype'] == 'float32'
    train = pd.read_csv(RUN_DIR / 'training_data.csv', keep_default_na=False)
    train_hash = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert train_hash == config['augmentation']['effective_train_sha256']
    dev_path = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
    assert sha256(dev_path) == config['dev_sha256']
    dev = pd.read_csv(dev_path, keep_default_na=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        config['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    weights = torch.load(MODEL_DIR / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda')
    model.to(device).eval()
    print('只读最佳epoch2，不创建优化器、不反向传播。')
    print('先复核Dev：')
    dev_result = predict(dev, tokenizer, model, device)
    saved = pd.read_csv(RUN_DIR / 'epoch_2/predictions_with_scores.csv', keep_default_na=False)
    assert dev_result.instance_id.tolist() == saved.instance_id.tolist()
    differences = {column: int(dev_result[column].ne(saved[column]).sum())
                   for column in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values()), f'Dev复现不一致：{differences}'
    print('与保存预测的差异：', differences)
    dev_result.to_csv(OUTPUT_DIR / 'dev_readout.csv', index=False, encoding='utf-8-sig')
    print('再预测当次3290行训练快照：')
    train_result = predict(train, tokenizer, model, device)
    train_result.to_csv(OUTPUT_DIR / 'train_readout.csv', index=False, encoding='utf-8-sig')
    subsets = {
        'train_all': train_result,
        'train_official': train_result[train_result.instance_id.str.startswith('dico_')],
        'train_old_scope200': train_result[train_result.instance_id.str.startswith('aug_scope_')],
        'train_new48': train_result[train_result.instance_id.str.startswith('aug_manual_r_')],
        'dev': dev_result,
    }
    assert [len(subsets[name]) for name in subsets] == [3290, 3042, 200, 48, 660]
    summaries = []
    for split, frame in subsets.items():
        for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
            correct = frame[column].eq(frame.gold)
            item = {'split': split, 'method': method, 'rows': len(frame),
                    'weighted_f1': f1_score(frame.gold, frame[column], average='weighted'),
                    'accuracy': accuracy_score(frame.gold, frame[column]),
                    'correct': int(correct.sum()),
                    'nonN_to_N': int((frame.gold.ne('N') & frame[column].eq('N')).sum()),
                    'truncated_rows': int(frame.truncated.sum())}
            for label in LABELS:
                item[f'{label}_correct'] = int((frame.gold.eq(label) & correct).sum())
            summaries.append(item)
            matrix = pd.crosstab(frame.gold, frame[column]).reindex(index=LABELS, columns=LABELS, fill_value=0)
            matrix.to_csv(OUTPUT_DIR / f'{split}_{method}_confusion.csv', encoding='utf-8-sig')
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / 'train_dev_readout_summary.csv', index=False, encoding='utf-8-sig')
    manifest = {'checkpoint': str(MODEL_DIR / 'best_model.pth'),
                'checkpoint_sha256': sha256(MODEL_DIR / 'best_model.pth'),
                'training_snapshot_sha256': train_hash, 'dev_prediction_differences': differences,
                'seed_checkpoint': 42, 'epoch': 2, 'batch_size': 16, 'max_length': 128,
                'dtype': 'float32', 'training_performed': False,
                'script_sha256': sha256(Path(__file__).resolve())}
    (OUTPUT_DIR / 'inference_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print('只读审计完成：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
