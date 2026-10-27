"""右键运行：用 0.9248 原版权重测同一批诊断题；只预测，不训练。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import score_logits
from experiments.deberta_generalization_audit.export_predictions import predict, sha256
from experiments.deberta_minimal_contrasts import config as cfg
from experiments.deberta_minimal_contrasts.evaluate_diagnostics import read_review, save_summaries


# RUN_NAME 只控制新结果位置；诊断题仍读取 prepared/run02_binding。
RUN_NAME = 'run03_original_binding'
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
MODEL_DIR = ROOT / 'model/deberta_nli_base_token_alignment_official_full_control_seed_42'
CHECKPOINT = MODEL_DIR / 'epoch_7.pth'


def load_original_model():
    cache_dir = get_model_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir, local_files_only=True,
    )
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir, local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    # 已核对与原 train.py 同结构、同参数名，不创建词汇分支。
    # 不 import train.py，避免触发它在模块顶层的数据和模型加载。
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    )
    weights = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    print('原版权重：', CHECKPOINT)
    print('推理协议：batch=16，float32，max_length=128；设备：', device)
    return model, tokenizer, device


def reproduce_dev(model, tokenizer, device):
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    result = predict(dev, tokenizer, model, device)
    historical = pd.read_csv(MODEL_DIR / 'dev_predictions_pair_aware.csv', keep_default_na=False)
    historical = historical.set_index('instance_id').loc[result.instance_id]
    full_to_short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    expected = historical.label.map(full_to_short).to_numpy()
    differences = int((result.pair_prediction.to_numpy() != expected).sum())
    print('与原版历史 pair-aware 预测不同的行数：', differences)
    if differences:
        raise ValueError('先解决原版复现差异，再解读诊断结果。')

    columns = [f'logit_{label}' for label in cfg.LABELS]
    logits = torch.tensor(result[columns].to_numpy(), dtype=torch.float32)
    gold_path = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
    summary = score_logits(logits, dev, OUTPUT_DIR / 'dev', gold_path=gold_path)
    return summary, differences


def evaluate_probes(model, tokenizer, device, output_dir=None):
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    review = read_review(cfg.DATA_DIR)
    # 和范围增强、复制对照使用同样的 200 条新题，排除已见的 anchor。
    selected = review[review.eligible & (review.kind != 'anchor') &
                      (review.review_status != 'rejected')].copy().reset_index(drop=True)
    inputs = selected.copy()
    inputs['label'] = inputs.proposed_label.map(dict(zip(cfg.LABELS, cfg.FULL_LABELS)))
    predicted = predict(inputs, tokenizer, model, device)
    columns = ['prediction', 'pair_prediction', 'max_probability']
    columns += [f'{prefix}_{label}' for prefix in ['logit', 'prob'] for label in cfg.LABELS]
    result = pd.concat([selected, predicted[columns]], axis=1)

    print('\n===== 当前权重：独立预测诊断 =====')
    save_summaries(result.copy(), output_dir / 'independent')

    # 另报原版完整系统的读数；分组使用 pair_id，不读取建议标签来解码。
    # 自建题保留了 F/B/E 双行、N 单行的结构，因此这不是纯语义能力指标。
    structured = result.rename(columns={
        'prediction': 'independent_prediction',
        'max_probability': 'independent_max_probability',
    }).copy()
    structured['prediction'] = structured['pair_prediction']
    print('\n===== 当前权重：pair-aware 后诊断（使用自建分组结构）=====')
    save_summaries(structured, output_dir / 'pair_aware')
    return len(result)


def main():
    if cfg.MAX_LENGTH != 128 or cfg.BATCH_SIZE != 16:
        raise ValueError('本次固定 batch16、长度128，保持与已有诊断一致。')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_original_model()
    summary, differences = reproduce_dev(model, tokenizer, device)
    rows = evaluate_probes(model, tokenizer, device)
    manifest = {
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256(CHECKPOINT),
        'diagnostic_data_dir': str(cfg.DATA_DIR),
        'examples_sha256': sha256(cfg.DATA_DIR / 'examples.csv'),
        'review_sha256': sha256(cfg.DATA_DIR / 'review.csv'),
        'dev_sha256': sha256(cfg.DEV_PATH),
        'batch_size': 16, 'max_length': 128, 'dtype': 'float32',
        'historical_pair_prediction_differences': differences,
        'dev_scores': summary.to_dict('records'), 'new_probe_rows': rows,
        'training_performed': False,
        'note': 'Same diagnostic candidates; unreviewed labels are proposals. Pair-aware uses constructed group metadata.',
    }
    (OUTPUT_DIR / 'evaluation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('\n诊断结果目录：', OUTPUT_DIR)
    print('本次只加载并预测，没有训练，也没有修改原版权重。')


if __name__ == '__main__':
    main()
