"""评估诊断候选。审核前的建议标签匹配率与审核后成绩分开保存。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

from experiments.deberta_minimal_contrasts import config as cfg
from experiments.deberta_minimal_contrasts.prepare_diagnostics import file_hash
from experiments.deberta_generalization_audit.export_predictions import predict, sha256
from experiments.deberta_row_ce_frozen.train_deberta_row_ce import build_model


def read_review(data_dir=None):
    data_dir = cfg.DATA_DIR if data_dir is None else Path(data_dir)
    examples = pd.read_csv(data_dir / 'examples.csv', keep_default_na=False)
    review = pd.read_csv(data_dir / 'review.csv', keep_default_na=False)
    if not review.instance_id.is_unique or set(review.instance_id) != set(examples.instance_id):
        raise ValueError('review.csv 的样本 ID 被删除、重复或增加，请保留原始样本。')
    review = review.set_index('instance_id').loc[examples.instance_id].reset_index()
    # 允许审核人改审核栏；不允许无记录地改文本、来源或建议标签。
    for column in examples.columns:
        if not examples[column].equals(review[column]):
            raise ValueError(f'review.csv 的 {column} 被修改；只改 review_status/label/note。')
    allowed = ['observed_train', 'pending', 'approved', 'rejected']
    if not review.review_status.isin(allowed).all():
        raise ValueError('review_status 只能写 observed_train / pending / approved / rejected。')
    if ((review.kind != 'anchor') & (review.review_status == 'observed_train')).any():
        raise ValueError('只有 anchor 来源行可以标为 observed_train。')
    approved = review.review_status == 'approved'
    if not review.loc[approved, 'review_label'].isin(cfg.LABELS).all():
        raise ValueError('approved 行必须在 review_label 中填写 F/B/E/N。')
    return review


def load_model():
    audit = json.loads((cfg.AUDIT_DIR / 'inference_manifest.json').read_text(encoding='utf-8'))
    checkpoint = cfg.MODEL_DIR / 'best_model.pth'
    if sha256(checkpoint) != audit['checkpoint_sha256']:
        raise ValueError('本脚本固定诊断审计中的 epoch8 权重，请勿替换后沿用旧结果目录。')
    for name, path in [('train', cfg.TRAIN_PATH), ('dev', cfg.DEV_PATH)]:
        if file_hash(path) != audit[name + '_sha256']:
            raise ValueError(name + ' 原始文件已变化，请重新核对协议。')
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    return model, tokenizer, device, audit


def check_dev(model, tokenizer, device):
    # 模型与原审计一致后，再解读新探针；Dev 不参与构造建议标签。
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    new = predict(dev, tokenizer, model, device)
    old = pd.read_csv(cfg.AUDIT_DIR / 'dev_predictions.csv', keep_default_na=False)
    if not (new.instance_id == old.instance_id).all():
        raise ValueError('Dev 顺序或 ID 与历史不一致。')
    differences = int((new.prediction != old.prediction).sum())
    if differences:
        raise ValueError(f'Dev 有 {differences} 条预测未复现，先查加载与推理协议。')
    return differences


def save_summaries(result, output_dir=None):
    output_dir = cfg.OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 建议标签只是诊断假设；即使模型不同意，也不能自动判定模型或标签谁错。
    result['matches_proposed_label'] = result.prediction == result.proposed_label
    provisional = result.groupby('kind', sort=False).agg(
        rows=('instance_id', 'size'), families=('family_id', 'nunique'),
        matches_proposal=('matches_proposed_label', 'sum'),
        provisional_match_rate=('matches_proposed_label', 'mean'),
    ).reset_index()
    provisional.to_csv(output_dir / 'provisional_summary.csv', index=False, encoding='utf-8-sig')

    family_records = []
    for family, group in result.groupby('family_id', sort=False):
        record = {'family_id': family}
        kinds = ['anchor', 'restriction_rephrase', 'equivalence_rephrase', 'modifier_scope',
                 'core_object_change', 'modifier_binding', 'conjunct_order']
        for kind in kinds:
            part = group[group.kind == kind]
            record[kind + '_rows'] = len(part)
            expected_rows = 1 if kind in ['core_object_change', 'modifier_binding'] else 2
            record[kind + '_all_match_proposal'] = bool(part.matches_proposed_label.all()) if len(part) == expected_rows else None
        novel = group[group.kind != 'anchor']
        record['generated_rows'] = len(novel)
        record['generated_all_match_proposal'] = bool(novel.matches_proposed_label.all()) if len(novel) == 10 else None
        family_records.append(record)
    family_summary = pd.DataFrame(family_records)
    family_summary.to_csv(output_dir / 'family_summary.csv', index=False, encoding='utf-8-sig')

    # 真正经过人工批准的子集单独评分，不能混入 pending 来扩大样本量。
    approved = result[(result.review_status == 'approved') & (result.kind != 'anchor')].copy()
    approved['correct_reviewed'] = approved.prediction == approved.review_label
    approved.to_csv(output_dir / 'approved_predictions.csv', index=False, encoding='utf-8-sig')
    reviewed_summary = {'approved_rows': len(approved), 'approved_families': approved.family_id.nunique()}
    if len(approved):
        reviewed_summary['correct'] = int(approved.correct_reviewed.sum())
        reviewed_summary['accuracy'] = float(approved.correct_reviewed.mean())
        reviewed_summary['weighted_f1'] = f1_score(
            approved.review_label, approved.prediction, labels=cfg.LABELS,
            average='weighted', zero_division=0,
        )
    matrix = confusion_matrix(approved.review_label, approved.prediction, labels=cfg.LABELS) if len(approved) else [[0] * 4 for _ in range(4)]
    pd.DataFrame(matrix, index=cfg.LABELS, columns=cfg.LABELS).to_csv(
        output_dir / 'approved_confusion.csv', encoding='utf-8-sig', index_label='review_label')
    (output_dir / 'approved_summary.json').write_text(
        json.dumps(reviewed_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    result.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    result[~result.matches_proposed_label].to_csv(
        output_dir / 'disagreements_to_review.csv', index=False, encoding='utf-8-sig')
    print('\n===== 按建议标签的临时诊断；不是正式准确率 =====')
    print(provisional.to_string(index=False))
    print('\n人工批准的新探针行数：', len(approved))
    if not len(approved):
        print('当前没有已审核的新探针成绩。先看 review.csv，再看模型预测，避免被模型答案影响。')
    return result, provisional, family_summary, reviewed_summary


def main():
    from experiments.deberta_minimal_contrasts.prepare_diagnostics import main as prepare
    prepare()
    review = read_review()
    selected = review[review.eligible & (review.review_status != 'rejected')].copy().reset_index(drop=True)
    if len(selected) == 0:
        raise ValueError('没有可评估的候选，请检查 eligible 和 review_status。')
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, audit = load_model()
    differences = check_dev(model, tokenizer, device)
    # 复用原审计的同一分词和前向实现；其参数固定为 batch16 / 长度128。
    if cfg.BATCH_SIZE != 16 or cfg.MAX_LENGTH != 128:
        raise ValueError('本次复用固定 batch16/长度128 的审计推理，请不要单改配置。')
    inputs = selected.copy()
    inputs['label'] = inputs.proposed_label.map(dict(zip(cfg.LABELS, cfg.FULL_LABELS)))
    predictions = predict(inputs, tokenizer, model, device)
    # 这里只研究独立预测，不把自建 pair_id 的结构解码成绩当语义诊断。
    columns = ['prediction', 'max_probability', 'encoded_length', 'truncated', 'input_hash']
    columns += [f'{prefix}_{label}' for prefix in ['logit', 'prob'] for label in cfg.LABELS]
    result = pd.concat([selected, predictions[columns]], axis=1)
    result, provisional, families, reviewed = save_summaries(result)
    manifest = {'checkpoint': str(cfg.MODEL_DIR / 'best_model.pth'),
                'checkpoint_sha256': audit['checkpoint_sha256'],
                'examples_sha256': file_hash(cfg.DATA_DIR / 'examples.csv'),
                'review_sha256': file_hash(cfg.DATA_DIR / 'review.csv'),
                'preparation_manifest_sha256': file_hash(cfg.DATA_DIR / 'manifest.json'),
                'batch_size': 16, 'dtype': 'float32', 'max_length': 128,
                'dev_prediction_differences': differences,
                'rows_evaluated': len(result), 'families': result.family_id.nunique(),
                'new_probe_rows': int((result.kind != 'anchor').sum()),
                'approved_new_rows': int(((result.kind != 'anchor') & (result.review_status == 'approved')).sum()),
                'warning': 'Train-derived exploratory diagnostics; pending labels are not verified gold; not an independent test set.'}
    (cfg.OUTPUT_DIR / 'evaluation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nDev 逐条复现差异：', differences)
    print('已评估行数：', len(result), '；结果目录：', cfg.OUTPUT_DIR)


if __name__ == '__main__':
    main()
