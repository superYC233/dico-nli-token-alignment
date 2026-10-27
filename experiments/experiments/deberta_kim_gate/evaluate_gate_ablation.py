"""同一个最佳模型的知识消融：只预测，不训练、不修改保存的权重。"""
import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_kim_gate import config as cfg
from experiments.deberta_kim_gate.data import read_data, build_feature_tensor, DiCoDataset
from experiments.deberta_kim_gate.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.deberta_kim_gate.relation_features import RELATION_NAMES, RELATION_VERSION
from experiments.deberta_kim_gate.wordnet_resource import resource_info
from experiments.deberta_kim_gate.model import DebertaDiCoModel
from experiments.deberta_kim_gate.evaluation import predict_logits, score_logits, save_gate_statistics


# 固定指向刚刚完成的 run02，不会因以后修改 config.RUN_NAME 而误读别的模型。
MODEL_DIR = PROJECT_ROOT / 'model/deberta_kim_gate/run02_gate_lr/kim_learned_seed_42'
VARIANTS = [
    ('learned', '正常学习到的门控'),
    ('fixed_half', '全部词典关系固定乘 0.5'),
    ('fixed_one', '全部词典关系固定乘 1.0'),
    ('no_knowledge', '关闭局部知识，保留原 20 维词汇分支'),
]


def check_saved_config(model_dir):
    saved = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    if not saved['use_kim_features'] or saved['gate_mode'] != 'learned':
        raise ValueError('本脚本需要已训练的 learned 门控模型。')
    if saved['feature_names'] != FEATURE_NAMES or saved['feature_version'] != FEATURE_VERSION:
        raise ValueError('原词汇特征与训练时不同。')
    if saved['relation_names'] != RELATION_NAMES or saved['relation_version'] != RELATION_VERSION:
        raise ValueError('局部词对关系与训练时不同。')
    for name, digest in saved['feature_source_sha256'].items():
        if hashlib.sha256((cfg.EXPERIMENT_DIR / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} 与训练时不同，请恢复后再评估。')
    if hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest() != saved['dev_sha256']:
        raise ValueError('Dev 数据与训练时不同。')
    if cfg.WORDNET_MAX_HYPERNYM_DEPTH != saved['wordnet_depth']:
        raise ValueError('WordNet 查询深度与训练时不同。')
    if resource_info() != saved['wordnet_resource']:
        raise ValueError('WordNet 资源与训练时不同。')
    return saved


def load_best_model(model_dir, saved, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    backbone = AutoModel.from_pretrained(
        saved['model_name'], cache_dir=get_model_cache_dir(), local_files_only=True,
        dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=len(saved['feature_names']),
        lexical_hidden=saved['lexical_hidden_size'],
        use_kim_features=True, gate_mode=saved['gate_mode'],
    )
    state = torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    model.eval()
    return model, tokenizer


@contextmanager
def use_variant(model, variant):
    # 临时改变计算方式，不改任何参数；即使报错也恢复原模式。
    if variant not in dict(VARIANTS):
        raise ValueError(f'未知消融方式：{variant}')
    previous_use = model.use_kim_features
    previous_mode = model.knowledge_gate.mode
    try:
        model.use_kim_features = variant != 'no_knowledge'
        modes = {'learned': 'learned', 'fixed_half': 'fixed', 'fixed_one': 'open'}
        if variant in modes:
            model.knowledge_gate.mode = modes[variant]
        yield
    finally:
        model.use_kim_features = previous_use
        model.knowledge_gate.mode = previous_mode


def compare_predictions(reference, current, output_dir):
    # 按 instance_id 对齐，不依赖 CSV 行顺序。
    columns = ['instance_id', 'gold_label', 'prediction', 'pair_prediction']
    if set(reference['instance_id']) != set(current['instance_id']):
        raise ValueError('比较文件的 Dev 样本编号不同。')
    joined = reference[columns].merge(
        current[columns], on='instance_id', suffixes=('_reference', '_current'), validate='one_to_one',
    )
    if not (joined['gold_label_reference'] == joined['gold_label_current']).all():
        raise ValueError('比较文件的金标不同。')
    full_to_short = dict(zip(cfg.FULL_LABELS, cfg.LABELS))
    gold = joined['gold_label_reference'].map(full_to_short)
    if gold.isna().any():
        raise ValueError('比较文件包含未知金标。')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        before = joined[f'{column}_reference']
        after = joined[f'{column}_current']
        old_correct, new_correct = before == gold, after == gold
        changed = before != after
        fixed = ~old_correct & new_correct
        broken = old_correct & ~new_correct
        records.append({
            'method': method, 'changed': int(changed.sum()),
            'fixed': int(fixed.sum()), 'broken': int(broken.sum()),
            'changed_wrong': int((changed & ~old_correct & ~new_correct).sum()),
        })
        joined.loc[changed].to_csv(output_dir / f'{method}_changed.csv', index=False, encoding='utf-8-sig')
        joined.loc[fixed].to_csv(output_dir / f'{method}_fixed.csv', index=False, encoding='utf-8-sig')
        joined.loc[broken].to_csv(output_dir / f'{method}_broken.csv', index=False, encoding='utf-8-sig')
    return pd.DataFrame(records)


def check_reproduction(current, reference, summary, best_info, output_dir):
    comparison = compare_predictions(reference, current, output_dir)
    print('正常门控与历史最佳预测比较：')
    print(comparison.to_string(index=False))
    score = float(summary.loc[summary['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
    if comparison['changed'].sum() != 0 or abs(score - best_info['weighted_f1']) > 1e-9:
        raise RuntimeError('正常门控未复现历史预测；先检查模型/分词/推理环境，不继续解释消融。')


def main(model_dir=MODEL_DIR):
    model_dir = Path(model_dir).resolve()
    saved = check_saved_config(model_dir)
    best_info = json.loads((model_dir / 'best_metrics.json').read_text(encoding='utf-8'))
    results_dir = cfg.EXPERIMENT_DIR / 'outputs' / model_dir.parent.name / model_dir.name
    historical_path = results_dir / f"epoch_{best_info['best_epoch']}" / 'predictions_with_scores.csv'
    if not historical_path.is_file():
        raise FileNotFoundError(f'需要历史最佳预测作为复现检查：{historical_path}')
    historical = pd.read_csv(historical_path, keep_default_na=False)
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA；此入口用于真实模型推理。')
    device = torch.device('cuda')
    model, tokenizer = load_best_model(model_dir, saved, device)
    _, dev = read_data()
    features = build_feature_tensor(dev)
    dataset = DiCoDataset(dev, tokenizer, features, saved['max_length'], use_kim_features=True)
    loader = DataLoader(dataset, batch_size=saved['dev_batch_size'], shuffle=False)
    output_dir = results_dir / ('gate_ablation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=False)
    print('同一个最佳权重：', model_dir / 'best_model.pth')
    print('最佳轮次：', best_info['best_epoch'], '；推理 batch：', saved['dev_batch_size'])
    print('输出目录：', output_dir)
    protocol = {
        'model_dir': str(model_dir), 'best_metrics': best_info, 'training': False,
        'dev_batch_size': saved['dev_batch_size'], 'dtype': 'float32',
        'variants': VARIANTS, 'historical_prediction': str(historical_path),
        'interpretation': 'inference intervention on one trained checkpoint; not a retrained baseline',
    }
    (output_dir / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    all_scores, all_changes = [], []
    learned_frame = None
    for variant, description in VARIANTS:
        print(f'\n===== {variant}：{description} =====')
        variant_dir = output_dir / variant
        with use_variant(model, variant):
            logits = predict_logits(model, loader, device)
            summary = score_logits(logits, dev, variant_dir)
            if variant != 'no_knowledge':
                save_gate_statistics(model, variant_dir)
        current = pd.read_csv(variant_dir / 'predictions_with_scores.csv', keep_default_na=False)
        if variant == 'learned':
            check_reproduction(current, historical, summary, best_info, output_dir / 'reproduction')
            learned_frame = current
        else:
            changes = compare_predictions(learned_frame, current, variant_dir / 'comparison_with_learned')
            changes.insert(0, 'variant', variant)
            all_changes.append(changes)
        summary.insert(0, 'variant', variant)
        all_scores.append(summary)
        pd.concat(all_scores, ignore_index=True).to_csv(
            output_dir / 'summary.csv', index=False, encoding='utf-8-sig',
        )
    print('\n===== 同一权重的四种推理方式 =====')
    table = pd.concat(all_scores, ignore_index=True)
    print(table[['variant', 'method', 'weighted_f1', 'E_correct', 'N_correct', 'FBE_correct']].to_string(index=False))
    changes = pd.concat(all_changes, ignore_index=True)
    changes.to_csv(output_dir / 'changes.csv', index=False, encoding='utf-8-sig')
    print('\n===== 相对正常门控的改动 =====')
    print(changes.to_string(index=False))
    print('结果目录：', output_dir)
    print('关闭知识只是在这个已训练模型上消融，不等于重新训练的原 Lexical。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
