"""复用原训练入口；每次只跑一个种子、一个版本。"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
import transformers

from experiments.deberta_lexical_reliability import config as cfg
from experiments.deberta_lexical_reliability.protocol import ensure_protocol


def bind_original_code():
    # 原模块没有顶层加载模型的副作用，可以安全导入；不导入旧版训练脚本。
    from experiments.deberta_lexical import data, evaluation, train_deberta_lexical
    # 只在当前子进程换配置对象；不修改原 config.py 文件。
    data.cfg = cfg
    evaluation.cfg = cfg
    train_deberta_lexical.cfg = cfg
    return train_deberta_lexical


def validate_completed(results_dir, model_dir, protocol_hash):
    marker = results_dir / 'completed.json'
    if not marker.is_file():
        return False
    info = json.loads(marker.read_text(encoding='utf-8'))
    if info['protocol_sha256'] != protocol_hash:
        raise ValueError('已完成结果来自不同配方。')
    history = pd.read_csv(results_dir / 'epoch_summary.csv')
    paired = history[history['method'] == 'pair_aware']
    if sorted(paired['epoch'].tolist()) != list(range(1, cfg.EPOCHS + 1)):
        raise ValueError('已完成标记存在，但轮次记录不完整。')
    best = json.loads((model_dir / 'best_metrics.json').read_text(encoding='utf-8'))
    selected = paired.sort_values(['weighted_f1', 'epoch'], ascending=[False, True]).iloc[0]
    if best['best_epoch'] != int(selected['epoch']) or abs(best['weighted_f1'] - selected['weighted_f1']) > 1e-12:
        raise ValueError('最佳轮次记录不一致。')
    prediction = results_dir / f"epoch_{best['best_epoch']}" / 'predictions_with_scores.csv'
    if not prediction.is_file() or not (model_dir / 'best_model.pth').is_file():
        raise FileNotFoundError('已完成运行缺少最佳预测或权重。')
    return True


def main(seed, arm):
    if seed not in cfg.SEEDS or arm not in ('baseline', 'lexical'):
        raise ValueError('仅运行预先指定的 seed 7/21 与 baseline/lexical。')
    cfg.SEED = seed
    protocol_hash = ensure_protocol()
    model_dir, results_dir = cfg.get_output_dirs(arm == 'lexical')
    if validate_completed(results_dir, model_dir, protocol_hash):
        print('已完成且配方一致，跳过：', arm, seed)
        return
    if model_dir.exists() or results_dir.exists():
        raise FileExistsError('存在未完成的运行目录；为保护结果，不覆盖、不自动删除。')
    if shutil.disk_usage(cfg.MODEL_ROOT.anchor).free < 2 * 1024 ** 3:
        raise OSError('模型所在磁盘不足 2 GB，先释放空间或修改模型目录。')
    print(f'\n===== 配对复验：{arm}，seed={seed} =====', flush=True)
    print('模型目录：', model_dir, flush=True)
    print('结果目录：', results_dir, flush=True)
    training = bind_original_code()
    training.main(use_lexical_features=arm == 'lexical')
    info = {
        'seed': seed, 'arm': arm, 'protocol_sha256': protocol_hash,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'epochs': cfg.EPOCHS,
    }
    marker = results_dir / 'completed.json'
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    if not validate_completed(results_dir, model_dir, protocol_hash):
        raise RuntimeError('运行未通过完整性检查。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', choices=['baseline', 'lexical'], required=True)
    args = parser.parse_args()
    main(args.seed, args.arm)
