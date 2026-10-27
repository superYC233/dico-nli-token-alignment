"""只检查数据与特征，不加载真实预训练模型、不训练。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.deberta_lexical_stage_probe.data import read_data, build_feature_tensor, PairDataset
from experiments.deberta_lexical_stage_probe.lexical_features import FEATURE_NAMES, build_lexical_features


def main():
    train, dev = read_data()
    train_features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    pairs = PairDataset(train, tokenizer=None, features=train_features)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 20
    print('Train / Dev：', len(train), '/', len(dev))
    print('Real Pair 组数：', len(pairs))
    print('Train 特征形状：', tuple(train_features.shape))
    print('Dev 特征形状：', tuple(dev_features.shape))

    # 改动标签和编号不应改变特征，证明特征只依赖 text1/text2。
    changed = train.copy()
    changed['label'] = 'EQUIVALENCE'
    changed['pair_id'] = 'not_used'
    changed['instance_id'] = 'not_used'
    assert torch.equal(train_features, build_feature_tensor(changed))

    print('\n示例：a black car / a car（这里只展示特征，不用规则给标签）')
    values = build_lexical_features('a black car', 'a car')
    for name, value in zip(FEATURE_NAMES, values):
        print(f'{name}: {value:.4f}')
    print('\n检查通过。正式训练请运行 train_deberta_stage_probe.py。')


if __name__ == '__main__':
    main()
