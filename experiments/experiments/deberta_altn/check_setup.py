"""只检查真实数据和特征，不加载大模型，不训练。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.deberta_altn.data import read_data, build_feature_tensor, PairDataset
from experiments.deberta_altn.altn_features import FEATURE_NAMES, build_features, alignment_audit
from experiments.deberta_altn.lexical_features import build_lexical_features


def main():
    train, dev = read_data()
    train_features, dev_features = build_feature_tensor(train), build_feature_tensor(dev)
    assert train_features.shape == (3042, 28) and dev_features.shape == (660, 28)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 28
    assert len(PairDataset(train, tokenizer=None, features=train_features)) == 1282
    assert torch.isfinite(train_features).all() and torch.isfinite(dev_features).all()
    for frame, features in [(train, train_features), (dev, dev_features)]:
        for index, row in frame.iterrows():
            original = torch.tensor(build_lexical_features(row['text1'], row['text2']))
            assert torch.equal(features[index, :20], original)
            reversed_features = torch.tensor(build_features(row['text2'], row['text1']))
            assert torch.equal(features[index, 20:24], reversed_features[24:28])
            assert torch.equal(features[index, 24:28], reversed_features[20:24])
    changed = train.copy()
    changed['label'] = 'EQUIVALENCE'
    changed['instance_id'] = 'not_used'
    changed['pair_id'] = 'not_used'
    assert torch.equal(train_features, build_feature_tensor(changed))
    print('Train / Dev：', len(train), '/', len(dev))
    print('真实 Pair：1282 组；特征：20 + 8 = 28 维。')
    print('原20维逐项一致、交换方向互换、特征不依赖标签/编号检查通过。')
    for first, second in [('train station', 'station'), ('red dog red', 'red dog'), ('dog bites man', 'man bites dog')]:
        print('\nText1 / Text2：', first, '/', second)
        print(json.dumps(alignment_audit(first, second), ensure_ascii=False))
        print('位置特征：', build_features(first, second)[20:])
    print('\n检查通过，可运行 train_deberta_altn.py。')


if __name__ == '__main__':
    main()
