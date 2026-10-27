"""准备 WordNet、检查全部数据和特征；不加载 DeBERTa 权重，不训练。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from experiments.deberta_wordnet import config as cfg
from experiments.deberta_wordnet.data import (
    read_data, build_feature_tensor, save_feature_table, summarize_wordnet, PairDataset,
)
from experiments.deberta_wordnet.lexical_features import build_lexical_features
from experiments.deberta_wordnet.wordnet_features import FEATURE_NAMES, WORDNET_FEATURE_NAMES, build_features
from experiments.deberta_wordnet.wordnet_resource import ensure_wordnet, resource_info


def main():
    ensure_wordnet(allow_download=True)
    print('WordNet 资源：', resource_info())
    train, dev = read_data()
    train_features, train_audit = build_feature_tensor(train, return_audit=True)
    dev_features, dev_audit = build_feature_tensor(dev, return_audit=True)
    control = build_feature_tensor(train, use_wordnet_features=False)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 32
    assert torch.equal(train_features[:, :20], control[:, :20])
    assert torch.count_nonzero(control[:, 20:]).item() == 0
    assert len(PairDataset(train, None, train_features)) == 1282
    for index, row in enumerate(train.itertuples()):
        assert train_features[index, :20].tolist() == torch.tensor(
            build_lexical_features(row.text1, row.text2), dtype=torch.float32,
        ).tolist()

    print('Train / Dev：', len(train), '/', len(dev))
    print('特征形状：', tuple(train_features.shape), '/', tuple(dev_features.shape))
    coverage = pd.DataFrame([
        summarize_wordnet(train, train_features, train_audit, 'train'),
        summarize_wordnet(dev, dev_features, dev_audit, 'dev'),
    ])
    print(coverage.to_string(index=False))
    output_dir = cfg.EXPERIMENT_DIR / 'prepared'
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / 'wordnet_coverage.csv', index=False, encoding='utf-8-sig')
    save_feature_table(train, train_features, output_dir / 'train_features.csv')
    train_audit.to_csv(output_dir / 'train_wordnet_audit.csv', index=False, encoding='utf-8-sig')
    dev_audit.to_csv(output_dir / 'dev_wordnet_audit.csv', index=False, encoding='utf-8-sig')

    print('\n示例只展示词义特征，不用词典直接指定任务标签：')
    for first, second in [('black car', 'black vehicle'), ('car', 'automobile'), ('hot', 'cold')]:
        values, _ = build_features(first, second)
        print(first, '/', second)
        print({name: value for name, value in zip(WORDNET_FEATURE_NAMES, values[20:]) if value})
    print('检查通过。正式训练请右键运行 train_deberta_wordnet.py。')


if __name__ == '__main__':
    main()
