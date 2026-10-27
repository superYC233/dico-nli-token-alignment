import pandas as pd

from experiments.deberta_feature_fusion import config as cfg
from experiments.deberta_lexical.data import (
    validate_groups, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)


def read_data():
    # 只读取官方数据；复用原版分组、分词和20维特征。
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev
