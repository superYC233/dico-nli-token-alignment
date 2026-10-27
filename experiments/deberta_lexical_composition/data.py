import pandas as pd

from experiments.deberta_lexical_composition import config as cfg
# 直接复用原数据类和特征，避免改变分组、分词或反向行的特征。
from experiments.deberta_lexical.data import (
    validate_groups, build_feature_tensor, save_feature_table, DiCoDataset, PairDataset,
)


def read_data():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False).reset_index(drop=True)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False).reset_index(drop=True)
    validate_groups(train)
    validate_groups(dev)
    return train, dev
