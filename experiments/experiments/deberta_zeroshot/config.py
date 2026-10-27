from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run02'
SEED = 42
MODEL_NAME = 'D:/BaiduNetdiskDownload/PretrainedModel/models/deberta-v3-base-zeroshot-v2.0'

# 第一版只增加词汇特征，不同时叠加 A–E 数据增强。
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
# 新初始化的词汇分支使用单独学习率；原模型仍为下面的 2e-5。
LEXICAL_LEARNING_RATE = 1e-3

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 沿用 0.9248 那版 train.py 的参数。
EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}


def get_output_dirs(use_lexical_features):
    version = 'lexical' if use_lexical_features else 'baseline'
    name = f'{version}_seed_{SEED}'
    model_dir = PROJECT_ROOT / 'model/deberta_zeroshot' / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir
