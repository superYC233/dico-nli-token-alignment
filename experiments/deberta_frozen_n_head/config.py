"""只修改本文件中的RUN_NAME和本次小头参数，保留旧模型。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RUN_NAME = 'run01_n_head'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
REFERENCE_MODEL = Path(
    'D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged/best_model.pth'
)
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'

# 本次只训练这个小头，整个原模型保持eval和冻结状态。
HEAD_HIDDEN_SIZE = 64
EPOCHS = 30
TRAIN_BATCH_SIZE = 64
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
N_POS_WEIGHT = 1.0  # 普通二元CE，先不额外提高N权重。

TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94
DEVICE = 'cuda'

DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_frozen_n_head') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME
