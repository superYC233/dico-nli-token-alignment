"""接着run01最佳N头模型微调；旧config.py和旧模型不动。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RUN_NAME = 'run03_top2_row_only'
SEED = 42
SOURCE_MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_frozen_n_head/run01_n_head')
SOURCE_RESULTS_DIR = HERE / 'outputs/run01_n_head'
TEACHER_CACHE = SOURCE_RESULTS_DIR / 'feature_cache/train.pt'

# 12层中只更新最后2层。原CLS分类器和Token Alignment参数保持冻结，
# 但允许梯度穿过这些固定运算，传回顶部编码层。
UNFREEZE_LAST_LAYERS = 2
ENCODER_LEARNING_RATE = 2e-6
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EPOCHS = 5
TRAIN_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
DEVICE = 'cuda'

# 一次前向同时计算两项，没有Pair-only阶段，也不添加NN样本。
# Row CE使用四类金标，不额外加N权重。
KEEP_WEIGHT = 0.0
KEEP_TEMPERATURE = 2.0
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94

DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_frozen_n_head') / RUN_NAME
RESULTS_DIR = HERE / 'outputs' / RUN_NAME
