from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_child_d_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

# 在原混合版上只增加Child-Tuning-D；不使用上一版pooler。
USE_CHILD_TUNING = True
RESERVE_P = 0.20
FISHER_BATCH_SIZE = 16
FISHER_GRAD_CLIP = 1.0
FISHER_SEED = SEED + 100

USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 10
MAX_LENGTH = 128
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5
SELECT_BY = 'independent'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_child_tuning') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
