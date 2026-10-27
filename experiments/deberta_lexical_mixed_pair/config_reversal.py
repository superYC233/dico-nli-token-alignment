from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run07_reversal_kl_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

# 原run01_mixed_pair架构和训练设置，只替换Pair CE。
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
ROW_LABEL_SMOOTHING = 0.0
CONSISTENCY_WEIGHT = 0.5
SELECT_BY = 'independent'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
