from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'run09_reason_all_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

# 原混合版架构与训练参数；只增加token依据监督。
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
EVIDENCE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 10
MAX_LENGTH = 128
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5
EVIDENCE_WEIGHT = 0.1
SELECT_BY = 'independent'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
LABEL_TO_ID = dict(zip(FULL_LABELS, range(4)))
TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
ANNOTATION_PATH = EXPERIMENT_DIR / 'reason_annotations.csv'
PREPARED_DIR = EXPERIMENT_DIR / 'outputs/reason_prepared_train_only'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair')
