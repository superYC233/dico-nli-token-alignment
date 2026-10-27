from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_native_ce_kl_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

# 原生四分类cross-encoder，不加Alignment、Lexical或三组合Pair CE。
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
CONSISTENCY_WEIGHT = 0.5
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
EPOCHS = 10
MAX_LENGTH = 128
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

# 每1/4轮评估一次；连续8次未刷新独立F1则停止，最多训练10轮。
EVAL_EPOCH_FRACTION = 0.25
EARLY_STOPPING_PATIENCE = 8
SELECT_BY = 'independent'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_reversal_kl') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
