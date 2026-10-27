from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

# OOF评分可以复用；更换最终训练的RUN_NAME，不需要重新跑三折。
OOF_RUN_NAME = 'oof01_train_only'
RUN_NAME = 'run06_oof_hard'
WEIGHT_MODE = 'hard'  # hard=难例加权；random=同类别随机加权对照；uniform=原始等权。
SEED = 42

MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# 三折模型均从NLI初始化训练固定3轮，不读Dev，不按留出结果挑轮次。
# 每折用约2/3的Train，3轮的更新次数约等于完整Train训练2轮。
OOF_FOLDS = 3
OOF_EPOCHS = 3

# 在每种分组类型内，最多选25%的组；只有OOF错判的边界组才候选。
HARD_FRACTION = 0.25
HARD_MULTIPLIER = 2.0
# 复核难例表后，可在这里填入不希望额外加权的pair_id；原始行仍参与训练。
EXCLUDE_PAIR_IDS = []

EPOCHS = 10
MAX_LENGTH = 128
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5
SELECT_BY = 'independent'

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}

OOF_DIR = EXPERIMENT_DIR / 'outputs' / OOF_RUN_NAME
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
