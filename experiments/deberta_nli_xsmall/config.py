from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_xsmall_mixed_pair'
SEED = 42

# 先运行 download_model.py；训练只读取这份本地模型。
MODEL_NAME = 'cross-encoder/nli-deberta-v3-xsmall'
MODEL_PATH = Path('D:/LLM/model/nli-deberta-v3-xsmall')

# 对照未加OOF的 run01_mixed_pair，仅替换预训练主干和对应Tokenizer。
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

EPOCHS = 10
MAX_LENGTH = 128
# 一组是一对 F/B/E，或者一条 N；8组最多16行。
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5

# 按独立预测选最佳；每轮也保存pair-aware的成绩和预测CSV。
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

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_nli_xsmall') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
