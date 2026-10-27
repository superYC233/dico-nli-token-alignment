from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# 沿用 0.930569 词汇分支版；只在 Pair 阶段加入少量四分类 CE。
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

# Pair 阶段：0.5 * Pair CE + 0.1 * 辅助四分类 CE。
# 辅助批次来自完整 Train（含 N），不使用 Dev，不保证每批都有 N。
# 改为 0.0 就恢复旧的纯 Pair CE；做新对照时同时更改 RUN_NAME。
PAIR_ROW_WEIGHT = 0.1
PAIR_ROW_BATCH_SIZE = 16
# 独立控制辅助批次抽样，不提前消耗原 Row / Pair 的 shuffle 随机数。
PAIR_ROW_SEED = SEED + 1000

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}


def get_output_dirs(use_lexical_features):
    version = 'lexical' if use_lexical_features else 'baseline'
    name = f'{version}_seed_{SEED}'
    model_dir = PROJECT_ROOT / 'model/deberta_lexical_joint' / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir

# 仅在训练后比较历史结果，不参与训练、特征或选模。
REFERENCE_RESULTS_DIR = (
    PROJECT_ROOT / 'experiments/deberta_lexical/outputs'
    / 'run02_branch_lr/lexical_seed_42'
)
