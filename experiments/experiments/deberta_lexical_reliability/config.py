from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'run01'
SEEDS = [7, 21]
SEED = 42  # run_one.py 启动后设为该次指定的种子。

# 新模型统一保存到用户指定的 D 盘，代码和小型预测结果仍在项目目录。
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical_reliability')
RESULTS_ROOT = EXPERIMENT_DIR / 'outputs'

# 冻结 0.930569 的训练配方，不随其他实验的 config 改动。
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3
TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
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
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
BOOTSTRAP_REPEATS = 20000
BOOTSTRAP_SEED = 20261021


def get_output_dirs(use_lexical_features):
    arm = 'lexical' if use_lexical_features else 'baseline'
    name = f'{arm}_seed_{SEED}'
    return MODEL_ROOT / RUN_NAME / name, RESULTS_ROOT / RUN_NAME / name
