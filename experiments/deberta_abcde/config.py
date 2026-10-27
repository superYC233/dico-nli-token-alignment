from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

# 再跑一批实验时，改成 run02，避免覆盖已有结果。
RUN_NAME = 'run01'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 以下参数沿用原 train.py，不照搬 Qwen 的训练参数。
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
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
REVERSE_LABEL = {
    FULL_LABELS[0]: FULL_LABELS[1],
    FULL_LABELS[1]: FULL_LABELS[0],
    FULL_LABELS[2]: FULL_LABELS[2],
    FULL_LABELS[3]: FULL_LABELS[3],
}


def get_output_dirs(mode):
    name = f'{mode}_seed_{SEED}'
    model_dir = PROJECT_ROOT / 'model/deberta_abcde' / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir
