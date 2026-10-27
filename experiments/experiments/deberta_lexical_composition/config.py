from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_composition'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# 本次只增加对齐比较后的组合层，保留原20维Lexical。
USE_COMPOSITION = True
COMPOSITION_LAYERS = 2
COMPOSITION_KERNEL_SIZE = 3
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 沿用官方3042行、0.930569那版Lexical的参数。
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

MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_lexical_composition')
REFERENCE_DIR = (
    PROJECT_ROOT / 'experiments/deberta_lexical/outputs/'
    'run02_branch_lr/lexical_seed_42'
)
REFERENCE_PREDICTIONS = REFERENCE_DIR / 'epoch_3/predictions_with_scores.csv'


def get_output_dirs(use_composition=USE_COMPOSITION):
    version = 'composition' if use_composition else 'lexical'
    name = f'{version}_seed_{SEED}'
    model_dir = MODEL_ROOT / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir
