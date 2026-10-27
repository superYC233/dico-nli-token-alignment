from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'run01_original_recipe'
SEED = 42
MODEL_NAME = 'tasksource/ModernBERT-base-nli'
MODEL_REVISION = 'main'
MODEL_PATH = Path('D:/LLM/model/ModernBERT-base-nli')
MODEL_ROOT = Path('D:/model/SEAMEVAL/modernbert_nli_base')

# 先复用0.9248配方，只换主干；之后可在同一目录开启Lexical。
USE_LEXICAL_FEATURES = False
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3
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

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}


def get_output_dirs(use_lexical_features):
    version = 'lexical' if use_lexical_features else 'baseline'
    name = f'{version}_seed_{SEED}'
    return MODEL_ROOT / RUN_NAME / name, EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
