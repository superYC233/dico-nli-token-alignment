from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_feature_fusion'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# 原对齐计算保留，改为先融合特征，再由一个分类头判断。
ALIGNMENT_DIMENSION = 128
ALIGNMENT_TEMPERATURE = 0.1
LEXICAL_HIDDEN_SIZE = 32
FUSION_HIDDEN_SIZE = 128
DROPOUT = 0.2

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 数据、训练目标和参数沿用官方3042行的0.930569 Lexical配方。
EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
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

MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_feature_fusion')
REFERENCE_DIR = (
    PROJECT_ROOT / 'experiments/deberta_lexical/outputs/'
    'run02_branch_lr/lexical_seed_42'
)
REFERENCE_PREDICTIONS = REFERENCE_DIR / 'epoch_3/predictions_with_scores.csv'


def get_output_dirs():
    name = f'feature_fusion_seed_{SEED}'
    model_dir = MODEL_ROOT / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir
