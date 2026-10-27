from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_NAME = 'run01'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# 只在原 20 维词汇分支上增加 8 维差异位置特征。
# False 时保留原 20 维分支，作为相同训练代码的 Lexical 对照。
USE_POSITION_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 与 0.930569 那版保持一致，不叠加 C 增强、WordNet 或新损失。
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

# 只在预测完成后用于比较，不参与特征、训练或解码。
LEXICAL_REFERENCE_PATH = (
    PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr'
    / 'lexical_seed_42/epoch_3/predictions_with_scores.csv'
)


def get_output_dirs(use_position_features):
    version = 'altn' if use_position_features else 'lexical_control'
    name = f'{version}_seed_{SEED}'
    model_dir = PROJECT_ROOT / 'model/deberta_altn' / RUN_NAME / name
    results_dir = EXPERIMENT_DIR / 'outputs' / RUN_NAME / name
    return model_dir, results_dir
