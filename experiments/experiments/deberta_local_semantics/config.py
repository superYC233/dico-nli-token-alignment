from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RUN_NAME = 'run01_local_semantics'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_local_semantics')

# 精确复用0.938145当次的3290行，不重新生成旧增强。
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
BASE_TRAIN_PATH = REFERENCE_DIR / 'training_data.csv'
OFFICIAL_TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
ANNOTATION_PATH = HERE / 'annotation_sources.csv'
PREPARED_DIR = HERE / 'prepared'

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

# 只在Row阶段、有局部标注的行上加此项；不额外抽样、不增加前向。
LOCAL_WEIGHT = 0.2
LOCAL_HIDDEN_SIZE = 128
LOCAL_LEARNING_RATE = 3e-4  # 只用于新初始化的局部头；原模型保持2e-5。
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
LOCAL_TO_ID = {label: index for index, label in enumerate(LABELS)}
INVERSE = {'F': 'B', 'B': 'F', 'E': 'E', 'N': 'N'}


def get_output_dirs(run_name=RUN_NAME):
    return MODEL_ROOT / run_name, HERE / 'outputs' / run_name
