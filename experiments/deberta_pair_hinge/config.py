from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# 右键运行默认先试hinge16。ce3是相同混合流程的对照。
RUN_NAME = 'run01'
PAIR_LOSS = 'hinge16'  # 'hinge16' 或 'ce3'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

EPOCHS = 10
GROUP_BATCH_SIZE = 8  # 一组为两行Pair或一行N；每步最多16行。
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]  # FB / BF / EE金标组权重；两臂相同。
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05  # 只用于ce3；hinge没有label smoothing。
HINGE_MARGIN = 1.0

# 沿用旧选模口径，每轮同时记录同一权重的独立分数。
SELECT_BY = 'pair_aware'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]

# 直接使用0.938145那次的3290行快照；不重新构造、不添加新数据。
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
TRAIN_PATH = REFERENCE_DIR / 'training_data.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_pair_hinge')


def get_run_name(run_name=RUN_NAME, pair_loss=PAIR_LOSS):
    return f'{run_name}_mixed_{pair_loss}'
