from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# 沿用原版模型，不增加词汇分支，不读取旧的 epoch_7 权重。
EPOCHS = 10
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5

# 一个组是一对 F/B/E，或者一条 N；8 组最多 16 行。
# 全部 1760 个训练组混合打乱，每轮约 220 步。
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

# 本次先要求模型能够独立四分类；两种解码成绩仍然每轮都报告。
# 全程固定这个选择规则。若以后改为 'pair_aware'，同时更改 RUN_NAME。
SELECT_BY = 'independent'

# 模型放 D 盘；每轮只保存小型预测 CSV，模型仅保留一份最佳权重。
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_joint_pair') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
