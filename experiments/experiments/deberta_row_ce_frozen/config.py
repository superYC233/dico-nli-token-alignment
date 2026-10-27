from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

# 代码复用，只切换下面两项；每次 RUN_NAME 必须是尚未训练过的新名字。
# 第一次：run02_scope + scope；对照：run03_copy + copy。
RUN_NAME = 'run03_copy'
AUGMENT_MODE = 'copy'  # scope=最小对照增强；copy=同数量同类别复制；none=原始 Train。
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
# 沿用刚跑过的冻结版；本次只关闭 Pair CE。
FREEZE_EMBEDDINGS = True
FREEZE_FIRST_N_LAYERS = 6
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.0

# 一个组是一对 F/B/E，或者一条 N；8 组最多 16 行。
# 原始数据 1760 组，每轮 220 步；增强/复制后 1880 组，每轮 235 步。
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

# 本次先要求模型能够独立四分类；两种解码成绩仍然每轮都报告。
# 全程固定这个选择规则。若以后改为 'pair_aware'，同时更改 RUN_NAME。
SELECT_BY = 'independent'

# 模型放 D 盘；每轮只保存小型预测 CSV，模型仅保留一份最佳权重。
MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_row_ce_frozen') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}
