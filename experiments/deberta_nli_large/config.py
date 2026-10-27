from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_large_mixed_pair'
SEED = 42

MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
# 直接使用之前下载好的 Hugging Face 缓存，不重新复制一份权重。
MODEL_CACHE_DIR = Path('D:/BaiduNetdiskDownload/PretrainedModel/models')
MODEL_REVISION = 'b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'
MODEL_PATH = (
    MODEL_CACHE_DIR
    / 'models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
    / 'snapshots'
    / MODEL_REVISION
)

# 沿用 run01_mixed_pair 的参数，不沿用后来降低学习率的实验。
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEARNING_RATE = 2e-5
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

EPOCHS = 10
MAX_LENGTH = 128
# 一组是一对 F/B/E，或者一条 N；8 组最多 16 行。
GROUP_BATCH_SIZE = 8
DEV_BATCH_SIZE = 16

# large 使用混合精度和梯度检查点，降低显存占用；参数本身仍为 float32。
BF16 = True
GRADIENT_CHECKPOINTING = True

ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5

# 保存模型直接回答最好的那一轮，同时报告它的 pair-aware 成绩。
SELECT_BY = 'independent'

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = [
    'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT',
    'EQUIVALENCE', 'NEGATIVE_OTHER',
]
LABEL_TO_ID = {label: index for index, label in enumerate(FULL_LABELS)}

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_nli_large') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
