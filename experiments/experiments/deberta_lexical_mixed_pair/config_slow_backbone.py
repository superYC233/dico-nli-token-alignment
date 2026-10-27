from pathlib import Path

from experiments.deberta_lexical_mixed_pair.config import (
    PROJECT_ROOT, EXPERIMENT_DIR, MODEL_NAME,
    TRAIN_PATH, DEV_PATH, GOLD_PATH, LABELS, FULL_LABELS, LABEL_TO_ID,
)


# 单独配置本次实验；原 config.py 不用修改。
RUN_NAME = 'run04_slow_backbone'
SEED = 42

USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
# 以 run01_mixed_pair 为对照：只降低预训练主干的学习率。
BACKBONE_LEARNING_RATE = 5e-6
LEARNING_RATE = 2e-5  # CLS 分类头 + Token Alignment
LEXICAL_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

EPOCHS = 10
MAX_LENGTH = 128
GROUP_BATCH_SIZE = 8  # 最多16行，原始 Train 每轮各出现一次。
DEV_BATCH_SIZE = 16
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_LABEL_SMOOTHING = 0.05
PAIR_WEIGHT = 0.5

# 本次目标是模型直接回答更准，同时记录该权重的结构分数。
SELECT_BY = 'independent'

MODEL_DIR = Path('D:/model/SEAMEVAL/deberta_lexical_mixed_pair') / RUN_NAME
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
