from experiments.deberta_lexical_mixed_pair import config_high_fusion as high
from experiments.deberta_lexical_mixed_pair import config_fusion as previous


PROJECT_ROOT = high.PROJECT_ROOT
EXPERIMENT_DIR = high.EXPERIMENT_DIR
RUN_NAME = 'fusion03_high_pair_qwen'
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME
DEV_PATH = high.DEV_PATH
GOLD_PATH = high.GOLD_PATH
LABELS = high.LABELS
FULL_LABELS = high.FULL_LABELS
MEMBERS = {**high.MEMBERS, 'qwen_r16': previous.MEMBERS['qwen_r16']}

# 每个模型先做四类softmax，再等权平均；不扫描温度或权重。
RECIPES = {
    'prob_top2': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935']},
    'prob_0938_qwen': {'mode': 'probability', 'members': ['relation_0938', 'qwen_r16']},
    'prob_0935_qwen': {'mode': 'probability', 'members': ['scope_0935', 'qwen_r16']},
    'prob_top2_qwen': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935', 'qwen_r16']},
    'prob_high3_qwen': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935', 'lexical_0930', 'qwen_r16']},
    'prob_high4_qwen': {'mode': 'probability', 'members': ['relation_0938', 'scope_0935', 'lexical_0930', 'original_0924', 'qwen_r16']},
}

BASELINE = 'relation_0938'
BOOTSTRAP_RECIPES = list(RECIPES)
BOOTSTRAP_ROUNDS = 5000
SEED = 42
