from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
QWEN_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = QWEN_DIR.parent.parent
RUN_NAME = 'run01_reason02'
MODE = 'C'
SEED = 42
MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
MODEL_ROOT = Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_reason')
RESULTS_ROOT = QWEN_DIR / 'outputs/qwen3_0p6b_reason'
PREPARED_DIR = RESULTS_ROOT / 'prepared_C'

REASON_WEIGHT = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_LOSS_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 16
MAX_LENGTH = 384
BF16 = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
REASON_PATH = PROJECT_ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/reason_prepared_train_only/train_with_reasons.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
LABEL_MAP = dict(zip(FULL_LABELS, LABELS))

# 与原C+r16的短Prompt逐字符保持一致。
LABEL_PROMPT = """Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
"""

# 独立的解释任务，只输入两段原文，不提供关系金标。
REASON_PROMPT = """Compare the meanings of Text 1 and Text 2.
Explain their shared content and any information-changing difference.
Use only the given text. Do not invent missing context.
Write a short explanation without giving an F, B, E, or N label.

Text 1: {text1}
Text 2: {text2}
Explanation:
"""
