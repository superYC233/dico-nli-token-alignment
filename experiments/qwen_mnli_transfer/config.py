from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUN_NAME = 'run02_fast'
SEED = 42

# 先和原 C+r16 对照，因此默认使用原来的 0.6B Base，不引入新模型变量。
MODEL_PATH = Path('D:/LLM/model/Qwen3-0.6B-Base')
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 512

# 第一阶段：只使用 MNLI 的 train；验证样本也从 train 中另外划出。
MNLI_DATASET = 'nyu-mll/multi_nli'
MNLI_REVISION = 'main'
MNLI_TRAIN_ROWS = 30000
MNLI_EVAL_ROWS = 1000
MNLI_EPOCHS = 1
MNLI_LEARNING_RATE = 1e-4
MNLI_BATCH_SIZE = 16
MNLI_GRAD_ACCUMULATION = 1  # 有效 batch 仍为 16，减少小批次前向/反向次数。
MNLI_EVAL_BATCH_SIZE = 16
# 复用上一轮已准备好的同一批数据，不重新下载、筛选或抽样。
# 如果改变 MNLI 样本数 / Prompt / seed 等准备配置，也要换这个名字。
MNLI_DATA_RUN_NAME = 'run01'

# 第二阶段：原来的 C+r16、短 Prompt、completion-only SFT。
MODE = 'C'
DICO_EPOCHS = 5
DICO_LEARNING_RATE = 1e-4
DICO_BATCH_SIZE = 16
DICO_GRAD_ACCUMULATION = 1
PREDICT_BATCH_SIZE = 16
USE_MNLI_ADAPTER = True

# 数据缓存放短路径，避免 Windows 缓存文件名过长。
CACHE_DIR = PROJECT_ROOT / '.hf_qmnli'
PREPARED_DIR = EXPERIMENT_DIR / 'outputs' / MNLI_DATA_RUN_NAME / 'prepared_mnli'
OUTPUT_ROOT = Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_mnli') / RUN_NAME
MNLI_OUTPUT_DIR = OUTPUT_ROOT / 'mnli'
MNLI_ADAPTER_DIR = MNLI_OUTPUT_DIR / 'final_adapter'
DICO_VARIANT = 'from_mnli' if USE_MNLI_ADAPTER else 'from_base'
DICO_OUTPUT_DIR = OUTPUT_ROOT / f'dico_{DICO_VARIANT}' / MODE
BEST_ADAPTER_DIR = DICO_OUTPUT_DIR / 'best_adapter'
RESULTS_DIR = EXPERIMENT_DIR / 'outputs' / RUN_NAME / f'dico_{DICO_VARIANT}' / MODE
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

# MNLI 保留三分类，不把 entailment 直接当作 DiCo 的 F。
NLI_PROMPT = '''Classify the relation from Text 1 (premise) to Text 2 (hypothesis).

entailment: If Text 1 is true, Text 2 must be true.
neutral: Text 1 does not determine whether Text 2 is true or false.
contradiction: If Text 1 is true, Text 2 must be false.

Output exactly one label: entailment, neutral, or contradiction.

Text 1: {text1}
Text 2: {text2}
Answer:
'''

# 与原 Qwen3-0.6B-Base 的短 Prompt 完全相同，不使用聊天模板。
DICO_PROMPT = '''Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
'''
