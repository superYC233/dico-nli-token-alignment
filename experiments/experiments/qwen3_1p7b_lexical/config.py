from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 放在短路径，避免 Windows 的 datasets 缓存锁文件超过路径长度限制。
DATA_CACHE_DIR = PROJECT_ROOT / '.hf_q17lex'

MODEL_ID = 'Qwen/Qwen3-1.7B'
MODEL_PATH = Path('D:/LLM/model/Qwen3-1.7B')

RUN_NAME = 'run01'
MODE = 'C'
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3
# 每次只投影少量受监督位置，仍对全部词表计算 CE。
LOSS_CHUNK_SIZE = 8
SEED = 42
MAX_LENGTH = 512
PREDICT_BATCH_SIZE = 16

OUTPUT_DIR = Path(f'D:/LLM/train_model/DiCo-NLI/qwen3_1p7b_lexical/{RUN_NAME}/{MODE}')
BEST_DIR = OUTPUT_DIR / 'best_adapter'
RESULTS_DIR = Path(__file__).resolve().parent / f'outputs/{RUN_NAME}/{MODE}'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

LABELS = ['F', 'B', 'E', 'N']

# 指令内容仍用原来的短 Prompt；外层由本模型自己的 chat template 包装。
TASK_PROMPT = '''Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
'''
