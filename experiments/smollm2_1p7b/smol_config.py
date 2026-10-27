from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = PROJECT_ROOT / 'experiments/qwen_lora'
# Windows 的 datasets 锁文件会重复拼接缓存路径，目录需要足够短。
DATA_CACHE_DIR = PROJECT_ROOT / '.hf_smol'

# 下载的是 Base 版；与原 Qwen 一样使用裸 Prompt，不套 chat template。
MODEL_ID = 'HuggingFaceTB/SmolLM2-1.7B'
MODEL_PATH = Path('D:/LLM/model/SmolLM2-1.7B')

MODE = 'C'
SEED = 42
MAX_LENGTH = 512
PREDICT_BATCH_SIZE = 16

OUTPUT_DIR = Path(f'D:/LLM/train_model/DiCo-NLI/smollm2_1p7b/run01/{MODE}')
BEST_DIR = OUTPUT_DIR / 'best_adapter'
RESULTS_DIR = Path(__file__).resolve().parent / f'outputs/run01/{MODE}'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'

LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}

# 固定原来的短 Prompt，避免以后修改 Qwen 文件时影响这个实验。
PROMPT_TEMPLATE = '''Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
'''
