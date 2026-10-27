"""一般只需要在这里修改路径；本实验不训练、不改动原 Adapter。"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
QWEN_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = QWEN_DIR.parent.parent

RUN_NAME = 'run01'
OUTPUT_DIR = QWEN_DIR / 'outputs/retrieval_prompt' / RUN_NAME
PREPARED_DIR = OUTPUT_DIR / 'prepared'
RESULTS_DIR = OUTPUT_DIR / 'evaluation'

MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
ADAPTER_PATH = (
    'D:/LLM/train_model/DiCo-NLI/'
    'qwen3_0p6b_abc/run04_r16/C/best_adapter'
)

# 可以改成本地 BGE 目录。首次运行没有缓存时会从 Hugging Face 下载。
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
EMBEDDING_DEVICE = 'cpu'       # 检索准备在 CPU 上完成，不占 Qwen 的显存。
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_LENGTH = 512

NUM_EXAMPLES = 2
SEED = 42
PREDICT_BATCH_SIZE = 16
MAX_PROMPT_TOKENS = 2048      # 超过就报错，不会悄悄截断例子或问题。
METHODS = ('baseline', 'random', 'retrieval')

TRAIN_FILE = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_FILE = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
REFERENCE_FILE = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
HISTORICAL_BASELINE = (
    QWEN_DIR / 'outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
)

LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}

