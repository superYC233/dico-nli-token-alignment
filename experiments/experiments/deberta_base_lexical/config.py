from pathlib import Path

from project_paths import get_model_cache_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

RUN_NAME = 'run01_raw_base'
SEED = 42
# 相对无增强0.930569版，只更换这里的预训练初始化。
MODEL_NAME = 'microsoft/deberta-v3-base'

USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3

EPOCHS = 10
ROW_BATCH_SIZE = 16
PAIR_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
PAIR_WEIGHTS = [1.0, 1.0, 1.25]
PAIR_WEIGHT = 0.5
PAIR_LABEL_SMOOTHING = 0.05

TRAIN_PATH = PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
REFERENCE_DIR = PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_base_lexical')


def get_pretrained_path():
    # 读取本机已有缓存的main版本，直接传本地路径，不触发远程adapter探测。
    repository = Path(get_model_cache_dir()) / ('models--' + MODEL_NAME.replace('/', '--'))
    revision = (repository / 'refs/main').read_text(encoding='utf-8').strip()
    return repository / 'snapshots' / revision


def get_output_dirs(run_name=RUN_NAME):
    name = f'lexical_seed_{SEED}'
    return MODEL_ROOT / run_name / name, EXPERIMENT_DIR / 'outputs' / run_name / name
