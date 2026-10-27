from pathlib import Path

from experiments.qwen_lora.reason_sft.config import (
    TRAIN_PATH, DEV_PATH, GOLD_PATH, REASON_PATH, LABELS, FULL_LABELS, LABEL_MAP,
    LABEL_PROMPT, REASON_PROMPT,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUN_NAME = 'run01_c_reason02'
MODE = 'C'
SEED = 42
MODEL_PATH = 'D:/LLM/model/Qwen2.5-3B-Instruct'
MODEL_ROOT = Path('D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_reason')
RESULTS_ROOT = EXPERIMENT_DIR / 'outputs'
PREPARED_DIR = RESULTS_ROOT / 'prepared_C'

REASON_WEIGHT = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 5
# 12GB显存：实际批次4，累积4次，有效batch仍为16。
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_LOSS_BATCH_SIZE = 4
PREDICT_BATCH_SIZE = 8
MAX_LENGTH = 384
BF16 = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def build_prompt_templates(tokenizer):
    # 两个任务均使用Instruct的聊天模板；答案从assistant起始位置生成。
    templates = []
    for prompt in [LABEL_PROMPT, REASON_PROMPT]:
        templates.append(tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user', 'content': prompt.rstrip()}],
            tokenize=False, add_generation_prompt=True,
        ))
    return templates
