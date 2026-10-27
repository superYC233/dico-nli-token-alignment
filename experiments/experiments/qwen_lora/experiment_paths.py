"""
分类后的历史脚本仍从 qwen_lora 读取公共代码和 outputs
"""
import os
import sys
from pathlib import Path

QWEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN_DIR.parent.parent


def setup_experiment_paths(run_as_script=False):
    # analysis 和 ablation 中的少数函数仍被其他历史实验复用。
    for folder in [PROJECT_ROOT, QWEN_DIR / 'analysis', QWEN_DIR / 'ablation', QWEN_DIR]:
        path = str(folder)
        if path not in sys.path:
            sys.path.insert(0, path)

    # 只在右键运行脚本时设置目录，导入工具函数时不改变工作目录。
    if run_as_script:
        os.chdir(QWEN_DIR)
    return QWEN_DIR
