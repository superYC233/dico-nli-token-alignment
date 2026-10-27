
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

def build_prompt_original(text1, text2):
    prompt = f"""
Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
"""

    return prompt.strip() + '\n'