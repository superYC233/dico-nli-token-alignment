
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from peft import PeftModel
from transformers import AutoModelForCausalLM
from baseline import evaluate, save_predictions, score_predictions

model_path = 'D:/LLM/model/Qwen3-0.6B-Base'

model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto')
model = PeftModel.from_pretrained(
    model = model,
    model_id='D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_lora/run04_random_n'
).to('cuda')

model.eval()

output_dir = 'outputs/qwen3_0p6b_lora/run04_random_n'

records = evaluate(model)
save_predictions(
    records,
    output_dir
)

report = score_predictions(
    records,
    output_dir
)

if report is not None:
    print('SoftCons:', report.soft_cons)
    print('HardCons:', report.hard_cons)