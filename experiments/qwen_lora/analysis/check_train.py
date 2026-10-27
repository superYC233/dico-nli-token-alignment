
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from baseline import evaluate, save_predictions
from analyze_errors import analyze_errors

if __name__ == '__main__':
    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'
    adapter_path = 'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_lora/run01'

    # 加载原始 Base，再加载已经训练好的 run01 Adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto'
    )

    model = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_path
    ).to('cuda')

    model.eval()

    # 预测训练集，不进行训练或参数更新
    records = evaluate(
        model,
        batch_size=32,
        split='train'
    )

    # 使用单独目录，保留之前的 Dev 预测结果
    output_dir = (
        QWEN_DIR
        / 'outputs/qwen3_0p6b_lora/run01/train_analysis'
    )

    save_predictions(records, output_dir)

    print('保存样本数：', len(records))
    print('预测文件：', output_dir / 'predictions.csv')