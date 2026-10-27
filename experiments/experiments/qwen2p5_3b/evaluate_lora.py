from peft import PeftModel
from transformers import AutoModelForCausalLM
from baseline import evaluate, save_predictions, score_predictions

model_path = 'D:/LLM/model/Qwen2.5-3B-Instruct'

model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto')
model = PeftModel.from_pretrained(
    model = model,
    model_id='D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_lora/run01'
).to('cuda')

model.eval()

output_dir = 'outputs/lora/run01'

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