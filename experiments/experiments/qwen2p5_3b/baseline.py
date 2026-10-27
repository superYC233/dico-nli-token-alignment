from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from data_utils import encode_prompt, tokenizer, get_datasets, load_data, label_map
from evaluation_functions import score_files

def predict_batch(prompts, model):
    inputs = encode_prompt(prompts).to('cuda')

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False, # 不进行随机采样
            num_beams=1,
            max_new_tokens=1,
            pad_token_id= tokenizer.pad_token_id
        )

        # 对 Qwen 这种 decoder-only 模型：
        # output_ids 包含“原输入 + 新生成内容”，仍是整数编号
        # 形状为 [n, L + K]，K 是生成部分的长度，最多16

        # shape[1]为补齐后的长度L
        prompt_length = inputs['input_ids'].shape[1]

        new_ids = output_ids[:, prompt_length:]  # [n,k]

        texts = tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True
        )

        predictions = [text.strip() for text in texts]

        return predictions

def evaluate(model,batch_size=8, split='dev'):
    dataset = load_data()
    dataset = get_datasets(dataset)
    split_dataset = dataset[split]
    records = []

    for start in tqdm(range(0, len(split_dataset), batch_size)):
        batch_data = split_dataset[start:start+batch_size]
        prompts = batch_data['prompt']
        predictions = predict_batch(prompts, model)

        for instance_id, pair_id, gold_label, prediction in zip(
            batch_data['instance_id'],
            batch_data['pair_id'],
            batch_data['label'],
            predictions
        ):
            records.append({
                'instance_id': instance_id,
                'pair_id': pair_id,
                'gold_label': gold_label,
                'prediction': prediction
            })

    return records

def save_predictions(records, output_dir):
    records = pd.DataFrame(records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records.to_csv(f'{output_dir}/predictions.csv', index=False, encoding='utf-8-sig')

def score_predictions(records, output_dir):
    predictions = [r['prediction'] for r in records]
    prediction_label_map = {v : k for k, v in label_map.items()}

    for prediction in predictions:
        if prediction not in ('F', 'N', 'E', 'B'):
            print(prediction)
            return None

    df = pd.DataFrame({
        'instance_id': [r['instance_id'] for r in records],
        'label' : [prediction_label_map[r['prediction']] for r in records],
    })
    path = f'{output_dir}/official_predictions.csv'
    df.to_csv(path, index=False)

    report = score_files(
        '../../data/dev/dico_nli_dev_track1_reference.csv',
        path,
        output_dir=output_dir,
    )
    print('Weighted F1:', report.weighted_f1)
    return report

if __name__ == '__main__':
    model_path = 'D:/LLM/model/Qwen2.5-3B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto').to('cuda')
    model.eval()

    records = evaluate(model)
    save_predictions(
        records,
        'outputs/zero_shot/run01'
    )

    report = score_predictions(
        records,
        'outputs/zero_shot/run01'
    )

    if report is not None:
        print('SoftCons:', report.soft_cons)
        print('HardCons:', report.hard_cons)