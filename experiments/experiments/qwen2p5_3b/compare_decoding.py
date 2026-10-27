from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from data_utils import encode_prompt, tokenizer, get_datasets, load_data, label_map
from evaluation_functions import score_files


def predict_batch_with_scores(prompts, model):
    inputs = encode_prompt(prompts).to('cuda')

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            do_sample=False, # 不进行随机采样
            num_beams=1,
            max_new_tokens=1,
            pad_token_id= tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # 对 Qwen 这种 decoder-only 模型：
        # output_ids 包含“原输入 + 新生成内容”，仍是整数编号
        # 形状为 [n, L + K]，K 是生成部分的长度，最多16

        output_ids = result.sequences
        # shape[1]为补齐后的长度L
        prompt_length = inputs['input_ids'].shape[1]

        new_ids = output_ids[:, prompt_length:]  # [n,k]

        texts = tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True
        )

        predictions = [text.strip() for text in texts]

        labels = ['F', 'B', 'E', 'N']
        token_ids = [
            tokenizer.encode(label, add_special_tokens=False)
            for label in labels
        ]

        # 确认每个标签只有一个 token，才能直接取 [0]
        assert all(len(ids) == 1 for ids in token_ids)

        label_ids = [ids[0] for ids in token_ids]

        # result.scores[0]：第一个生成步骤中，所有样本的全词表分数，形状 [n, V]
        # n：当前批次的样本数；V：词表大小
        # [:, label_ids]
        # label_ids → 只取 F、B、E、N 对应的四列，形状变为 [n, 4]
        label_logits = result.scores[0][:, label_ids].float().cpu()
        return predictions, label_logits


def evaluate_with_scores(model, batch_size=16, split='dev'):
    dataset = load_data()
    dataset = get_datasets(dataset)
    split_dataset = dataset[split]
    records = []

    for start in tqdm(range(0, len(split_dataset), batch_size)):
        batch_data = split_dataset[start:start+batch_size]
        prompts = batch_data['prompt']
        predictions, label_logits = predict_batch_with_scores(prompts, model)

        for instance_id, pair_id, gold_label, prediction, scores in zip(
                batch_data['instance_id'],
                batch_data['pair_id'],
                batch_data['label'],
                predictions,
                label_logits.tolist()
        ):
            records.append({
                'instance_id': instance_id,
                'pair_id': pair_id,
                'gold_label': gold_label,
                'prediction': prediction,
                'logit_F': scores[0],
                'logit_B': scores[1],
                'logit_E': scores[2],
                'logit_N': scores[3],
            })

    return records

if __name__ == '__main__':
    model_path = 'D:/LLM/model/Qwen3-0.6B-Base'
    adapter_path = 'D:/LLM/train_model/DiCo-NLI/n_ablation/run_B/final'

    # 加载基础模型，再挂载 B 实验训练好的 Adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype='auto'
    )
    model = PeftModel.from_pretrained(
        model,
        adapter_path
    ).to('cuda')

    model.eval()

    # 固定 batch_size=16，与之前正式评估一致
    records = evaluate_with_scores(
        model,
        batch_size=16,
        split='dev'
    )

    # 保存到新目录
    output_dir = (
        Path(__file__).resolve().parent
        / 'outputs/readout_comparison/qwen_B_seed42'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    output_path = output_dir / 'predictions_with_scores.csv'

    df.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig'
    )

    print('结果形状：', df.shape)  # 预期为 (660, 8)
    print('保存位置：', output_path)
    print(df.head(3).to_string(index=False))