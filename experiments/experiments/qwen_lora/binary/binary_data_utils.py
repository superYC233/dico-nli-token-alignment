
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import math

import torch
from tqdm import tqdm

from data_utils import load_data, tokenizer, encode_prompt


binary_label_map = {
    'FORWARD_ENTAILMENT': '1',
    'BACKWARD_ENTAILMENT': '0',
    'EQUIVALENCE': '1',
    'NEGATIVE_OTHER': '0'
}


def build_binary_prompt(text1, text2):
    prompt = f"""
Determine whether Text 1 entails Text 2.

1: The meaning of Text 1 necessarily implies the meaning of Text 2.
0: Text 1 does not necessarily imply Text 2.

Semantic similarity or topical relatedness alone is not sufficient for entailment.

Output exactly one label: 1 or 0.

Text 1: {text1}
Text 2: {text2}
Answer:
"""
    return prompt.strip() + '\n'


def build_binary_example(example):
    return {
        'prompt': build_binary_prompt(
            example['text1'],
            example['text2']
        ),
        'completion': binary_label_map[example['label']]
    }


def get_binary_datasets(dataset):
    return dataset.map(
        build_binary_example,
        load_from_cache_file=False
    )


def predict_binary_logprobs(prompts, model):
    inputs = encode_prompt(prompts).to(model.device)

    token_ids = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in ['0', '1']
    ]

    assert all(len(ids) == 1 for ids in token_ids)
    label_ids = [ids[0] for ids in token_ids]

    with torch.inference_mode():
        result = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # 只在 0、1 两个候选之间归一化。
        logits = result.scores[0][:, label_ids].float()
        log_probs = torch.log_softmax(logits, dim=-1)

    # 每行依次是 log P(0)、log P(1)。
    return log_probs.cpu().tolist()


def evaluate_binary_with_scores(model, batch_size=16, split='dev'):
    # 读取原始文本，现场生成二元 Prompt。
    # 不经过旧版四分类 Prompt 的 map 缓存。
    rows = list(load_data()[split])

    question_pairs = []
    seen_pairs = set()

    for row in rows:
        pairs = [
            (row['text1'], row['text2']),
            (row['text2'], row['text1'])
        ]

        for pair in pairs:
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                question_pairs.append(pair)

    # 相同的有序文本对只预测一次。
    scores_by_pair = {}

    for start in tqdm(
        range(0, len(question_pairs), batch_size),
        desc='二元蕴含预测'
    ):
        batch_pairs = question_pairs[start:start + batch_size]

        prompts = [
            build_binary_prompt(text1, text2)
            for text1, text2 in batch_pairs
        ]

        batch_scores = predict_binary_logprobs(prompts, model)

        for pair, scores in zip(batch_pairs, batch_scores):
            scores_by_pair[pair] = scores

    labels = ['F', 'B', 'E', 'N']
    records = []

    for row in rows:
        forward = scores_by_pair[(row['text1'], row['text2'])]
        reverse = scores_by_pair[(row['text2'], row['text1'])]

        # 用两个方向的概率乘积定义四类分数。
        # 在 log 空间中，相乘变成相加，避免数值下溢。
        scores = [
            forward[1] + reverse[0],  # F：正向是，反向否
            forward[0] + reverse[1],  # B：正向否，反向是
            forward[1] + reverse[1],  # E：两向都是
            forward[0] + reverse[0]   # N：两向都否
        ]

        best_index = scores.index(max(scores))

        records.append({
            'instance_id': row['instance_id'],
            'pair_id': row['pair_id'],
            'gold_label': row['label'],
            'prediction': labels[best_index],
            'logit_F': scores[0],
            'logit_B': scores[1],
            'logit_E': scores[2],
            'logit_N': scores[3],
            'forward_entail_probability': math.exp(forward[1]),
            'reverse_entail_probability': math.exp(reverse[1])
        })

    return records