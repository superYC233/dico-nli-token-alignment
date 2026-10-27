
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from data_utils import tokenizer, encode_prompt
from data_variants import build_train_dataset

script_dir = QWEN_DIR
project_root = script_dir.parent.parent

# 六类训练时使用的单 token 标签
six_labels = ['F', 'B', 'E', 'S', 'R', 'O']

# 最终提交和评分仍然使用四类
four_labels = ['F', 'B', 'E', 'N']

six_label_map = {
    'F': 'F',
    'B': 'B',
    'E': 'E',
    'SIMI': 'S',
    'REL': 'R',
    'OPPO': 'O'
}


def build_sixclass_prompt(text1, text2):
    prompt = f"""
Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
S: Similar meanings, but neither text entails the other.
R: Related meanings, but neither text entails the other.
O: Opposite meanings, but neither text entails the other.

Output exactly one label: F, B, E, S, R, or O.

Text 1: {text1}
Text 2: {text2}
Answer:
"""
    return prompt.strip() + '\n'


def build_sixclass_example(row):
    return {
        'prompt': build_sixclass_prompt(
            row['text1'],
            row['text2']
        ),
        'completion': six_label_map[row['six_label']]
    }


def get_sixclass_train_dataset(mode='C'):
    path = (
        script_dir
        / 'outputs/sixclass_data/train_with_relations.csv'
    )

    dataset = load_dataset(
        'csv',
        data_files=str(path),
        split='train'
    )

    # 继续使用你的 C 构造方法。
    # 反向 N 保留原来的 SIMI / REL / OPPO 标签。
    dataset = build_train_dataset(dataset, mode)

    print('训练版本：', mode)
    print('训练样本数：', len(dataset))
    print(pd.Series(dataset['six_label']).value_counts())

    # 重新生成六类 Prompt，避免误用旧缓存。
    return dataset.map(
        build_sixclass_example,
        load_from_cache_file=False
    )


def get_sixclass_token_ids():
    token_ids = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in six_labels
    ]

    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('六个标签必须分别对应一个 token')

    label_ids = [ids[0] for ids in token_ids]

    if len(set(label_ids)) != 6:
        raise ValueError('六个标签的 token ID 必须互不相同')

    return label_ids


def predict_sixclass_batch(prompts, model):
    inputs = encode_prompt(prompts).to(model.device)
    label_ids = get_sixclass_token_ids()

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

        # 取六个候选的分数，形状 [batch_size, 6]。
        # 列顺序：F、B、E、S、R、O。
        logits = result.scores[0][:, label_ids].float()

        # 在六个候选之间归一化，得到 log 概率。
        six_log_probs = torch.log_softmax(logits, dim=-1)

        # P(N) = P(S) + P(R) + P(O)。
        # logsumexp 在 log 空间里完成概率相加。
        n_log_prob = torch.logsumexp(
            six_log_probs[:, 3:],
            dim=-1,
            keepdim=True
        )

        # 合并后得到四列：F、B、E、N。
        four_log_probs = torch.cat(
            [six_log_probs[:, :3], n_log_prob],
            dim=-1
        )

    return (
        four_log_probs.cpu().tolist(),
        six_log_probs.exp().cpu().tolist()
    )


def evaluate_sixclass_with_scores(model, batch_size=16):
    # Dev 仍然只使用官方四类标签。
    dev_path = (
        project_root
        / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
    )
    dev = pd.read_csv(dev_path, keep_default_na=False)

    old_padding_side = tokenizer.padding_side
    was_training = model.training

    model.eval()
    records = []

    try:
        for start in tqdm(
            range(0, len(dev), batch_size),
            desc='六类模型预测 Dev'
        ):
            batch = dev.iloc[start:start + batch_size]

            prompts = [
                build_sixclass_prompt(
                    row['text1'],
                    row['text2']
                )
                for _, row in batch.iterrows()
            ]

            four_scores, six_probs = predict_sixclass_batch(
                prompts,
                model
            )

            for row, scores, probs in zip(
                batch.to_dict('records'),
                four_scores,
                six_probs
            ):
                # 在合并后的四类中选择预测。
                best_index = scores.index(max(scores))

                record = {
                    'instance_id': row['instance_id'],
                    'pair_id': row['pair_id'],
                    'gold_label': row['label'],
                    'prediction': four_labels[best_index],

                    # 为兼容已有 pair-aware，沿用这些列名。
                    # 实际保存的是四类 log 概率。
                    'logit_F': scores[0],
                    'logit_B': scores[1],
                    'logit_E': scores[2],
                    'logit_N': scores[3]
                }

                # 额外保存六类概率，方便之后分析。
                for label, probability in zip(six_labels, probs):
                    record[f'prob_{label}'] = probability

                records.append(record)

    finally:
        # 恢复训练需要的补齐方式和模型状态。
        tokenizer.padding_side = old_padding_side
        model.train(was_training)

    return records