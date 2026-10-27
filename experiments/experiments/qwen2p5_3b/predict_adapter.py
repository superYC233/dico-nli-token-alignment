"""用已训练的 Adapter 预测任意 CSV（dev / trial / 其他）。

产物：
  predictions_with_scores.csv    instance_id, pair_id, prediction, logit_F/B/E/N
  submission_pair_aware.csv      instance_id, label（官方全名，可直接提交）
  submission_independent.csv     instance_id, label（同上，未做结构解码）

注意：prompt 必须和训练时逐字一致，否则 Adapter 的分数没有意义。
"""

from pathlib import Path
import argparse
import sys

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pair_aware_decode 在 qwen_lora 目录下
QWEN_LORA_DIR = SCRIPT_DIR.parent / 'qwen_lora'
if str(QWEN_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_LORA_DIR))

from analyze_decoding import FULL_LABELS, pair_aware_decode


BASE_MODEL = 'D:/LLM/model/Qwen2.5-3B-Instruct'
ADAPTER = 'D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_valid/run01/final'

LABELS = ['F', 'B', 'E', 'N']

# 必须与训练脚本里的 SYSTEM_PROMPT 完全一致
SYSTEM_PROMPT = (
    'Classify the semantic relation between Text 1 and Text 2.\n'
    'F: Text 1 entails Text 2, but Text 2 does not entail Text 1.\n'
    'B: Text 2 entails Text 1, but Text 1 does not entail Text 2.\n'
    'E: Text 1 and Text 2 entail each other.\n'
    'N: Neither text entails the other.\n'
    'Answer with exactly one letter: F, B, E, or N.'
)


def build_chat_prompt(tokenizer, text1, text2):
    """与训练时同一套 chat template。"""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Text 1: {text1}\nText 2: {text2}'},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def check_label_tokens(tokenizer):
    """四个标签必须是单 token，且在 chat template 之后仍然成立。"""
    token_ids = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'{label} 不是单 token：{ids}')
        token_ids.append(ids[0])

    prefix = build_chat_prompt(tokenizer, 'train station', 'station')
    a = tokenizer(prefix, add_special_tokens=False)['input_ids']
    b = tokenizer(prefix + 'F', add_special_tokens=False)['input_ids']
    if len(b) != len(a) + 1 or b[-1] != token_ids[0]:
        raise ValueError('标签在 chat template 之后不是单 token')

    print('label_ids:', dict(zip(LABELS, token_ids)))
    return token_ids


@torch.inference_mode()
def predict_frame(model, tokenizer, df, label_ids, batch_size=8):
    """逐批前向，取最后位置的四个候选分数。"""
    model.eval()
    records = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        prompts = [
            build_chat_prompt(tokenizer, a, b)
            for a, b in zip(batch['text1'], batch['text2'])
        ]

        encoded = tokenizer(prompts, padding=True, return_tensors='pt')
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        # logits_to_keep=1：只保留最后一个位置，省显存
        outputs = model(**encoded, logits_to_keep=1)

        # 左补齐时，最后位置就是最后一条真实 token
        scores = outputs.logits[:, -1, label_ids].float().cpu()
        predictions = [LABELS[i] for i in scores.argmax(dim=-1).tolist()]

        for (_, row), prediction, score in zip(
            batch.iterrows(), predictions, scores
        ):
            records.append({
                'instance_id': row['instance_id'],
                'pair_id': row['pair_id'],
                'prediction': prediction,
                'logit_F': float(score[0]),
                'logit_B': float(score[1]),
                'logit_E': float(score[2]),
                'logit_N': float(score[3]),
            })

    return pd.DataFrame(records)


def summarize(df, column, gold=None):
    """本地算一下准确率；只用于自查，正式分数仍以官方 scorer 为准。"""
    if gold is None or 'label' not in df.columns:
        return

    truth = df['label'].map(
        {full: short for short, full in FULL_LABELS.items()}
    )
    accuracy = (truth.to_numpy() == df[column].to_numpy()).mean()
    print(f'{column} 准确率：{accuracy:.6f}')


def main():
    parser = argparse.ArgumentParser(description='用已训练 Adapter 预测 CSV')
    parser.add_argument(
        '--csv',
        default=str(
            PROJECT_ROOT
            / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
        ),
        help='要预测的 CSV，必须有 instance_id / pair_id / text1 / text2',
    )
    parser.add_argument('--out', required=True, help='输出目录')
    parser.add_argument('--base', default=BASE_MODEL)
    parser.add_argument('--adapter', default=ADAPTER)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, keep_default_na=False)
    required = {'instance_id', 'pair_id', 'text1', 'text2'}
    if not required <= set(df.columns):
        raise ValueError(f'缺少列：{sorted(required - set(df.columns))}')

    print('输入：', csv_path)
    print('行数：', len(df))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=True, padding_side='left'
    )
    label_ids = check_label_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype='auto', local_files_only=True
    )
    model = PeftModel.from_pretrained(model, args.adapter).to('cuda')

    decoded = predict_frame(
        model, tokenizer, df, label_ids, args.batch_size
    )

    # 把原文标签带回来，方便本地自查
    if 'label' in df.columns:
        decoded = decoded.merge(
            df[['instance_id', 'label']], on='instance_id', how='left'
        )

    # 结构解码（pair_aware_decode 不需要 gold）
    decoded = pair_aware_decode(decoded)

    decoded.to_csv(
        out_dir / 'predictions_with_scores.csv',
        index=False,
        encoding='utf-8-sig',
    )

    # 两套预测都导出官方格式
    for column, name in [
        ('prediction', 'submission_independent.csv'),
        ('pair_prediction', 'submission_pair_aware.csv'),
    ]:
        submission = pd.DataFrame({
            'instance_id': decoded['instance_id'],
            'label': decoded[column].map(FULL_LABELS),
        })
        submission.to_csv(out_dir / name, index=False)
        summarize(decoded, column, gold=True)

    print('输出目录：', out_dir)


if __name__ == '__main__':
    main()