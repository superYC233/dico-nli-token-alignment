"""生成式短语 NLI 公共工具；模型、tokenizer、Prompt 和缓存由各实验显式提供。"""
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from evaluation_functions import score_files

LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}


def load_data(project_root, cache_dir):
    return load_dataset('csv', cache_dir=str(cache_dir), data_files={
        'train': str(project_root / 'data/train/dico_nli_train_track1_participant_labeled.csv'),
        'dev': str(project_root / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'),
    })


def build_sft_example(row, prompt_template):
    return {
        'prompt': prompt_template.format(text1=row['text1'], text2=row['text2']),
        'completion': LABEL_MAP[row['label']],
    }


def get_datasets(dataset, prompt_template):
    return dataset.map(
        build_sft_example,
        fn_kwargs={'prompt_template': prompt_template},
        load_from_cache_file=True,
    )


def get_label_ids(tokenizer):
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError(f'四个标签必须各为一个 token，实际是：{token_ids}')
    label_ids = [ids[0] for ids in token_ids]
    if len(set(label_ids)) != 4:
        raise ValueError('四个标签必须使用不同的 token ID。')
    return label_ids


def load_tokenizer(path):
    path = Path(path)
    if not (path / 'tokenizer_config.json').exists():
        raise FileNotFoundError(f'尚未准备 tokenizer：{path}\n请先运行 download_model.py。')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError('tokenizer 没有 EOS token。')
    if tokenizer.pad_token_id is None:
        # 使用已有 EOS 补齐，不新增词表项；attention_mask 区分正文和补齐。
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    get_label_ids(tokenizer)
    return tokenizer


def check_examples(dataset, tokenizer, max_length):
    """检查完整输入长度和答案边界，防止标签被截断或与 Prompt 合并。"""
    label_ids = dict(zip(LABELS, get_label_ids(tokenizer)))
    longest = 0
    for split, rows in dataset.items():
        for row in rows:
            prompt_ids = tokenizer(row['prompt'])['input_ids']
            full_ids = tokenizer(row['prompt'] + row['completion'] + tokenizer.eos_token)['input_ids']
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f'{split} 的 Prompt/答案边界不稳定：{row["instance_id"]}')
            answer_ids = full_ids[len(prompt_ids):]
            if answer_ids != [label_ids[row['completion']], tokenizer.eos_token_id]:
                raise ValueError(f'{split} 的答案不是标签 + EOS：{answer_ids}')
            if len(full_ids) > max_length:
                raise ValueError(f'{split} 输入超过 max_length：{row["instance_id"]}')
            longest = max(longest, len(full_ids))
    print('四类标签 token ID：', label_ids)
    print('最长完整输入 token 数：', longest)


def predict_with_scores(model, tokenizer, dataset, batch_size=16):
    label_ids = get_label_ids(tokenizer)
    device = next(model.parameters()).device
    old_padding = tokenizer.padding_side
    was_training = model.training
    records = []
    model.eval()
    tokenizer.padding_side = 'left'
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    try:
        for start in tqdm(range(0, len(dataset), batch_size), desc='Dev 批量预测'):
            batch = dataset[start:start + batch_size]
            inputs = tokenizer(batch['prompt'], padding=True, return_tensors='pt').to(device)
            with torch.inference_mode():
                result = model.generate(
                    **inputs, generation_config=generation_config,
                )
            scores = result.scores[0][:, label_ids].float().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError('候选分数出现 NaN 或无穷。')
            raw_ids = result.sequences[:, inputs['input_ids'].shape[1]:]
            raw_texts = tokenizer.batch_decode(raw_ids, skip_special_tokens=True)
            predictions = [LABELS[index] for index in scores.argmax(dim=1).tolist()]

            for i, prediction in enumerate(predictions):
                record = {
                    'instance_id': batch['instance_id'][i],
                    'pair_id': batch['pair_id'][i],
                    'gold_label': batch['label'][i],
                    # 正式读出固定为四个候选的 argmax；全词表生成另外记录。
                    'prediction': prediction,
                    'generated_prediction': raw_texts[i].strip(),
                }
                for label, value in zip(LABELS, scores[i].tolist()):
                    record[f'logit_{label}'] = value
                records.append(record)
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    return pd.DataFrame(records)


def pair_aware_decode(frame):
    frame = frame.copy()
    if not frame['instance_id'].is_unique or not frame.index.is_unique:
        raise ValueError('预测中存在重复编号。')
    frame['pair_prediction'] = frame['prediction']
    combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]
    for _, group in frame.groupby('pair_id'):
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise ValueError('Track 1 出现了超过两行的分组。')
        row0, row1 = group.iloc[0], group.iloc[1]
        scores = [
            row0['logit_F'] + row1['logit_B'],
            row0['logit_B'] + row1['logit_F'],
            row0['logit_E'] + row1['logit_E'],
        ]
        label0, label1 = combinations[scores.index(max(scores))]
        frame.loc[group.index[0], 'pair_prediction'] = label0
        frame.loc[group.index[1], 'pair_prediction'] = label1
    return frame


def score_methods(frame, output_dir, gold_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    inverse = {short: full for full, short in LABEL_MAP.items()}
    summary = []
    reports = {}
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        if not frame[column].isin(LABELS).all():
            raise ValueError('预测中存在非法标签。')
        score_dir = output_dir / method
        score_dir.mkdir(parents=True, exist_ok=True)
        path = score_dir / 'official_predictions.csv'
        pd.DataFrame({
            'instance_id': frame['instance_id'], 'label': frame[column].map(inverse),
        }).to_csv(path, index=False)
        report = score_files(str(gold_path), str(path), output_dir=str(score_dir))
        n_metrics = report.classification.per_label['NEGATIVE_OTHER']
        summary.append({
            'method': method, 'weighted_f1': report.weighted_f1,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_f1': n_metrics.f1, 'N_correct': n_metrics.true_positive,
            'accuracy': report.classification.accuracy,
            'FBE_correct': sum(
                report.classification.per_label[inverse[label]].true_positive
                for label in ['F', 'B', 'E']
            ),
        })
        reports[method] = report
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    illegal = (~frame['generated_prediction'].isin(LABELS)).sum()
    disagreement = (frame['prediction'] != frame['generated_prediction']).sum()
    print('全词表生成的非法标签数：', int(illegal))
    print('全词表生成与四候选读出的不同数：', int(disagreement))
    return reports
