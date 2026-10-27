"""逐行一步生成、四标签分数和pair-aware；直接运行评估本目录best_adapter。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from evaluation_functions import score_files
from experiments.qwen_scope_aug.data_utils import DEV_PATH, GOLD_PATH, LABELS, LABEL_MAP


def pair_aware_decode(frame):
    frame = frame.copy()
    frame['pair_prediction'] = frame['prediction']
    combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]
    for _, group in frame.groupby('pair_id'):
        if len(group) == 1:
            continue
        if len(group) != 2:
            raise ValueError('pair_id 不是单例或两行组。')
        first, second = group.iloc[0], group.iloc[1]
        # 与历史 C+r16 同式；对固定的一对，log-softmax后相加也会得到相同排序。
        scores = [first.logit_F + second.logit_B,
                  first.logit_B + second.logit_F,
                  first.logit_E + second.logit_E]
        label0, label1 = combinations[scores.index(max(scores))]
        frame.loc[group.index[0], 'pair_prediction'] = label0
        frame.loc[group.index[1], 'pair_prediction'] = label1
    return frame


def predict_with_scores(model, tokenizer, dev, prompt_template, batch_size=16):
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('F/B/E/N必须各自对应一个token。')
    label_ids = [ids[0] for ids in token_ids]
    device = next(model.parameters()).device
    old_padding, was_training = tokenizer.padding_side, model.training
    records = []
    try:
        tokenizer.padding_side = 'left'
        model.eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(dev), batch_size), desc='Qwen Dev预测'):
                batch = dev.iloc[start:start + batch_size]
                prompts = [prompt_template.format(text1=row.text1, text2=row.text2)
                           for row in batch.itertuples()]
                inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
                result = model.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True,
                )
                new_ids = result.sequences[:, inputs['input_ids'].shape[1]:]
                predictions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                logits = result.scores[0][:, label_ids].float().cpu()
                for row, prediction, values in zip(batch.itertuples(), predictions, logits.tolist()):
                    records.append({
                        'instance_id': row.instance_id, 'pair_id': row.pair_id,
                        'text1': row.text1, 'text2': row.text2, 'gold_label': row.label,
                        'prediction': prediction.strip(),
                        **{f'logit_{label}': value for label, value in zip(LABELS, values)},
                    })
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    return pd.DataFrame(records)


def check_label_scores(frame):
    # generate在全词表中选token，四候选按F/B/E/N列顺序做argmax。
    # 最高分并列时，两者可选中不同标签；只要求生成标签属于最高分集合。
    # 例如F=E=27.875，生成E完全合法，不能因四候选argmax先取F就中断训练。
    label_scores = frame[[f'logit_{label}' for label in LABELS]]
    highest_scores = label_scores.max(axis=1)
    generated_scores = pd.Series(
        [getattr(row, f'logit_{row.prediction}') for row in frame.itertuples()],
        index=frame.index,
    )
    below_max = generated_scores != highest_scores
    if below_max.any():
        ids = frame.loc[below_max, 'instance_id'].tolist()
        raise ValueError(f'生成标签的分数没有达到四候选最高分，请检查token映射：{ids}')
    ties = label_scores.eq(highest_scores, axis=0).sum(axis=1) > 1
    four_argmax = label_scores.to_numpy().argmax(axis=1)
    differences = sum(prediction != LABELS[index]
                      for prediction, index in zip(frame.prediction, four_argmax))
    return {'rows': len(frame), 'rows_with_tied_maximum': int(ties.sum()),
            'argmax_order_disagreements': int(differences), 'predictions_below_maximum': 0}


def score_methods(frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    invalid = ~frame.prediction.isin(LABELS)
    if invalid.any():
        raise ValueError(f'有{int(invalid.sum())}条非法生成，已保存预测，请检查训练输入。')
    check = check_label_scores(frame)
    (output_dir / 'label_score_check.json').write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    if check['argmax_order_disagreements']:
        print(f'有{check["argmax_order_disagreements"]}条因最高分并列而与四候选argmax不同，保留原生成答案。')

    full_labels = {short: full for full, short in LABEL_MAP.items()}
    gold = frame.gold_label.map(LABEL_MAP)
    summaries = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        folder = output_dir / method
        folder.mkdir(parents=True, exist_ok=True)
        prediction_path = folder / 'official_predictions.csv'
        pd.DataFrame({'instance_id': frame.instance_id,
                      'label': frame[column].map(full_labels)}).to_csv(prediction_path, index=False)
        report = score_files(GOLD_PATH, prediction_path, output_dir=folder)
        n = report.classification.per_label['NEGATIVE_OTHER']
        row = {'method': method, 'weighted_f1': report.weighted_f1,
               'accuracy': report.classification.accuracy,
               'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
               'N_precision': n.precision, 'N_recall': n.recall, 'N_f1': n.f1,
               'N_correct': n.true_positive,
               'nonN_to_N': int(((gold != 'N') & (frame[column] == 'N')).sum())}
        for label in LABELS[:3]:
            metrics = report.classification.per_label[full_labels[label]]
            row[f'{label}_correct'], row[f'{label}_f1'] = metrics.true_positive, metrics.f1
        row['FBE_correct'] = sum(row[f'{label}_correct'] for label in LABELS[:3])
        summaries.append(row)
        matrix = pd.crosstab(gold.rename('gold'), frame[column].rename('prediction'))
        matrix.reindex(index=LABELS, columns=LABELS, fill_value=0).to_csv(
            folder / 'confusion_matrix.csv', encoding='utf-8-sig',
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    old_correct, new_correct = frame.prediction == gold, frame.pair_prediction == gold
    changed = frame.prediction != frame.pair_prediction
    pd.DataFrame([{'changed': int(changed.sum()), 'fixed': int((~old_correct & new_correct).sum()),
                   'broken': int((old_correct & ~new_correct).sum()),
                   'changed_wrong': int((changed & ~old_correct & ~new_correct).sum())}]).to_csv(
        output_dir / 'decoding_changes.csv', index=False, encoding='utf-8-sig',
    )
    print(summary.to_string(index=False))
    return summary


def evaluate_model(model, tokenizer, dev, prompt_template, output_dir):
    frame = predict_with_scores(model, tokenizer, dev, prompt_template, batch_size=16)
    frame = pair_aware_decode(frame)
    return score_methods(frame, output_dir)


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.qwen_scope_aug.train_lora_aug import OUTPUT_DIR, RESULTS_DIR

    config = json.loads((RESULTS_DIR / 'run_config.json').read_text(encoding='utf-8'))
    adapter_dir = OUTPUT_DIR / 'best_adapter'
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config['model_path'], dtype='auto', local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter_dir).to('cuda')
    model.eval()
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    evaluate_model(model, tokenizer, dev, config['prompt_template'], RESULTS_DIR / 'best_evaluation')
    print('结果目录：', RESULTS_DIR / 'best_evaluation')


if __name__ == '__main__':
    main()
