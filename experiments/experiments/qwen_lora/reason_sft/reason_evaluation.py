"""只用原标签Prompt预测Dev，解释Prompt不进入评估。"""
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import TrainerCallback

from evaluation_functions import score_files
from experiments.qwen_lora.reason_sft import config as cfg


def pair_aware_decode(frame):
    frame = frame.copy()
    frame['pair_prediction'] = frame['prediction']
    combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        first, second = group.iloc[0], group.iloc[1]
        scores = [
            first.logit_F + second.logit_B,
            first.logit_B + second.logit_F,
            first.logit_E + second.logit_E,
        ]
        labels = combinations[scores.index(max(scores))]
        frame.loc[group.index[0], 'pair_prediction'] = labels[0]
        frame.loc[group.index[1], 'pair_prediction'] = labels[1]
    return frame


def predict_dev(model, tokenizer, dev, prompt_template, batch_size=16):
    old_padding = tokenizer.padding_side
    was_training = model.training
    device = next(model.parameters()).device
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in cfg.LABELS]
    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('F/B/E/N必须各为单token。')
    label_ids = [ids[0] for ids in token_ids]
    records = []
    model.eval()
    tokenizer.padding_side = 'left'
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(dev), batch_size), desc='原标签Prompt预测Dev'):
                rows = dev.iloc[start:start + batch_size]
                prompts = [prompt_template.format(text1=r.text1, text2=r.text2) for r in rows.itertuples()]
                inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
                outputs = model.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True,
                )
                scores = outputs.scores[0][:, label_ids].float().cpu()
                raw = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                for row, values, generated in zip(rows.itertuples(), scores.tolist(), raw):
                    records.append({
                        'instance_id': row.instance_id, 'pair_id': row.pair_id, 'gold_label': row.label,
                        'prediction': cfg.LABELS[values.index(max(values))],
                        'generated_prediction': generated.strip(),
                        **{f'logit_{label}': value for label, value in zip(cfg.LABELS, values)},
                    })
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    return pair_aware_decode(pd.DataFrame(records))


def score_methods(frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    inverse = dict(zip(cfg.LABELS, cfg.FULL_LABELS))
    gold = frame.gold_label.map(cfg.LABEL_MAP)
    illegal = int((~frame.generated_prediction.isin(cfg.LABELS)).sum())
    summary = []
    for method, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        directory = output_dir / method
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'official_predictions.csv'
        pd.DataFrame({'instance_id': frame.instance_id, 'label': frame[column].map(inverse)}).to_csv(path, index=False)
        report = score_files(cfg.GOLD_PATH, path, output_dir=directory)
        correct = frame[column].eq(gold)
        n = report.classification.per_label['NEGATIVE_OTHER']
        record = {
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_precision': n.precision, 'N_recall': n.recall, 'N_f1': n.f1,
            'N_correct': int((correct & gold.eq('N')).sum()),
            'FBE_correct': int((correct & gold.ne('N')).sum()),
            'illegal_generated': illegal,
            'raw_generation_accuracy': float(frame.generated_prediction.eq(gold).mean()),
        }
        for label in cfg.LABELS[:3]:
            record[f'{label}_correct'] = int((correct & gold.eq(label)).sum())
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary


class BestF1Callback(TrainerCallback):
    def __init__(self, dev, results_dir, prompt_template, batch_size=cfg.PREDICT_BATCH_SIZE):
        self.dev = dev
        self.results_dir = results_dir
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.history = []

    def on_evaluate(self, args, state, control, model=None, processing_class=None, metrics=None, **kwargs):
        epoch = int(round(state.epoch or 0))
        frame = predict_dev(model, processing_class, self.dev, self.prompt_template, self.batch_size)
        summary = score_methods(frame, self.results_dir / f'epoch_{epoch}')
        metrics['eval_independent_f1'] = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
        metrics['eval_pair_f1'] = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
        for row in summary.to_dict('records'):
            self.history.append({'epoch': epoch, 'global_step': state.global_step, **row})
        pd.DataFrame(self.history).to_csv(self.results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        (self.results_dir / 'last_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        return control
