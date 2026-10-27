import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessor

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.data_utils import get_label_ids
from experiments.qwen3_1p7b_lexical.lexical_features import build_lexical_features


class LexicalLogitsProcessor(LogitsProcessor):
    def __init__(self, label_ids, correction):
        self.label_ids = label_ids
        self.correction = correction

    def __call__(self, input_ids, scores):
        # 本实验只生成一个 token；只修正 F/B/E/N 的四列，其余词表分数保留。
        return scores.index_add(1, self.label_ids, self.correction.to(scores.dtype))


def predict_with_scores(model, tokenizer, dataset, batch_size=16):
    device = next(model.parameters()).device
    label_ids = torch.tensor(get_label_ids(tokenizer), dtype=torch.long, device=device)
    old_padding = tokenizer.padding_side
    was_training = model.training
    model.eval()
    tokenizer.padding_side = 'left'
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
        return_dict_in_generate=True, output_scores=True,
    )
    records = []
    try:
        for start in tqdm(range(0, len(dataset), batch_size), desc='Qwen + 词汇分支预测'):
            batch = dataset[start:start + batch_size]
            inputs = tokenizer(batch['prompt'], padding=True, return_tensors='pt').to(device)
            features = torch.tensor([
                build_lexical_features(first, second)
                for first, second in zip(batch['text1'], batch['text2'])
            ], dtype=torch.float32, device=device)
            with torch.inference_mode():
                correction = model.lexical_correction(features)
                processor = LexicalLogitsProcessor(label_ids, correction)
                result = model.language_model.generate(
                    **inputs, generation_config=generation_config, logits_processor=[processor],
                )
            scores = result.scores[0][:, label_ids].float().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError('预测分数出现 NaN 或无穷。')
            generated_ids = result.sequences[:, inputs['input_ids'].shape[1]:]
            texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions = [cfg.LABELS[index] for index in scores.argmax(dim=1).tolist()]
            correction = correction.float().cpu()
            for index, prediction in enumerate(predictions):
                row = {
                    'instance_id': batch['instance_id'][index], 'pair_id': batch['pair_id'][index],
                    'gold_label': batch['label'][index], 'prediction': prediction,
                    'generated_prediction': texts[index].strip(),
                }
                for column, label in enumerate(cfg.LABELS):
                    row[f'logit_{label}'] = scores[index, column].item()
                    row[f'lexical_{label}'] = correction[index, column].item()
                records.append(row)
    finally:
        tokenizer.padding_side = old_padding
        model.train(was_training)
    return pd.DataFrame(records)
