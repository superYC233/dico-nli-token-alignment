"""右键运行：用同一个原 C+r16 Adapter 预测三组 Prompt，不进行训练。"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import config as cfg
from retrieval_utils import read_frame, digest

sys.path.insert(0, str(cfg.QWEN_DIR))
sys.path.insert(0, str(cfg.PROJECT_ROOT))
from evaluate_lora_pair import pair_aware_decode
from evaluation_functions import score_files


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_prepared():
    manifest_path = cfg.PREPARED_DIR / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('请先运行 prepare_examples.py，或者直接运行 run_all.py。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    dev = read_frame(cfg.DEV_FILE)
    queries = dev[['instance_id', 'pair_id', 'text1', 'text2']]
    train = read_frame(cfg.TRAIN_FILE)
    spec = manifest['spec']
    if digest(queries.to_dict('records')) != spec['queries_sha256']:
        raise ValueError('Dev 文本或顺序改变，请换 RUN_NAME 后重新准备。')
    if digest(train.to_dict('records')) != spec['train_sha256']:
        raise ValueError('Train 改变，请换 RUN_NAME 后重新准备。')
    if spec['num_examples'] != cfg.NUM_EXAMPLES or spec['seed'] != cfg.SEED:
        raise ValueError('示例数量或 seed 改变，请换 RUN_NAME 后重新准备。')
    prompts = pd.read_csv(cfg.PREPARED_DIR / 'prompts.csv', keep_default_na=False)
    if set(prompts['method']) != set(cfg.METHODS):
        raise ValueError('准备的 Prompt 方法不完整。')
    for method in cfg.METHODS:
        rows = prompts[prompts['method'] == method]
        if rows['instance_id'].tolist() != dev['instance_id'].tolist():
            raise ValueError(f'{method} 的问题顺序或样本数不正确。')
        if rows['pair_id'].tolist() != dev['pair_id'].tolist():
            raise ValueError(f'{method} 的 pair_id 不正确。')
    # pair-aware 只按元数据分组；先验证两个文本确实互换。
    for pair_id, group in dev.groupby('pair_id', sort=False):
        if len(group) == 2:
            first, second = group.iloc[0], group.iloc[1]
            if first['text1'] != second['text2'] or first['text2'] != second['text1']:
                raise ValueError(f'{pair_id} 不是文本互换组，不能使用当前 pair-aware。')
        elif len(group) != 1:
            raise ValueError(f'{pair_id} 的行数不是 1 或 2。')
    return dev, prompts, manifest


def predict_batch(prompts, model, tokenizer, device, max_prompt_tokens):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=False).to(device)
    if inputs['input_ids'].shape[1] > max_prompt_tokens:
        raise ValueError('Prompt 超过长度限制，不会静默截断。')
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in cfg.LABELS]
    if not all(len(item) == 1 for item in ids) or len({item[0] for item in ids}) != 4:
        raise ValueError('F/B/E/N 必须是四个不同的单 token。')
    with torch.inference_mode():
        result = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    generated = result.sequences[:, inputs['input_ids'].shape[1]:]
    raw_texts = [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]
    scores = result.scores[0][:, [item[0] for item in ids]].float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError('四类候选分数出现 NaN 或 Inf。')
    return raw_texts, scores


def score_method(frame, directory, reference_file=None):
    reference_file = reference_file or cfg.REFERENCE_FILE
    records = []
    gold = frame['gold_label'].map(cfg.LABEL_MAP)
    for name, column in [('independent', 'prediction'), ('pair_aware', 'pair_prediction')]:
        output = Path(directory) / name
        output.mkdir(parents=True, exist_ok=True)
        predictions = frame[['instance_id', column]].rename(columns={column: 'label'})
        inverse = {short: full for full, short in cfg.LABEL_MAP.items()}
        predictions['label'] = predictions['label'].map(inverse)
        path = output / 'official_predictions.csv'
        predictions.to_csv(path, index=False)
        report = score_files(str(reference_file), str(path), output_dir=str(output))
        correct = frame[column] == gold
        n = report.classification.per_label['NEGATIVE_OTHER']
        record = {
            'method': frame['method'].iloc[0], 'decoding': name,
            'weighted_f1': report.weighted_f1,
            'accuracy': float(correct.mean()),
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_precision': n.precision, 'N_recall': n.recall, 'N_f1': n.f1,
            'N_correct': int((correct & (gold == 'N')).sum()),
            'FBE_correct': int((correct & (gold != 'N')).sum()),
            'illegal_generated': int((~frame['generated_prediction'].isin(cfg.LABELS)).sum()),
            'generated_vs_candidate_disagreement': int((frame['generated_prediction'] != frame['prediction']).sum()),
            'mean_prompt_tokens': float(frame['prompt_tokens'].mean()),
        }
        records.append(record)
        print(f"{record['method']} / {name}: W-F1={report.weighted_f1:.6f}，"
              f"N={record['N_correct']}，F/B/E={record['FBE_correct']}，"
              f"非法生成={record['illegal_generated']}")
    (Path(directory) / 'metrics.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return records


def compare_historical_baseline(frame):
    if not cfg.HISTORICAL_BASELINE.exists():
        return {'historical_file_available': False}
    previous = pd.read_csv(cfg.HISTORICAL_BASELINE, keep_default_na=False).set_index('instance_id')
    current = frame.set_index('instance_id')
    if not previous.index.is_unique or set(previous.index) != set(current.index):
        raise ValueError('历史基线与当前 Dev 的 instance_id 不一致。')
    previous = previous.loc[current.index]
    different = (previous['prediction'] != current['prediction']) | (previous['pair_prediction'] != current['pair_prediction'])
    result = {
        'historical_file_available': True, 'historical_file': str(cfg.HISTORICAL_BASELINE),
        'independent_differences': int((previous['prediction'] != current['prediction']).sum()),
        'pair_aware_differences': int((previous['pair_prediction'] != current['pair_prediction']).sum()),
        'different_instance_ids': current.index[different].tolist(),
    }
    return result


def evaluate():
    dev, prompts, manifest = load_prepared()
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA。本入口需要你的 GPU；准备示例和汇总可以在 CPU 运行。')
    adapter_dir = Path(cfg.ADAPTER_PATH)
    if not (adapter_dir / 'adapter_config.json').exists():
        raise FileNotFoundError(f'找不到原 C+r16 Adapter：{adapter_dir}')
    weight_files = sorted([*adapter_dir.glob('*.safetensors'), *adapter_dir.glob('*.bin')])
    if not weight_files:
        raise FileNotFoundError('Adapter 目录没有权重文件。')
    inference_spec = {
        'version': 1, 'base_model': cfg.MODEL_PATH, 'adapter': cfg.ADAPTER_PATH,
        'adapter_sha256': {path.name: file_sha256(path) for path in [adapter_dir / 'adapter_config.json', *weight_files]},
        'prompt_file_sha256': file_sha256(cfg.PREPARED_DIR / 'prompts.csv'),
        'reference_sha256': file_sha256(cfg.REFERENCE_FILE),
        'batch_size': cfg.PREDICT_BATCH_SIZE, 'max_prompt_tokens': cfg.MAX_PROMPT_TOKENS,
        'seed': cfg.SEED, 'dtype': 'auto', 'padding_side': 'left',
        'readout': 'four_label_argmax; also save unrestricted one-token generation',
        'preparation_spec': manifest['spec'],
    }
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = cfg.RESULTS_DIR / 'inference_config.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf-8')) != inference_spec:
        raise ValueError('当前目录已有不同预测配置，请换 RUN_NAME 后运行，避免覆盖比较结果。')
    spec_path.write_text(json.dumps(inference_spec, ensure_ascii=False, indent=2), encoding='utf-8')
    set_seed(cfg.SEED)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_PATH, dtype='auto')
    model = PeftModel.from_pretrained(model, cfg.ADAPTER_PATH).to('cuda').eval()

    for method in cfg.METHODS:
        rows = prompts[prompts['method'] == method].reset_index(drop=True)
        records = []
        for start in tqdm(range(0, len(rows), cfg.PREDICT_BATCH_SIZE), desc=method):
            batch = rows.iloc[start:start + cfg.PREDICT_BATCH_SIZE]
            raw_texts, scores = predict_batch(batch['prompt'].tolist(), model, tokenizer,
                                             'cuda', cfg.MAX_PROMPT_TOKENS)
            probabilities = torch.softmax(scores, dim=-1)
            labels = [cfg.LABELS[i] for i in scores.argmax(dim=1).tolist()]
            for query, raw, label, logits, probs in zip(
                batch.to_dict('records'), raw_texts, labels, scores.tolist(), probabilities.tolist()
            ):
                record = {'instance_id': query['instance_id'], 'pair_id': query['pair_id'],
                          'method': method, 'prediction': label, 'generated_prediction': raw,
                          'prompt_tokens': query['prompt_tokens']}
                for index, short in enumerate(cfg.LABELS):
                    record[f'logit_{short}'] = logits[index]
                    record[f'prob_{short}'] = probs[index]
                records.append(record)
        frame = pd.DataFrame(records)
        # 金标到评分阶段才合并，未传入 Prompt 或检索器。
        frame = frame.merge(dev[['instance_id', 'label']].rename(columns={'label': 'gold_label'}),
                            on='instance_id', how='left', validate='one_to_one', sort=False)
        frame = pair_aware_decode(frame)
        directory = cfg.RESULTS_DIR / method
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
        score_method(frame, directory)
        if method == 'baseline':
            comparison = compare_historical_baseline(frame)
            (cfg.RESULTS_DIR / 'baseline_reproduction.json').write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print('与历史基线逐条比较：', comparison)
    print('预测完成：', cfg.RESULTS_DIR)


if __name__ == '__main__':
    evaluate()
    from summarize_results import summarize
    summarize()

