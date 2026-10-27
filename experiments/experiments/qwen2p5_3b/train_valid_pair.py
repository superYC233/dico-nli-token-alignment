"""Qwen2.5-3B-Instruct + LoRA + 合法组合约束（B 数据）

与 0.6B 版本的差异（只有三处）：
  1. 模型换成 Qwen2.5-3B-Instruct
  2. prompt 改用 chat template（Instruct 模型必须）
  3. batch 从 16 降到 8（3B 在 12GB 上装不下 16），pair 批同步减半

保持不变：LoRA 配置、lr、5 epochs、合法组合损失形式、λ=0.1、
          B 的数据构造（3514 行）、真实 pair（1282 组）、最终轮评估。
"""

from pathlib import Path
import json
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# analyze_decoding.py 在 qwen_lora 目录下，需要把它加进来
QWEN_LORA_DIR = SCRIPT_DIR.parent / 'qwen_lora'
if str(QWEN_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_LORA_DIR))

from analyze_decoding import pair_aware_decode, score_one_method


MODEL_PATH = 'D:/LLM/model/Qwen2.5-3B-Instruct'

RUN_NAME = 'run01'
RUN_DIR = Path('D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_valid') / RUN_NAME
RESULT_DIR = SCRIPT_DIR / 'outputs/qwen2p5_3b_valid' / RUN_NAME

SOURCE_CSV = (
    SCRIPT_DIR.parent / 'qwen_lora'
    / 'outputs/n_ablation/swap_sources.csv'
)
SEED = 42
PAIR_WEIGHT = 0.1
BATCH_SIZE = 8                 # 0.6B 用的是 16；3B 显存不够
PAIR_BATCH_SIZE = max(1, BATCH_SIZE // 4)   # 保持 row:pair 的比例不变
EPOCHS = 5
LR = 1e-4

LABELS = ['F', 'B', 'E', 'N']
LABEL_SHORT = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}

LORA_CONFIG = dict(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules='all-linear', bias='none', task_type='CAUSAL_LM',
)

SYSTEM_PROMPT = (
    'Classify the semantic relation between Text 1 and Text 2.\n'
    'F: Text 1 entails Text 2, but Text 2 does not entail Text 1.\n'
    'B: Text 2 entails Text 1, but Text 1 does not entail Text 2.\n'
    'E: Text 1 and Text 2 entail each other.\n'
    'N: Neither text entails the other.\n'
    'Answer with exactly one letter: F, B, E, or N.'
)


# ----------------------------------------------------------------- 1. tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.padding_side = 'right'          # SFT 分支用右补齐


def build_chat_prompt(text1, text2):
    """Instruct 模型必须走 chat template。"""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'Text 1: {text1}\nText 2: {text2}'},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def check_label_tokens():
    """四个标签必须是单 token，且在真实 prompt 末尾仍然成立。"""
    ids = []
    for label in LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f'{label} 不是单 token：{token_ids}')
        ids.append(token_ids[0])

    # 关键校验：在 chat template 的 assistant 起始位置之后，
    # 标签是否仍然是同一个 token（防止和换行符合并）
    prefix = build_chat_prompt('train station', 'station')
    a = tokenizer(prefix, add_special_tokens=False)['input_ids']
    b = tokenizer(prefix + 'F', add_special_tokens=False)['input_ids']
    if len(b) != len(a) + 1 or b[-1] != ids[0]:
        raise ValueError(
            '标签在 chat template 之后不是单 token，需要换编码方式'
        )

    print('label_ids:', dict(zip(LABELS, ids)))
    return ids


# ----------------------------------------------------------------- 2. 数据

def build_sft_example(example):
    return {
        'prompt': build_chat_prompt(example['text1'], example['text2']),
        'completion': LABEL_SHORT[example['label']],
    }


def build_real_pairs(train_df):
    """只从原始 Train 构造真实双向组，不把复制的 N 当 pair。"""
    reverse = {
        'FORWARD_ENTAILMENT': 'BACKWARD_ENTAILMENT',
        'BACKWARD_ENTAILMENT': 'FORWARD_ENTAILMENT',
        'EQUIVALENCE': 'EQUIVALENCE',
    }

    pairs = []
    for pair_id, group in train_df.groupby('pair_id'):
        if len(group) == 1:
            if group.iloc[0]['label'] != 'NEGATIVE_OTHER':
                raise ValueError(f'非 N 单例：{pair_id}')
            continue
        if len(group) != 2:
            raise ValueError(f'组内行数异常：{pair_id}')

        a, b = group.iloc[0], group.iloc[1]
        if not (a['text1'] == b['text2'] and a['text2'] == b['text1']):
            raise ValueError(f'文本不是互换关系：{pair_id}')
        if reverse.get(a['label']) != b['label']:
            raise ValueError(f'标签不是合法反转：{pair_id}')

        pairs.append((
            build_chat_prompt(a['text1'], a['text2']),
            build_chat_prompt(b['text1'], b['text2']),
        ))

    return pairs


def collate_pairs(batch):
    prompts = []
    for a, b in batch:
        prompts.append(a)
        prompts.append(b)
    return prompts


# ----------------------------------------------------------------- 3. 损失

def legal_pair_loss(pair_logits):
    """pair_logits: [k, 2, 4]，四类内归一化（必须含 N）。"""
    if pair_logits.ndim != 3 or pair_logits.shape[1:] != (2, 4):
        raise ValueError('pair_logits 形状必须是 [k, 2, 4]')

    log_p = F.log_softmax(pair_logits.float(), dim=-1)

    log_fb = log_p[:, 0, 0] + log_p[:, 1, 1]
    log_bf = log_p[:, 0, 1] + log_p[:, 1, 0]
    log_ee = log_p[:, 0, 2] + log_p[:, 1, 2]

    log_valid = torch.stack([log_fb, log_bf, log_ee], dim=1)

    # -log[P(FB)+P(BF)+P(EE)]，不能再套 softmax 或 CE
    return -torch.logsumexp(log_valid, dim=1).mean()


class ValidPairTrainer(SFTTrainer):
    def __init__(self, pair_loader, pair_tokenizer, label_ids,
                 pair_weight, **kwargs):
        super().__init__(**kwargs)
        self.pair_loader = pair_loader
        self.pair_iterator = iter(pair_loader)
        self.pair_tokenizer = pair_tokenizer
        self.label_ids = label_ids
        self.pair_weight = pair_weight

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        sft_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        # Dev eval_loss 只算 SFT；也不用它选 checkpoint
        if not model.training:
            return (sft_loss, outputs) if return_outputs else sft_loss

        valid_loss = sft_loss.new_zeros(())

        if self.pair_weight > 0:
            try:
                prompts = next(self.pair_iterator)
            except StopIteration:
                self.pair_iterator = iter(self.pair_loader)
                prompts = next(self.pair_iterator)

            pair_inputs = self.pair_tokenizer(
                prompts, padding=True, return_tensors='pt'
            ).to(self.args.device)

            # 左补齐：显式给 position_ids，避免位置错位
            mask = pair_inputs['attention_mask']
            positions = mask.cumsum(dim=1) - 1
            pair_inputs['position_ids'] = positions.masked_fill(mask == 0, 0)

            pair_outputs = model(
                **pair_inputs, use_cache=False, logits_to_keep=1
            )

            # 左补齐时，最后位置就是最后一条真实 token
            label_logits = pair_outputs.logits[:, -1, self.label_ids]
            pair_logits = label_logits.reshape(-1, 2, 4)
            valid_loss = legal_pair_loss(pair_logits)

        loss = sft_loss + self.pair_weight * valid_loss

        if not torch.isfinite(loss):
            raise FloatingPointError('训练损失出现 NaN 或 Inf')

        self._metrics['train']['sft_loss'].append(sft_loss.detach().item())
        self._metrics['train']['valid_loss'].append(valid_loss.detach().item())

        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------- 4. 评估

def evaluate_with_scores(model, dev_df, label_ids, batch_size=8):
    model.eval()
    tokenizer.padding_side = 'left'      # 生成/取最后位置用左补齐
    records = []

    for start in range(0, len(dev_df), batch_size):
        batch = dev_df.iloc[start:start + batch_size]
        prompts = [
            build_chat_prompt(a, b)
            for a, b in zip(batch['text1'], batch['text2'])
        ]

        enc = tokenizer(prompts, padding=True, return_tensors='pt')
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc, logits_to_keep=1)

        scores = out.logits[:, -1, label_ids].float().cpu()   # [n, 4]
        preds = [LABELS[i] for i in scores.argmax(dim=-1).tolist()]

        for (_, row), pred, score in zip(batch.iterrows(), preds, scores):
            records.append({
                'instance_id': row['instance_id'],
                'pair_id': row['pair_id'],
                'gold_label': row['label'],
                'prediction': pred,
                'logit_F': float(score[0]),
                'logit_B': float(score[1]),
                'logit_E': float(score[2]),
                'logit_N': float(score[3]),
            })

    tokenizer.padding_side = 'right'
    return records


# ----------------------------------------------------------------- 5. 主流程

def main():
    os.chdir(SCRIPT_DIR)

    for path in [RUN_DIR, RESULT_DIR]:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'目录已有内容，请改 RUN_NAME：{path}')

    if not torch.cuda.is_available():
        raise RuntimeError('需要 CUDA GPU')

    set_seed(SEED)
    label_ids = check_label_tokens()

    dataset = load_dataset('csv', data_files={
        'train': str(PROJECT_ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'),
        'dev': str(PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'),
    })

    original_train = dataset['train']
    dev_df = dataset['dev'].to_pandas()

    pairs = build_real_pairs(original_train.to_pandas())

    # 复现 B：原始 Train + 472 条 N 的原样副本
    sources = pd.read_csv(SOURCE_CSV, keep_default_na=False)
    source_ids = set(sources['instance_id'])
    extra = original_train.filter(lambda row: row['instance_id'] in source_ids)

    if len(source_ids) != 472 or len(extra) != 472:
        raise ValueError('未复现 B 的 472 条复制来源')
    if set(extra['label']) != {'NEGATIVE_OTHER'}:
        raise ValueError('B 的额外来源必须全是 N')

    dataset['train'] = concatenate_datasets([original_train, extra])
    if len(dataset['train']) != 3514 or len(pairs) != 1282:
        raise ValueError('训练行数或真实 pair 数不符合当前基线')

    sft_dataset = dataset.map(build_sft_example)

    # pair 分支单独一个左补齐 tokenizer，互不影响
    pair_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, padding_side='left'
    )

    pair_generator = torch.Generator().manual_seed(SEED + 1000)
    pair_loader = DataLoader(
        pairs, batch_size=PAIR_BATCH_SIZE, shuffle=True,
        collate_fn=collate_pairs, num_workers=0, generator=pair_generator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype='auto', local_files_only=True
    )

    set_seed(SEED)                      # LoRA 初始化前定种子
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=str(RUN_DIR),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        seed=SEED,
        data_seed=SEED,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=None,
        load_best_model_at_end=False,
        logging_steps=20,
        logging_first_step=True,
        report_to='none',
        loss_type='chunked_nll',
    )

    trainer = ValidPairTrainer(
        model=model, args=args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
        pair_loader=pair_loader,
        pair_tokenizer=pair_tokenizer,
        label_ids=label_ids,
        pair_weight=PAIR_WEIGHT,
    )

    print(f'模型：{MODEL_PATH}')
    print(f'SFT 行数：{len(sft_dataset["train"])}')
    print(f'真实 pair：{len(pairs)}')
    print(f'batch={BATCH_SIZE}  pair_batch={PAIR_BATCH_SIZE}  λ={PAIR_WEIGHT}')

    trainer.train()
    trainer.save_model(str(RUN_DIR / 'final'))
    trainer.save_state()

    # 顺手把每个 epoch 的 checkpoint 都评一遍，
    # 否则没法复现你上次"loss 最低在第1轮、W-F1 最高在第5轮"的现象
    df = pd.DataFrame(evaluate_with_scores(trainer.model, dev_df, label_ids))
    df = pair_aware_decode(df)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_DIR / 'predictions_with_scores.csv',
              index=False, encoding='utf-8-sig')

    rows = []
    for name, column in [('independent', 'prediction'),
                         ('pair_aware', 'pair_prediction')]:
        report = score_one_method(df, column, RESULT_DIR / name)
        n = report.classification.per_label['NEGATIVE_OTHER']
        rows.append({
            'method': name,
            'weighted_f1': report.weighted_f1,
            'soft_cons': report.soft_cons,
            'hard_cons': report.hard_cons,
            'N_precision': n.precision,
            'N_recall': n.recall,
            'N_f1': n.f1,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / 'metrics_summary.csv', index=False)

    (RESULT_DIR / 'run_config.json').write_text(json.dumps({
        'model': MODEL_PATH,
        'prompt': 'chat_template',
        'seed': SEED,
        'pair_weight': PAIR_WEIGHT,
        'batch_size': BATCH_SIZE,
        'pair_batch_size': PAIR_BATCH_SIZE,
        'gradient_accumulation': 1,
        'epochs': EPOCHS,
        'sft_rows': 3514,
        'real_pairs': len(pairs),
        'selection': 'final_epoch',
        'note': 'batch 从 16 降到 8，step 数约翻倍，与 0.6B 版本不同',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print(summary.round(6).to_string(index=False))


if __name__ == '__main__':
    main()