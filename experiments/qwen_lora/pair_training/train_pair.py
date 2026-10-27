
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import sys
from pathlib import Path
from torch.utils.data import DataLoader

import torch
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from analyze_decoding import pair_aware_decode, score_one_method
from data_utils import load_data, build_prompt, build_sft_example, label_map
import os
import pandas as pd
from datasets import concatenate_datasets
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed

SEED = 42
set_seed(SEED)

# 当前路径
# __file__ 当前文件的路径
# Path(...)	变成路径对象，方便后面拼路径
# .resolve()	变成完整绝对路径
SCRIPT_DIR = QWEN_DIR
# 项目目录
PROJECT_DIR = SCRIPT_DIR.parent.parent

MODEL_PATH      = 'D:/LLM/model/Qwen3-0.6B-Base'
RUN_NAME        = 'run01'

RUN_DIR    = Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_validpair') / RUN_NAME
RESULT_DIR = SCRIPT_DIR / 'outputs/qwen3_0p6b_validpair' / RUN_NAME

SEED            = 42
PAIR_WEIGHT     = 0.1
PAIR_BATCH_SIZE = 4
LABELS          = ['F', 'B', 'E', 'N']

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'right'

pair_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')

label_ids = [
    tokenizer.encode(label, add_special_tokens=False)[0]
    for label in LABELS
]

dataset = load_data()
train_df = dataset['train'].to_pandas()
dev_df = dataset['dev'].to_pandas()

extra = dataset['train'].filter(
    lambda row: row['label'] == 'NEGATIVE_OTHER'
)

dataset['train'] = concatenate_datasets([dataset['train'], extra])

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype='auto')
model = get_peft_model(model, LoraConfig(
    r = 8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules='all-linear',
    bias='none',
    task_type='CAUSAL_LM'
))

args = SFTConfig(
    output_dir=str(RUN_DIR),
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=32,
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
    report_to='none',
    loss_type='chunked_nll',
)

def build_real_pairs(train_df):
    reverse = {
        'FORWARD_ENTAILMENT': 'BACKWARD_ENTAILMENT',
        'BACKWARD_ENTAILMENT': 'FORWARD_ENTAILMENT',
        'EQUIVALENCE': 'EQUIVALENCE',
    }

    pairs = []

    # 按照pair_id分组
    for pair_id, group in train_df.groupby('pair_id'):
        if len(group) == 1:
            continue

        a, b = group.iloc[0], group.iloc[1]

        pairs.append((
            build_prompt(a['text1'], a['text2']),
            build_prompt(b['text1'], b['text2']),
        ))

    return pairs

def legal_pair_loss(pair_logits):
    '''
    合法组合约束。

    pair_logits 形状：[k, 2, 4]
        k = 这一批有几组 pair
        2 = 每组两行（第 0 行 = 正方向，第 1 行 = 反方向）
        4 = 每行四个分数，顺序固定是 F / B / E / N

    返回一个标量：-log[ P(FB) + P(BF) + P(EE) ]，再对 k 组取平均。
    '''

    log_p = F.log_softmax(pair_logits.float(), dim=-1)

    log_fb = log_p[:, 0, 0] + log_p[:, 1, 1]

    log_bf = log_p[:,0, 1] + log_p[:, 1, 0]

    log_ee = log_p[:, 0, 2] + log_p[:, 1, 2]

    # 把三个组合叠成[k, 3]
    # 每一组三个数，分别是 FB / BF / EE 的对数概率
    log_pair = torch.stack([log_fb, log_bf, log_ee], dim=-1)

    # 算 - log[P(FB) + P(BF) + P(EE)]
    return -torch.logsumexp(log_pair, dim=1).mean()

def collate_pairs(batch):
    # batch = [(a1,b1), (a2,b2), ...]
    # 摊成 [a1, b1, a2, b2, ...]
    prompts = []

    for a, b in batch:
        prompts.append(a)
        prompts.append(b)
    return prompts

class PairTrainer(SFTTrainer):
    def __init__(
            self,
            pair_dataloader,
            pair_tokenizer,
            label_ids,
            pair_weight,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.pair_dataloader = pair_dataloader
        self.pair_iterator = iter(pair_dataloader)
        self.pair_tokenizer = pair_tokenizer
        self.label_ids = label_ids
        self.pair_weight = pair_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sft_loss, sft_outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if not model.training:
            if return_outputs:
                return sft_loss, sft_outputs
            return sft_loss

        try:
            prompts = next(self.pair_iterator)
        except StopIteration:
            self.pair_iterator = iter(self.pair_dataloader)
            prompts = next(self.pair_iterator)

        pair_inputs = self.pair_tokenizer(
            prompts,
            padding=True,
            return_tensors='pt'
        ).to('cuda')

        # pair 分支采用左补齐, 最后位置都是有效的prompt
        # 每条样本的有效token位置编号从 0 开始
        mask = pair_inputs['attention_mask']
        positions = mask.cumsum(dim=1) - 1
        pair_inputs['position_ids'] = (
            positions.masked_fill(mask == 0, 0)
        )

        pair_outputs = model(
            **pair_inputs,
            use_cache=False,
            logits_to_keep=1
        )

        pair_logits = pair_outputs.logits[:, -1, self.label_ids]

        pair_logits = pair_logits.reshape(-1,2,4)

        pair_loss = legal_pair_loss(pair_logits)

        if not hasattr(self, '_printed'):
            self._printed = True
            print('sft_loss', sft_loss.item(), 'pair_loss', pair_loss.item())

        loss = sft_loss + self.pair_weight * pair_loss

        if return_outputs:
            return loss, sft_outputs

        return loss

if __name__ == '__main__':
    pairs = build_real_pairs(train_df)

    sft_dataset = dataset.map(build_sft_example)

    pair_dataloader = DataLoader(
        pairs,
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pairs
    )

    trainer = PairTrainer(
        model=model,
        args=args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
        pair_dataloader=pair_dataloader,
        pair_tokenizer=pair_tokenizer,
        label_ids=label_ids,
        pair_weight=PAIR_WEIGHT
    )

    trainer.train()
    trainer.save_model(str(RUN_DIR / 'final'))
    trainer.save_state()
