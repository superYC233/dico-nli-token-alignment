
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
SCRIPT_DIR = QWEN_DIR

# 项目目录
PROJECT_DIR = SCRIPT_DIR.parent.parent

MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
RUN_NAME = 'run01'

# 【修改 1】新目录，避免覆盖原来的 validpair
RUN_DIR = Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_pairce') / RUN_NAME
RESULT_DIR = SCRIPT_DIR / 'outputs/qwen3_0p6b_pairce' / RUN_NAME

if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f'目录已有内容，请修改 RUN_NAME：{RUN_DIR}')

SEED = 42
PAIR_WEIGHT = 0.1
PAIR_BATCH_SIZE = 4
LABELS = ['F', 'B', 'E', 'N']

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

tokenizer.padding_side = 'right'

pair_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    padding_side='left'
)

label_ids = [
    tokenizer.encode(label, add_special_tokens=False)[0]
    for label in LABELS
]

dataset = load_data()
train_df = dataset['train'].to_pandas()
dev_df = dataset['dev'].to_pandas()

# 保持原来的数据：复制全部 478 条 N
extra = dataset['train'].filter(
    lambda row: row['label'] == 'NEGATIVE_OTHER'
)

dataset['train'] = concatenate_datasets([dataset['train'], extra])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype='auto'
)

model = get_peft_model(model, LoraConfig(
    r=8,
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
    # 【修改 2】保存每组真实的组合类别
    pair_label_map = {
        ('F', 'B'): 0,
        ('B', 'F'): 1,
        ('E', 'E'): 2,
    }

    pairs = []

    # 按照 pair_id 分组
    for pair_id, group in train_df.groupby('pair_id'):
        if len(group) == 1:
            continue

        a, b = group.iloc[0], group.iloc[1]

        # 例如真实标签是 E、E，则 target = 2
        combination = (
            label_map[a['label']],
            label_map[b['label']],
        )
        target = pair_label_map[combination]

        pairs.append((
            build_prompt(a['text1'], a['text2']),
            build_prompt(b['text1'], b['text2']),
            target,
        ))

    return pairs


# 【修改 3】用真实组合 CE 替换原来的合法组合约束
def real_pair_ce_loss(pair_logits, targets):
    '''
    pair_logits：[k, 2, 4]
        k：这一批的 pair 数量
        2：每组两行，按 a、b 的顺序
        4：F / B / E / N 四个分数

    targets：[k]
        0=FB，1=BF，2=EE
    '''
    pair_logits = pair_logits.float()

    score_fb = pair_logits[:, 0, 0] + pair_logits[:, 1, 1]
    score_bf = pair_logits[:, 0, 1] + pair_logits[:, 1, 0]
    score_ee = pair_logits[:, 0, 2] + pair_logits[:, 1, 2]

    # [k, 3]：FB / BF / EE 三个组合的分数
    pair_scores = torch.stack(
        [score_fb, score_bf, score_ee],
        dim=1
    )

    # 要求真实组合得分最高
    # 这个辅助 CE 不直接约束 N，N 仍由 SFT 监督
    return F.cross_entropy(pair_scores, targets)


# 【修改 4】同时返回 prompt 和真实组合标签
def collate_pairs(batch):
    # batch = [(a1, b1, target1), (a2, b2, target2), ...]
    # prompts = [a1, b1, a2, b2, ...]
    prompts = []
    targets = []

    for a, b, target in batch:
        prompts.append(a)
        prompts.append(b)
        targets.append(target)

    targets = torch.tensor(targets, dtype=torch.long)
    return prompts, targets


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

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None
    ):
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

        # 【修改 5】读取并传递真实组合标签
        try:
            prompts, targets = next(self.pair_iterator)
        except StopIteration:
            self.pair_iterator = iter(self.pair_dataloader)
            prompts, targets = next(self.pair_iterator)

        targets = targets.to('cuda')

        pair_inputs = self.pair_tokenizer(
            prompts,
            padding=True,
            return_tensors='pt'
        ).to('cuda')

        # pair 分支采用左补齐，最后位置都是有效的 prompt
        # 每条样本的有效 token 位置编号从 0 开始
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

        # [2k, 1, V] → [2k, 4]
        pair_logits = pair_outputs.logits[:, -1, self.label_ids]

        # [2k, 4] → [k, 2, 4]
        pair_logits = pair_logits.reshape(-1, 2, 4)

        # 【修改 5】传入真实组合标签，计算三分类 CE
        pair_loss = real_pair_ce_loss(pair_logits, targets)

        if not hasattr(self, '_printed'):
            self._printed = True
            print(
                'sft_loss', sft_loss.item(),
                'pair_loss', pair_loss.item()
            )

        loss = sft_loss + self.pair_weight * pair_loss

        if return_outputs:
            return loss, sft_outputs

        return loss


if __name__ == '__main__':
    # train_df 是复制 N 之前的原始数据
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