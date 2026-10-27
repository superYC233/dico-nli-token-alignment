
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

# 【修改 1】使用 Trainer 计算四分类 CE
from transformers import AutoTokenizer, TrainingArguments, Trainer

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

# 【修改 2】新目录，保留旧实验
RUN_DIR = Path(
    'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_row4ce_pairce'
) / RUN_NAME

RESULT_DIR = (
    SCRIPT_DIR / 'outputs/qwen3_0p6b_row4ce_pairce' / RUN_NAME
)

if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f'目录已有内容，请修改 RUN_NAME：{RUN_DIR}')

SEED = 42
PAIR_WEIGHT = 0.1
PAIR_BATCH_SIZE = 4
LABELS = ['F', 'B', 'E', 'N']

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 【修改 3】现在只输入 prompt
# 左补齐后，每行最后一个位置都是 prompt 结尾
tokenizer.padding_side = 'left'

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

# 原 SFTTrainer 会自动启用；换 Trainer 后显式保留
model.enable_input_require_grads()

# 【修改 4】使用普通 TrainingArguments
# 不再使用 completion_only_loss、packing、chunked_nll
args = TrainingArguments(
    output_dir=str(RUN_DIR),
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=32,
    bf16=True,
    seed=SEED,
    data_seed=SEED,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=None,
    load_best_model_at_end=False,
    logging_steps=20,
    report_to='none',

    # 保留 prompt / completion，交给 collate_rows 处理
    remove_unused_columns=False,
    label_names=['labels'],

    # 与原 SFTConfig 的默认设置保持一致
    gradient_checkpointing=True,
)


# 【修改 5】新增：整理单条样本
def collate_rows(batch):
    # 只把 prompt 输入模型，不拼接 completion
    prompts = [row['prompt'] for row in batch]

    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors='pt'
    )

    # labels：[n]，是分类编号，不是 token ID
    # F=0、B=1、E=2、N=3
    inputs['labels'] = torch.tensor(
        [LABELS.index(row['completion']) for row in batch],
        dtype=torch.long
    )

    # 左补齐时，有效 token 的位置编号从 0 开始
    mask = inputs['attention_mask']
    positions = mask.cumsum(dim=1) - 1
    inputs['position_ids'] = positions.masked_fill(mask == 0, 0)

    return inputs


def build_real_pairs(train_df):
    # 保存每组真实的组合类别
    pair_label_map = {
        ('F', 'B'): 0,
        ('B', 'F'): 1,
        ('E', 'E'): 2,
    }

    pairs = []

    for pair_id, group in train_df.groupby('pair_id'):
        if len(group) == 1:
            continue

        a, b = group.iloc[0], group.iloc[1]

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


# Pair CE 保持原来的计算方式
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

    pair_scores = torch.stack(
        [score_fb, score_bf, score_ee],
        dim=1
    )

    # N 不参与这个辅助 CE，由单条四分类 CE 监督
    return F.cross_entropy(pair_scores, targets)


def collate_pairs(batch):
    prompts = []
    targets = []

    for a, b, target in batch:
        prompts.append(a)
        prompts.append(b)
        targets.append(target)

    targets = torch.tensor(targets, dtype=torch.long)
    return prompts, targets


# 【修改 6】继承 Trainer，替换单条损失
class PairTrainer(Trainer):
    def __init__(
            self,
            pair_dataloader,
            pair_tokenizer,
            label_ids,
            pair_weight,
            **kwargs
    ):
        super().__init__(**kwargs)

        # 自定义损失已经取平均，不使用额外的 token 数归一化
        self.model_accepts_loss_kwargs = False

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
        # ===== 单条四分类 CE：替换原来的 SFT =====

        # [n]，每条样本的真实分类编号
        row_targets = inputs['labels']

        # 不把 labels 传入语言模型，避免计算全词表 LM loss
        row_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
            use_cache=False,
            logits_to_keep=1
        )

        # [n, 1, V] → [n, 4]
        # 每行只取 F/B/E/N 四个 token 的分数
        row_logits = row_outputs.logits[
            :, -1, self.label_ids
        ].float()

        row_loss = F.cross_entropy(row_logits, row_targets)

        # 返回四类分数给 Trainer
        outputs = {'logits': row_logits}

        # Dev 的 eval_loss 只计算单条四类 CE
        if not model.training:
            if return_outputs:
                return row_loss, outputs
            return row_loss

        # ===== 以下 Pair CE 分支保持原来的逻辑 =====

        try:
            prompts, targets = next(self.pair_iterator)
        except StopIteration:
            self.pair_iterator = iter(self.pair_dataloader)
            prompts, targets = next(self.pair_iterator)

        targets = targets.to(self.args.device)

        pair_inputs = self.pair_tokenizer(
            prompts,
            padding=True,
            return_tensors='pt'
        ).to(self.args.device)

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

        pair_loss = real_pair_ce_loss(pair_logits, targets)

        if not hasattr(self, '_printed'):
            self._printed = True
            print(
                'row_ce', row_loss.item(),
                'pair_loss', pair_loss.item()
            )

        loss = row_loss + self.pair_weight * pair_loss

        if return_outputs:
            return loss, outputs

        return loss


if __name__ == '__main__':
    # 从复制 N 之前的原始数据构造 pair
    pairs = build_real_pairs(train_df)

    # 继续复用原来的 prompt / completion 构造
    # completion 只用于生成分类目标，不会输入模型
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

        # 【修改 7】使用新增的单条样本整理函数
        data_collator=collate_rows,

        pair_dataloader=pair_dataloader,
        pair_tokenizer=pair_tokenizer,
        label_ids=label_ids,
        pair_weight=PAIR_WEIGHT
    )

    trainer.train()
    trainer.save_model(str(RUN_DIR / 'final'))
    trainer.save_state()