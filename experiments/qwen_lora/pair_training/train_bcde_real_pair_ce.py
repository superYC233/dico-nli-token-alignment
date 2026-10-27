
# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import gc
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


SCRIPT_DIR = QWEN_DIR
PROJECT_DIR = SCRIPT_DIR.parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import (
    load_data, build_prompt, build_sft_example, label_map, tokenizer
)
from train_n_ablation import (
    build_train_dataset, COMMON_SFT, LORA_CONFIG
)
from compare_decoding import evaluate_with_scores
from analyze_decoding import pair_aware_decode, score_one_method


MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'

MODES = ['B', 'C', 'D', 'E']
RUN_NAME = 'run01'
SEED = 42

PAIR_WEIGHT = 0.1
PAIR_BATCH_SIZE = 4
LABELS = ['F', 'B', 'E', 'N']

# 与旧模型分开保存
RUN_ROOT = (
    Path('D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_ablation_pairce')
    / RUN_NAME
)

OUTPUT_ROOT = (
    SCRIPT_DIR / 'outputs/n_ablation_real_pair_ce' / RUN_NAME
)

# 固定使用之前筛选的 472 条 N 来源
SOURCE_CSV = SCRIPT_DIR / 'outputs/n_ablation/swap_sources.csv'

# 已有的不加训练期 Pair CE 的结果
OLD_SUMMARY = (
    SCRIPT_DIR / 'outputs/n_ablation_pair_aware/run01/summary.csv'
)


def build_real_pairs(train_df):
    # 只使用原始 Train 的真实双向组
    pair_label_map = {
        ('F', 'B'): 0,
        ('B', 'F'): 1,
        ('E', 'E'): 2,
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

        if not (
            a['text1'] == b['text2']
            and a['text2'] == b['text1']
        ):
            raise ValueError(f'文本不是互换关系：{pair_id}')

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


def real_pair_ce_loss(pair_logits, targets):
    # pair_logits：[k, 2, 4]
    # targets：[k]，0=FB、1=BF、2=EE
    pair_logits = pair_logits.float()

    score_fb = pair_logits[:, 0, 0] + pair_logits[:, 1, 1]
    score_bf = pair_logits[:, 0, 1] + pair_logits[:, 1, 0]
    score_ee = pair_logits[:, 0, 2] + pair_logits[:, 1, 2]

    # [k, 3]
    pair_scores = torch.stack(
        [score_fb, score_bf, score_ee],
        dim=1,
    )

    return F.cross_entropy(pair_scores, targets)


def collate_pairs(batch):
    prompts = []
    targets = []

    for a, b, target in batch:
        prompts.extend([a, b])
        targets.append(target)

    return prompts, torch.tensor(targets, dtype=torch.long)


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
        # 保留原 B/C/D/E 的全词表 SFT
        sft_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        # Dev 的 eval_loss 只计算 SFT
        if not model.training:
            if return_outputs:
                return sft_loss, outputs
            return sft_loss

        try:
            prompts, targets = next(self.pair_iterator)
        except StopIteration:
            self.pair_iterator = iter(self.pair_dataloader)
            prompts, targets = next(self.pair_iterator)

        targets = targets.to(self.args.device)

        pair_inputs = self.pair_tokenizer(
            prompts,
            padding=True,
            return_tensors='pt',
        ).to(self.args.device)

        # 左补齐后，最后位置都是有效 prompt
        mask = pair_inputs['attention_mask']
        positions = mask.cumsum(dim=1) - 1
        pair_inputs['position_ids'] = positions.masked_fill(
            mask == 0, 0
        )

        pair_outputs = model(
            **pair_inputs,
            use_cache=False,
            logits_to_keep=1,
        )

        # [2k, 1, V] → [2k, 4] → [k, 2, 4]
        pair_logits = pair_outputs.logits[
            :, -1, self.label_ids
        ].reshape(-1, 2, 4)

        pair_loss = real_pair_ce_loss(pair_logits, targets)
        loss = sft_loss + self.pair_weight * pair_loss

        if not torch.isfinite(loss):
            raise FloatingPointError('损失出现 NaN 或 Inf')

        # 分别记录两个损失
        self._metrics['train']['sft_loss'].append(
            sft_loss.detach().item()
        )
        self._metrics['train']['pair_ce'].append(
            pair_loss.detach().item()
        )

        if return_outputs:
            return loss, outputs
        return loss


def train_one(mode, dataset, sources, pairs):
    set_seed(SEED)

    # 复用旧实验的数据构造，不修改原始 dataset
    train_ds, note, audit = build_train_dataset(
        dataset['train'], sources, mode
    )

    expected = 3514 if mode in ['B', 'C'] else 6084
    if len(train_ds) != expected:
        raise ValueError(
            f'{mode} 的训练行数异常：{len(train_ds)}'
        )

    print(f'\n===== {mode} + Real Pair CE =====')
    print('数据构造：', note)
    print('SFT 行数：', len(train_ds))
    print('真实 pair 数：', len(pairs))
    print('Pair CE 权重：', PAIR_WEIGHT)

    if audit:
        print('增强记录：', audit)

    sft_train = train_ds.map(build_sft_example)
    sft_dev = dataset['dev'].map(build_sft_example)

    # 上一组评估会把共享 tokenizer 改成左补齐
    # 每组训练开始前重新设为右补齐
    tokenizer.padding_side = 'right'

    pair_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        padding_side='left',
    )

    token_ids = [
        pair_tokenizer.encode(
            label,
            add_special_tokens=False,
        )
        for label in LABELS
    ]

    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('F/B/E/N 必须分别是单 token')

    label_ids = [ids[0] for ids in token_ids]

    # 每组从 Base 重新开始，不加载旧 Adapter
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype='auto',
        local_files_only=True,
    )

    # 与原消融脚本的初始化位置一致
    set_seed(SEED)
    model = get_peft_model(
        model,
        LoraConfig(**LORA_CONFIG),
    )

    run_dir = RUN_ROOT / f'run_{mode}'
    output_dir = OUTPUT_ROOT / f'run_{mode}'

    # 复用原消融参数：
    # r=8、alpha=16、lr=1e-4、5 epochs、batch=16
    settings = COMMON_SFT.copy()
    settings.update(
        seed=SEED,
        data_seed=SEED,
        loss_type='chunked_nll',
    )

    args = SFTConfig(
        output_dir=str(run_dir),
        **settings,
    )

    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            '本版固定梯度累积为 1，与原消融一致'
        )

    # Pair 单独打乱，避免占用 row sampler 的随机生成器
    pair_generator = torch.Generator().manual_seed(SEED + 1000)

    pair_dataloader = DataLoader(
        pairs,
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=0,
        generator=pair_generator,
    )

    trainer = PairTrainer(
        model=model,
        args=args,
        train_dataset=sft_train,
        eval_dataset=sft_dev,
        processing_class=tokenizer,
        pair_dataloader=pair_dataloader,
        pair_tokenizer=pair_tokenizer,
        label_ids=label_ids,
        pair_weight=PAIR_WEIGHT,
    )

    trainer.train()
    trainer.save_model(str(run_dir / 'final'))
    trainer.save_state()

    # 固定最后一轮，与原 B/C/D/E 比较
    trainer.model.eval()

    records = evaluate_with_scores(
        trainer.model,
        batch_size=16,
        split='dev',
    )

    df = pair_aware_decode(pd.DataFrame(records))

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_dir / 'predictions_with_scores.csv',
        index=False,
        encoding='utf-8-sig',
    )

    rows = []

    for method, column in [
        ('independent', 'prediction'),
        ('pair_aware', 'pair_prediction'),
    ]:
        report = score_one_method(
            df,
            column,
            output_dir / method,
        )

        n = report.classification.per_label['NEGATIVE_OTHER']

        rows.append({
            'model': mode,
            'method': method,
            'weighted_f1': report.weighted_f1,
            'soft_cons': report.soft_cons,
            'hard_cons': report.hard_cons,
            'N_precision': n.precision,
            'N_recall': n.recall,
            'N_f1': n.f1,
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        output_dir / 'metrics_summary.csv',
        index=False,
    )

    print(result.round(6).to_string(index=False))
    return rows


def main():
    os.chdir(SCRIPT_DIR)

    # 开始前检查所有目标目录，避免覆盖旧实验
    for mode in MODES:
        for path in [
            RUN_ROOT / f'run_{mode}',
            OUTPUT_ROOT / f'run_{mode}',
        ]:
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(
                    f'目录已有内容，请修改 RUN_NAME：{path}'
                )

    if not torch.cuda.is_available():
        raise RuntimeError('当前配置需要 CUDA GPU')

    dataset = load_data()

    sources = pd.read_csv(
        SOURCE_CSV,
        keep_default_na=False,
    )
    source_ids = set(sources['instance_id'])

    picked = dataset['train'].filter(
        lambda row: row['instance_id'] in source_ids
    )

    if (
        len(sources) != 472
        or sources['instance_id'].nunique() != 472
        or len(picked) != 472
        or set(picked['label']) != {'NEGATIVE_OTHER'}
    ):
        raise ValueError('固定的 472 条 N 来源不符合预期')

    # 四组都只从原始 Train 提取这 1282 个真实 pair
    pairs = build_real_pairs(
        dataset['train'].to_pandas()
    )

    if len(pairs) != 1282:
        raise ValueError(f'真实 pair 数异常：{len(pairs)}')

    all_rows = []

    for mode in MODES:
        rows = train_one(
            mode,
            dataset,
            sources,
            pairs,
        )
        all_rows.extend(rows)

        # 每完成一组就保存一次汇总
        pd.DataFrame(all_rows).to_csv(
            OUTPUT_ROOT / 'summary.csv',
            index=False,
            encoding='utf-8-sig',
        )

        # 释放当前组，接着训练下一组
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(all_rows)

    print('\n===== 四组新结果 =====')
    print(summary.round(6).to_string(index=False))

    # 自动比较：相同数据组，加 Pair CE 前后有何变化
    if OLD_SUMMARY.exists():
        old = pd.read_csv(OLD_SUMMARY)
        columns = ['model', 'method', 'weighted_f1']

        comparison = old[columns].merge(
            summary[columns],
            on=['model', 'method'],
            suffixes=('_old', '_new'),
        )

        # 差值单位为百分点
        comparison['delta_pp'] = (
            comparison['weighted_f1_new']
            - comparison['weighted_f1_old']
        ) * 100

        comparison.to_csv(
            OUTPUT_ROOT / 'comparison.csv',
            index=False,
            encoding='utf-8-sig',
        )

        print('\n===== 与原实验比较：delta_pp 单位为百分点 =====')
        print(comparison.round(6).to_string(index=False))

    print('\n汇总文件：', OUTPUT_ROOT / 'summary.csv')


if __name__ == '__main__':
    main()