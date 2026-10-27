"""C + r16 + 旧短 Prompt + 答案位置四候选 R-Drop。"""

# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

script_dir = QWEN_DIR
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from best_f1_callback import BestF1Callback
from data_utils import load_data, get_datasets, tokenizer, build_prompt
from data_variants import build_train_dataset


# 本次只新增 R-Drop；其余沿用 C + r16 的配置。
MODE = 'C'
SEED = 42
RDROP_WEIGHT = 1.0  # 本次探索固定为 1.0，不代表已经找到最优值。
MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
OUTPUT_DIR = f'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_rdrop/run01/{MODE}'


def symmetric_kl(scores1, scores2):
    """每条样本在四类内归一化，两个方向的 KL 取平均。"""
    log_p = F.log_softmax(scores1.float(), dim=-1)
    log_q = F.log_softmax(scores2.float(), dim=-1)
    # 不 detach：两次前向都要接收一致性损失的梯度。
    kl_pq = F.kl_div(log_q, log_p.exp(), reduction='batchmean')
    kl_qp = F.kl_div(log_p, log_q.exp(), reduction='batchmean')
    return 0.5 * (kl_pq + kl_qp)


class RDropTrainer(SFTTrainer):
    def __init__(self, label_ids, rdrop_weight, **kwargs):
        super().__init__(**kwargs)
        self.label_ids = label_ids
        self.rdrop_weight = rdrop_weight

        if self.padding_free or self.args.packing:
            raise ValueError('本脚本需要普通补齐批次，请关闭 packing 和 padding_free。')
        if not self.completion_only_loss:
            raise ValueError('需要 completion_only_loss=True，才能定位答案。')

        base = self.model.get_base_model()
        if base.config.model_type != 'qwen3':
            raise ValueError('本脚本的四候选投影仅针对当前 Qwen3 模型。')
        if not isinstance(base.get_output_embeddings(), torch.nn.Linear):
            raise ValueError('本脚本要求原始线性 lm_head，不支持在 lm_head 上加入 Adapter。')

    def forward_with_scores(self, model, inputs, positions, num_items_in_batch):
        """一次正常 SFT 前向，同时取出预测答案的四个分数。"""
        base = self.accelerator.unwrap_model(model).get_base_model()
        head = base.get_output_embeddings()
        captured_scores = []

        def capture_answer_scores(module, args, outputs):
            # outputs.last_hidden_state：[批大小, 序列长度, 隐藏维度]
            hidden = outputs.last_hidden_state
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            answer_hidden = hidden[rows, positions]

            # 只计算 F/B/E/N 四列，避免额外生成整段输入的全词表 logits。
            weight = head.weight[self.label_ids]
            bias = None if head.bias is None else head.bias[self.label_ids]
            scores = F.linear(answer_hidden, weight, bias)
            captured_scores.append(scores.float())

        # 临时观察 Qwen 主干的输出；不修改它，也不切断梯度。
        handle = base.base_model.register_forward_hook(capture_answer_scores)
        try:
            loss, outputs = super().compute_loss(
                model,
                dict(inputs),
                return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )
        finally:
            # 立即移除，避免干扰后续反向重算、评估和生成。
            handle.remove()

        if len(captured_scores) != 1:
            raise ValueError('未能正确取得答案位置分数，请检查模型结构。')
        return loss, outputs, captured_scores[0]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 常规 Dev loss 保留原有 SFT 定义；选模型仍看回调的 pair-aware F1。
        if not model.training:
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch
            )

        labels = inputs['labels']
        supervised = labels != -100
        if not supervised.any(dim=1).all():
            raise ValueError('批次中存在没有答案监督的样本。')

        # Prompt 的 labels 是 -100；第一个非 -100 位置就是 F/B/E/N。
        first_answer = supervised.long().argmax(dim=1)
        if (first_answer == 0).any():
            raise ValueError('答案前缺少 Prompt，无法取得预测答案的位置。')

        rows = torch.arange(labels.shape[0], device=labels.device)
        answer_tokens = labels[rows, first_answer]
        allowed = torch.tensor(self.label_ids, device=labels.device)
        if not torch.isin(answer_tokens, allowed).all():
            raise ValueError('答案首 token 不是 F/B/E/N，请检查 Prompt 和 completion。')

        # 因果语言模型：位置 t-1 的输出，预测位置 t 的答案。
        positions = first_answer - 1

        # 同样的输入做两次前向；不要重置随机种子，保留不同 Dropout。
        loss1, outputs1, scores1 = self.forward_with_scores(
            model, inputs, positions, num_items_in_batch
        )
        loss2, _, scores2 = self.forward_with_scores(
            model, inputs, positions, num_items_in_batch
        )

        # SFT 仍包含原来的全词表分类、completion mask 和 EOS 监督。
        sft_loss = 0.5 * (loss1 + loss2)
        kl_loss = symmetric_kl(scores1, scores2)
        loss = sft_loss + self.rdrop_weight * kl_loss

        # 每 logging_steps 次，分别报告两部分，方便判断是否正常工作。
        self._metrics['train']['sft_loss'].append(sft_loss.detach().item())
        self._metrics['train']['rdrop_kl'].append(kl_loss.detach().item())
        self._metrics['train']['rdrop_weighted_kl'].append(
            self.rdrop_weight * kl_loss.detach().item()
        )
        return (loss, outputs1) if return_outputs else loss


def main():
    # 兼容已有 load_data 和评分函数中的相对路径。
    os.chdir(script_dir)
    set_seed(SEED)

    prompt = build_prompt('example A', 'example B')
    if 'Check both directions separately:' in prompt:
        raise ValueError('当前仍是长 Prompt，请先恢复旧短 Prompt。')

    token_ids = [tokenizer.encode(label, add_special_tokens=False)
                 for label in ['F', 'B', 'E', 'N']]
    if not all(len(ids) == 1 for ids in token_ids):
        raise ValueError('每个标签必须恰好对应一个 token。')
    label_ids = [ids[0] for ids in token_ids]
    if len(set(label_ids)) != 4:
        raise ValueError('四个标签的 token ID 必须互不相同。')

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype='auto')
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
        bias='none',
        task_type='CAUSAL_LM'
    )
    lora_model = get_peft_model(model, lora_config)

    train_args = SFTConfig(
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        padding_free=False,
        # 与当前 TRL 默认一致，仍是全词表 SFT，只做分块以节省显存。
        loss_type='chunked_nll',
        seed=SEED,
        data_seed=SEED,
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch',
        save_strategy='best',
        load_best_model_at_end=True,
        metric_for_best_model='eval_pair_f1',
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        logging_steps=20,
        report_to='none'
    )

    dataset = load_data()
    original_size = len(dataset['train'])
    dataset['train'] = build_train_dataset(dataset['train'], mode=MODE)
    sft_dataset = get_datasets(dataset)
    tokenizer.padding_side = 'right'

    print('当前版本：', MODE)
    print('原始训练样本数：', original_size)
    print('构造后训练样本数：', len(dataset['train']))
    print('Dev 样本数：', len(dataset['dev']))
    print('R-Drop 权重：', RDROP_WEIGHT)
    print('模型保存目录：', OUTPUT_DIR)

    trainer = RDropTrainer(
        model=lora_model,
        args=train_args,
        train_dataset=sft_dataset['train'],
        eval_dataset=sft_dataset['dev'],
        processing_class=tokenizer,
        callbacks=[BestF1Callback()],
        label_ids=label_ids,
        rdrop_weight=RDROP_WEIGHT
    )

    # 确认实际训练输入与当前模板一致，避免再次误用旧缓存。
    expected_ids = tokenizer.encode(
        sft_dataset['train'][0]['prompt'], add_special_tokens=False
    )
    actual_ids = trainer.train_dataset[0]['input_ids']
    if actual_ids[:len(expected_ids)] != expected_ids:
        raise ValueError('实际训练输入与当前 Prompt 不一致，请检查缓存。')

    trainer.train()

    best_dir = f'{OUTPUT_DIR}/best_adapter'
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳 Adapter：', best_dir)


if __name__ == '__main__':
    main()
