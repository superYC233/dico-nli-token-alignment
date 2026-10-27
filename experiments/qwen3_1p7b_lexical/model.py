"""Qwen 的全词表 SFT：只在答案标签位置加入四类词汇修正。"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights


class QwenLexicalModel(nn.Module):
    accepts_loss_kwargs = True
    loss_type = 'ForCausalLM'

    def __init__(self, language_model, label_ids, eos_token_id,
                 feature_size=20, lexical_hidden=32, chunk_size=8, use_lexical_features=True):
        super().__init__()
        self.language_model = language_model
        self.config = language_model.config
        self.eos_token_id = eos_token_id
        self.feature_size = feature_size
        self.chunk_size = chunk_size
        self.use_lexical_features = use_lexical_features
        if len(label_ids) != 4 or len(set(label_ids)) != 4 or eos_token_id in label_ids:
            raise ValueError('四个标签必须是不同 token，且不能使用 EOS。')
        if chunk_size < 1:
            raise ValueError('chunk_size 必须大于 0。')
        self.register_buffer('label_ids', torch.tensor(label_ids, dtype=torch.long))
        # 独立初始化，避免推进 LoRA/dropout 后续使用的随机序列。
        with torch.random.fork_rng(devices=[]):
            self.lexical_head = nn.Sequential(
                nn.Linear(feature_size, lexical_hidden), nn.ReLU(), nn.Linear(lexical_hidden, 4),
            ).float()
            nn.init.zeros_(self.lexical_head[-1].weight)
            nn.init.zeros_(self.lexical_head[-1].bias)
        if hasattr(language_model.get_output_embeddings(), 'lora_A'):
            raise ValueError('此分块实现要求 lm_head 不挂 LoRA；请保持原版 all-linear 的默认排除。')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs,
        )
        self.language_model.enable_input_require_grads()

    def lexical_correction(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError('词汇特征必须是 [batch, 20]。')
        if not torch.isfinite(features).all():
            raise ValueError('词汇特征出现非法数值。')
        if not self.use_lexical_features:
            return features.new_zeros((features.shape[0], 4), dtype=torch.float32)
        # 小分支保持 float32，避免 bf16 下微小修正被过早舍入。
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.lexical_head(features.float())

    def _token_loss(self, hidden, correction, targets, answer_mask):
        lm_head = self.language_model.get_output_embeddings()
        logits = F.linear(hidden, lm_head.weight, lm_head.bias).float()
        # 分母仍包含全部词表，不是四分类 CE；EOS 行的 correction 被置为 0。
        values = correction * answer_mask[:, None]
        logits = logits.index_add(1, self.label_ids, values)
        return F.cross_entropy(logits, targets, reduction='sum')

    def forward(self, input_ids, attention_mask, lexical_features, labels=None,
                num_items_in_batch=None, **kwargs):
        if labels is None:
            raise ValueError('训练 forward 需要 labels；预测请使用 predict_with_scores。')
        if lexical_features.shape != (input_ids.shape[0], self.feature_size):
            raise ValueError('词汇特征与当前输入批次没有对齐。')
        supervised = labels != -100
        # 本实验每行严格监督一个分类标签和一个 EOS。
        if not (supervised.sum(dim=1) == 2).all() or supervised[:, 0].any():
            raise ValueError('每行必须只监督标签 + EOS，且答案前必须有 Prompt。')
        positions = supervised.long().argmax(dim=1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        if not (labels[rows, positions, None] == self.label_ids).any(dim=1).all():
            raise ValueError('第一个监督 token 不是 F/B/E/N。')
        if not (labels[rows, positions + 1] == self.eos_token_id).all():
            raise ValueError('标签后必须紧跟 EOS。')

        # LoRA 层已在这个 decoder 内；不额外跑一次 Qwen，也不返回所有层隐藏状态。
        causal_lm = self.language_model.get_base_model()
        output = causal_lm.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
        )
        # 位置 i 的隐藏向量预测 token i+1，只投影需要监督的位置。
        batch_indices, token_indices = torch.where(labels[:, 1:] != -100)
        hidden = output.last_hidden_state[batch_indices, token_indices]
        targets = labels[batch_indices, token_indices + 1]
        answer_mask = token_indices == positions[batch_indices] - 1
        correction = self.lexical_correction(lexical_features)[batch_indices]

        total_loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(targets), self.chunk_size):
            end = start + self.chunk_size
            arguments = (hidden[start:end], correction[start:end], targets[start:end], answer_mask[start:end])
            if self.training and torch.is_grad_enabled():
                # 反向时重算小块 logits，避免为每个块保留完整词表张量。
                loss = checkpoint(self._token_loss, *arguments, use_reentrant=False)
            else:
                loss = self._token_loss(*arguments)
            total_loss = total_loss + loss
        denominator = len(targets) if num_items_in_batch is None else num_items_in_batch
        loss = total_loss / denominator
        return CausalLMOutputWithPast(loss=loss)

    def save_adapter_and_head(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(str(directory), safe_serialization=True)
        state = {name: tensor.detach().cpu().float() for name, tensor in self.lexical_head.state_dict().items()}
        torch.save(state, directory / 'lexical_head.pt')

    def load_adapter_and_head(self, directory):
        directory = Path(directory)
        weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
        set_peft_model_state_dict(self.language_model, weights, adapter_name='default')
        state = torch.load(directory / 'lexical_head.pt', map_location='cpu', weights_only=True)
        self.lexical_head.load_state_dict(state, strict=True)
