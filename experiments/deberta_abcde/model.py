"""原 train.py 的 Token Alignment 模型，改为显式传入主干和 tokenizer。

计算公式、参数名称和初始化顺序保持不变；导入本文件不会加载预训练权重。
"""
import torch
from torch import nn
import torch.nn.functional as F


class DebertaDiCoModel(nn.Module):
    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = self.deberta_model.config.hidden_size

        # 整体 CLS 四分类分支。
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(hidden_size, 4)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

        # 保留原版随机数隔离和零初始化。
        with torch.random.fork_rng(devices=[]):
            self.align_projection = nn.Linear(hidden_size, 128)
            self.compare_layer = nn.Sequential(
                nn.Linear(4 * 128, 128), nn.ReLU(),
            )
            self.match_output = nn.Sequential(
                nn.LayerNorm(128 * 4), nn.Dropout(p=0.2), nn.Linear(128 * 4, 4),
            )
            nn.init.zeros_(self.match_output[-1].weight)
            nn.init.zeros_(self.match_output[-1].bias)

    def _align(self, tokens, scores, target_mask):
        mask = target_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(weights, tokens)

    def _pool(self, features, mask):
        float_mask = mask.unsqueeze(-1).to(features.dtype)
        mean_feature = (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)
        max_feature = features.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(features.dtype).min,
        ).max(dim=1).values
        max_feature = torch.where(
            mask.any(dim=1, keepdim=True), max_feature, torch.zeros_like(max_feature),
        )
        return torch.cat([mean_feature, max_feature], dim=-1)

    def forward(self, input_ids, attention_mask):
        deberta_output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = deberta_output.last_hidden_state
        cls_logits = self.output(self.dropout(last_hidden_state[:, 0, :]))

        # [CLS] A [SEP] B [SEP]：根据 SEP 次数区分两个文本。
        is_sep = input_ids == self.sep_id
        sep_count = is_sep.long().cumsum(dim=1)
        valid = attention_mask.bool() & ~is_sep & (input_ids != self.cls_id)
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)

        tokens = self.align_projection(last_hidden_state)
        normalized = F.normalize(tokens, dim=-1)
        scores = torch.bmm(normalized, normalized.transpose(1, 2)) / 0.1
        aligned_b = self._align(tokens, scores, mask_b)
        aligned_a = self._align(tokens, scores, mask_a)
        features_a = self.compare_layer(torch.cat(
            [tokens, aligned_b, tokens - aligned_b, tokens * aligned_b], dim=-1,
        ))
        features_b = self.compare_layer(torch.cat(
            [tokens, aligned_a, tokens - aligned_a, tokens * aligned_a], dim=-1,
        ))
        pair_feature = torch.cat(
            [self._pool(features_a, mask_a), self._pool(features_b, mask_b)], dim=-1,
        )
        match_logits = self.match_output(pair_feature)
        return cls_logits + match_logits
