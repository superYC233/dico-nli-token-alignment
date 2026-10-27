"""原Token Alignment的特征提取 + 单一特征融合四分类头，不使用CNN。"""
import torch
from torch import nn
import torch.nn.functional as F


class DebertaDiCoModel(nn.Module):
    def __init__(
        self, backbone, tokenizer, feature_size, lexical_hidden=32,
        alignment_dimension=128, alignment_temperature=0.1,
        fusion_hidden=128, dropout=0.2,
    ):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = backbone.config.hidden_size
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.feature_size = feature_size
        self.alignment_temperature = alignment_temperature
        self.dropout = nn.Dropout(p=dropout)

        # 与原版相同的token投影、比较、mean/max汇总。
        self.align_projection = nn.Linear(hidden_size, alignment_dimension)
        self.compare_layer = nn.Sequential(
            nn.Linear(4 * alignment_dimension, alignment_dimension), nn.ReLU(),
        )

        # 三条分支输出特征，不各自输出四分类分数。
        self.cls_norm = nn.LayerNorm(hidden_size)
        self.match_norm = nn.LayerNorm(4 * alignment_dimension)
        self.lexical_encoder = nn.Sequential(
            nn.Linear(feature_size, lexical_hidden),
            nn.ReLU(),
            nn.LayerNorm(lexical_hidden),
        )

        self.fusion_input_size = hidden_size + 4 * alignment_dimension + lexical_hidden
        self.fusion_output = nn.Sequential(
            nn.Linear(self.fusion_input_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, 4),
        )
        # 使用PyTorch正常初始化，第一批Row CE即可反传到三个特征分支。

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

    def forward(self, input_ids, attention_mask, lexical_features):
        output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        is_sep = input_ids == self.sep_id
        sep_count = is_sep.long().cumsum(dim=1)
        valid = attention_mask.bool() & ~is_sep & (input_ids != self.cls_id)
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)

        tokens = self.align_projection(last_hidden_state)
        normalized = F.normalize(tokens, dim=-1)
        scores = torch.bmm(normalized, normalized.transpose(1, 2)) / self.alignment_temperature
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

        cls_feature = self.cls_norm(self.dropout(last_hidden_state[:, 0, :]))
        match_feature = self.match_norm(pair_feature)
        lexical_feature = self.lexical_encoder(lexical_features.to(cls_feature.dtype))
        fused_feature = torch.cat([cls_feature, match_feature, lexical_feature], dim=-1)
        return self.fusion_output(fused_feature)
