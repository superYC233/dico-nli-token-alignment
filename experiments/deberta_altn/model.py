"""保留 0.930569 Lexical 模型，只为 8 维差异位置增加零初始化投影。"""
import torch
from torch import nn
import torch.nn.functional as F


class DebertaDiCoModel(nn.Module):
    def __init__(self, backbone, tokenizer, feature_size=28, lexical_hidden=32, use_position_features=True):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = self.deberta_model.config.hidden_size
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(hidden_size, 4)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.use_position_features = use_position_features
        self.feature_size = feature_size
        if feature_size != 28:
            raise ValueError('本实验特征应为原 20 维 + 新增 8 维。')

        # 以下初始化、计算公式与原版 Token Alignment 一致。
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

        # 原 20 维分支按原来的形状、顺序初始化，避免扩维改变它的初始权重。
        with torch.random.fork_rng(devices=[]):
            self.lexical_output = nn.Sequential(
                nn.Linear(20, lexical_hidden),
                nn.ReLU(),
                nn.Linear(lexical_hidden, 4),
            )
            nn.init.zeros_(self.lexical_output[-1].weight)
            nn.init.zeros_(self.lexical_output[-1].bias)

        # 仅增加 8 -> 32、无偏置的投影，共 256 个参数，不增加 dropout。
        if use_position_features:
            with torch.random.fork_rng(devices=[]):
                self.position_projection = nn.Linear(8, lexical_hidden, bias=False)
                nn.init.zeros_(self.position_projection.weight)

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
        deberta_output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = deberta_output.last_hidden_state
        cls_logits = self.output(self.dropout(last_hidden_state[:, 0, :]))

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
        logits = cls_logits + match_logits

        if lexical_features.shape != (input_ids.shape[0], self.feature_size):
            raise ValueError('lexical_features 必须是 [batch, 28] 的特征。')
        lexical_features = lexical_features.to(logits.dtype)
        hidden = self.lexical_output[0](lexical_features[:, :20])
        if self.use_position_features:
            hidden = hidden + self.position_projection(lexical_features[:, 20:])
        hidden = self.lexical_output[1](hidden)
        lexical_logits = self.lexical_output[2](hidden)
        return logits + lexical_logits
