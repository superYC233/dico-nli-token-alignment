"""保留最优词汇版，只在 Token Alignment 比较层加入四维词对知识。"""
import torch
from torch import nn
import torch.nn.functional as F


class DebertaDiCoModel(nn.Module):
    def __init__(self, backbone, tokenizer, feature_size, lexical_hidden=32, use_lexical_features=True, use_kim_features=True):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = self.deberta_model.config.hidden_size
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(hidden_size, 4)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.use_lexical_features = use_lexical_features
        self.feature_size = feature_size
        self.use_kim_features = use_kim_features

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

        # 新增部分：20 维词汇特征 -> 32 维 -> 四类修正分数。
        # 不加 dropout，避免额外消耗原训练流程的随机数。
        if use_lexical_features:
            with torch.random.fork_rng(devices=[]):
                self.lexical_output = nn.Sequential(
                    nn.Linear(feature_size, lexical_hidden),
                    nn.ReLU(),
                    nn.Linear(lexical_hidden, 4),
                )
                # 开始时修正分数为 0，初始预测与原版一致；之后由 CE 学习。
                nn.init.zeros_(self.lexical_output[-1].weight)
                nn.init.zeros_(self.lexical_output[-1].bias)

        # KIM 式局部知识投影：4 个词对关系 -> 原有比较层的 128 维。
        # 零初始化且隔离 RNG：新增模块不会改变旧参数的初始值。
        if use_kim_features:
            with torch.random.fork_rng(devices=[]):
                self.relation_projection = nn.Linear(4, 128, bias=False)
                nn.init.zeros_(self.relation_projection.weight)

    def _align(self, tokens, scores, target_mask):
        mask = target_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(weights, tokens), weights

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

    def forward(self, input_ids, attention_mask, lexical_features, relations=None):
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
        aligned_b, weights_b = self._align(tokens, scores, mask_b)
        aligned_a, weights_a = self._align(tokens, scores, mask_a)
        first = torch.cat([tokens, aligned_b, tokens - aligned_b, tokens * aligned_b], dim=-1)
        second = torch.cat([tokens, aligned_a, tokens - aligned_a, tokens * aligned_a], dim=-1)

        # 先计算原有线性比较，再加入对齐加权的知识，最后使用原有 ReLU。
        features_a = self.compare_layer[0](first)
        features_b = self.compare_layer[0](second)
        if self.use_kim_features:
            length = input_ids.shape[1]
            if relations is None or relations.shape != (input_ids.shape[0], length, length, 4):
                raise ValueError('relations 必须是 [batch, length, length, 4]。')
            relations = relations.to(tokens.dtype)
            valid_edges = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)
            relations = relations * valid_edges.unsqueeze(-1)
            # sum_j alpha_ij * R_ij：只汇总当前 token 对齐到的词对知识。
            local_a = (weights_b.unsqueeze(-1) * relations).sum(dim=2)
            # 换方向时，car->vehicle 的“更具体”须变成 vehicle->car 的“更一般”。
            reverse = relations.transpose(1, 2)[..., [0, 2, 1, 3]]
            local_b = (weights_a.unsqueeze(-1) * reverse).sum(dim=2)
            features_a = features_a + self.relation_projection(local_a)
            features_b = features_b + self.relation_projection(local_b)
        features_a = self.compare_layer[1](features_a)
        features_b = self.compare_layer[1](features_b)
        pair_feature = torch.cat(
            [self._pool(features_a, mask_a), self._pool(features_b, mask_b)], dim=-1,
        )
        match_logits = self.match_output(pair_feature)
        logits = cls_logits + match_logits

        if self.use_lexical_features:
            if lexical_features.shape != (input_ids.shape[0], self.feature_size):
                raise ValueError('lexical_features 必须是 [batch, 20] 的特征。')
            lexical_logits = self.lexical_output(lexical_features.to(logits.dtype))
            logits = logits + lexical_logits
        return logits
