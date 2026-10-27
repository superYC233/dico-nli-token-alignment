"""沿用原N头结构，增加一条可反传的特征计算路径。"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_frozen_n_head.model import FrozenNHeadModel


class Top2NHeadModel(FrozenNHeadModel):
    def __init__(self, relation_model, hidden_size=64, n_prior=0.16, unfreeze_last_layers=2):
        super().__init__(relation_model, hidden_size, n_prior)
        self.unfreeze_last_layers = unfreeze_last_layers
        layers = self.relation_model.deberta_model.encoder.layer
        assert 0 < unfreeze_last_layers <= len(layers)
        for layer in layers[-unfreeze_last_layers:]:
            layer.requires_grad_(True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.relation_model.eval()
        self.n_head.train(mode)
        # 冻结部分的dropout关闭；只有顶部训练层按训练/评估切换。
        layers = self.relation_model.deberta_model.encoder.layer
        for layer in layers[-self.unfreeze_last_layers:]:
            layer.train(mode)
        return self

    def extract_features(self, input_ids, attention_mask):
        # 这里不能加no_grad，也不能用上轮缓存的学生特征。
        # 公式与旧模型相同，但现在损失必须能反传到顶部编码层。
        base = self.relation_model
        hidden = base.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        cls_feature = hidden[:, 0, :]
        cls_logits = base.output(base.dropout(cls_feature))

        is_sep = input_ids == base.sep_id
        sep_count = is_sep.long().cumsum(dim=1)
        valid = attention_mask.bool() & ~is_sep & (input_ids != base.cls_id)
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)
        tokens = base.align_projection(hidden)
        normalized = F.normalize(tokens, dim=-1)
        scores = torch.bmm(normalized, normalized.transpose(1, 2)) / 0.1
        aligned_b = base._align(tokens, scores, mask_b)
        aligned_a = base._align(tokens, scores, mask_a)
        features_a = base.compare_layer(torch.cat(
            [tokens, aligned_b, tokens - aligned_b, tokens * aligned_b], dim=-1,
        ))
        features_b = base.compare_layer(torch.cat(
            [tokens, aligned_a, tokens - aligned_a, tokens * aligned_a], dim=-1,
        ))
        pair_feature = torch.cat(
            [base._pool(features_a, mask_a), base._pool(features_b, mask_b)], dim=-1,
        )
        base_logits = cls_logits + base.match_output(pair_feature)
        features = torch.cat([cls_feature, pair_feature], dim=-1)
        return base_logits, features


def build_model(backbone_config, tokenizer, hidden_size=64, n_prior=0.16, unfreeze_last_layers=2):
    backbone = AutoModel.from_config(backbone_config, dtype=torch.float32)
    relation_model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    )
    return Top2NHeadModel(relation_model, hidden_size, n_prior, unfreeze_last_layers)
