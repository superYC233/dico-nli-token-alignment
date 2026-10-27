"""一个编码器、一套四类分数：固定F/B/E，只重新学习N。"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from experiments.deberta_lexical.model import DebertaDiCoModel


class FrozenNHeadModel(nn.Module):
    def __init__(self, relation_model, hidden_size=64, n_prior=0.16):
        super().__init__()
        self.relation_model = relation_model
        self.relation_model.requires_grad_(False)
        self.relation_model.eval()

        # CLS包含整体信息；alignment的512维池化特征包含两段文本的比较信息。
        feature_size = relation_model.deberta_model.config.hidden_size + 512
        self.n_head = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # 初始N概率只使用Train的N比例，不从Dev决定偏置。
        nn.init.zeros_(self.n_head[-1].weight)
        nn.init.constant_(self.n_head[-1].bias, math.log(n_prior / (1 - n_prior)))

    def train(self, mode=True):
        super().train(mode)
        # 即使外部调用model.train()，原模型的dropout也始终关闭。
        self.relation_model.eval()
        return self

    @torch.no_grad()
    def extract_features(self, input_ids, attention_mask):
        """沿用原版计算，返回原四类分数及新头需要的特征。"""
        base = self.relation_model
        hidden = base.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        cls_feature = hidden[:, 0, :]
        cls_logits = base.output(base.dropout(cls_feature))

        # 以下Token Alignment公式与原模型相同，没有改对齐方式。
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

    def logits_from_features(self, base_logits, features):
        n_logit = self.n_head(features).squeeze(-1)
        relation_logits = base_logits[:, :3]

        # softmax后的P(N)=sigmoid(n_logit)。其余概率为：
        # P(F/B/E)=(1-P(N))*softmax(原F/B/E分数)。
        # 直接保留原F/B/E数值，三组合加和也保持逐位一致。
        new_n = torch.logsumexp(relation_logits, dim=1) + n_logit
        return torch.cat([relation_logits, new_n.unsqueeze(1)], dim=1)

    def forward(self, input_ids, attention_mask, lexical_features=None):
        base_logits, features = self.extract_features(input_ids, attention_mask)
        return self.logits_from_features(base_logits, features)


def build_model(backbone_config, tokenizer, hidden_size=64, n_prior=0.16):
    # 这里只创建结构；训练脚本随后加载0.9381的完整权重。
    # 独立评估时则加载本次保存的完整best_model，不依赖旧权重文件。
    # 原0.9381使用float32；下载配置中的默认dtype可能是float16。
    backbone = AutoModel.from_config(backbone_config, dtype=torch.float32)
    relation_model = DebertaDiCoModel(
        backbone, tokenizer, feature_size=0, use_lexical_features=False,
    )
    return FrozenNHeadModel(relation_model, hidden_size, n_prior)
