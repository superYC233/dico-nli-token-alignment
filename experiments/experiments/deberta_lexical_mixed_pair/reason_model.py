"""原Alignment+Lexical模型，只增加训练使用的token二分类头。"""
import torch
from torch import nn
import torch.nn.functional as F

from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel


class ReasonDebertaModel(DebertaDiCoModel):
    def __init__(self, backbone, tokenizer, feature_size, lexical_hidden=32, use_lexical_features=True):
        super().__init__(backbone, tokenizer, feature_size, lexical_hidden, use_lexical_features)
        # 初始化新增层不改变原有模型参数与后续随机序列。
        with torch.random.fork_rng(devices=[]):
            self.evidence_head = nn.Linear(backbone.config.hidden_size, 2)
        self._last_hidden = None
        self.deberta_model.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self, module, inputs, output):
        self._last_hidden = output.last_hidden_state

    def forward(self, input_ids, attention_mask, lexical_features, return_evidence=False):
        logits = super().forward(input_ids, attention_mask, lexical_features)
        hidden = self._last_hidden
        self._last_hidden = None
        if return_evidence:
            return logits, self.evidence_head(hidden)
        return logits


def evidence_loss(logits, targets):
    valid = targets.ne(-100)
    annotated = valid.any(dim=1)
    if not annotated.any():
        return logits.sum() * 0.0
    token_loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction='none')
    # 每个已标注样本等权；不让长短语或本批未标注行数量改变监督强度。
    row_loss = token_loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return row_loss[annotated].mean()
