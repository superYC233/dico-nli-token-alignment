"""保留原CLS+Token Alignment输出；局部头只在训练/诊断时使用。"""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel as OriginalModel
from experiments.deberta_local_semantics import config as cfg


class DebertaLocalModel(OriginalModel):
    def __init__(self, backbone, tokenizer, local_hidden=128):
        super().__init__(backbone, tokenizer, feature_size=0, use_lexical_features=False)
        # 新层初始化不改变原模型的随机序列；局部头不加额外dropout。
        with torch.random.fork_rng(devices=[]):
            self.local_relation_head = nn.Sequential(
                nn.LayerNorm(4 * 128),
                nn.Linear(4 * 128, local_hidden), nn.ReLU(),
                nn.Linear(local_hidden, 4),
            )

    def forward(self, input_ids, attention_mask, lexical_features=None, return_tokens=False):
        hidden = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        cls_logits = self.output(self.dropout(hidden[:, 0, :]))
        is_sep = input_ids == self.sep_id
        sep_count = is_sep.long().cumsum(dim=1)
        valid = attention_mask.bool() & ~is_sep & (input_ids != self.cls_id)
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)
        tokens = self.align_projection(hidden)
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
        logits = cls_logits + self.match_output(pair_feature)
        return (logits, tokens) if return_tokens else logits

    def local_logits(self, tokens, span1_mask, span2_mask):
        def mean_span(mask):
            weights = mask.unsqueeze(-1).to(tokens.dtype)
            return (tokens * weights).sum(dim=1) / weights.sum(dim=1)

        first = mean_span(span1_mask)
        second = mean_span(span2_mask)
        comparison = torch.cat([first, second, first - second, first * second], dim=-1)
        return self.local_relation_head(comparison)


def build_model(tokenizer, model_name=cfg.MODEL_NAME, local_hidden=cfg.LOCAL_HIDDEN_SIZE, revision=None):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), local_files_only=True,
        revision=revision, dtype=torch.float32, use_safetensors=True,
    )
    return DebertaLocalModel(backbone, tokenizer, local_hidden)


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    settings = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(
        tokenizer, settings['model_name'], settings['local_hidden_size'], settings['base_revision'],
    )
    model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location='cpu', weights_only=True))
    return model.to(device).eval(), tokenizer, settings
