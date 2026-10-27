"""原0.938的CLS/Token Alignment特征，加双向联合头；全量微调。"""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_bidirectional.model import BidirectionalDeberta


class AlignmentJointDeberta(BidirectionalDeberta):
    def __init__(self, backbone, tokenizer, dimension=128, head_dropout=0.2):
        # 先按原模型顺序初始化，保留原Token Alignment模块的初值。
        original = DebertaDiCoModel(
            backbone, tokenizer, feature_size=0, use_lexical_features=False,
        )
        super().__init__(backbone, dimension, head_dropout)
        self.cls_id = original.cls_id
        self.sep_id = original.sep_id
        self.cls_dropout = original.dropout
        self.align_projection = original.align_projection
        self.compare_layer = original.compare_layer
        self.alignment_norm = original.match_output[0]
        self.alignment_dropout = original.match_output[1]

        # 原来的两个四分类输出层不用；保留它们之前的特征处理。
        # 每个方向：[CLS的768维, Alignment的512维]，再投影到联合头。
        feature_size = backbone.config.hidden_size + 4 * self.align_projection.out_features
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_size), nn.Linear(feature_size, dimension), nn.GELU(),
        )

    # 原算法原样复用：soft alignment、mean/max pooling均不改变。
    _align = DebertaDiCoModel._align
    _pool = DebertaDiCoModel._pool

    def extract_features(self, hidden, input_ids, attention_mask):
        cls_feature = self.cls_dropout(hidden[:, 0, :])
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
        alignment_feature = self.alignment_dropout(self.alignment_norm(pair_feature))
        return torch.cat([cls_feature, alignment_feature], dim=-1)

    def forward(self, input_ids, attention_mask, row_units=None, row_swapped=None):
        hidden = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        features = self.projection(self.extract_features(hidden, input_ids, attention_mask))
        canonical_logits = self.joint_logits(features[0::2], features[1::2])
        if row_units is None:
            return canonical_logits
        logits = canonical_logits[row_units]
        return torch.where(
            row_swapped.unsqueeze(-1), logits.index_select(1, self.reverse_labels), logits,
        )


def build_model(model_name, revision, tokenizer, dimension=128, head_dropout=0.2):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    # 没有LoRA，也不冻结embedding或encoder层。
    return AlignmentJointDeberta(backbone, tokenizer, dimension, head_dropout)


def parameter_report(model):
    records = []
    for name, parameter in model.named_parameters():
        if name.startswith('deberta_model.'):
            group = 'encoder'
        elif name.startswith(('align_projection.', 'compare_layer.', 'alignment_norm.')):
            group = 'original_alignment'
        else:
            group = 'joint_head'
        records.append({'name': name, 'group': group, 'parameters': parameter.numel(),
                        'trainable': parameter.requires_grad})
    return records


def save_weights(model, path):
    # 全量微调必须保存完整encoder，不能使用LoRA版只存适配参数的约定。
    path = Path(path)
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    temporary = path.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)


def load_weights(model, path):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(
        config['model_name'], config['base_revision'], tokenizer,
        config['dimension'], config['head_dropout'],
    )
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config
