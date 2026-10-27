"""原Alignment与Lexical不变，只让CLS分支经过NLI pooler。"""
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler
from transformers.utils.hub import cached_file

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel


class DebertaPoolerModel(DebertaDiCoModel):
    def __init__(self, backbone, tokenizer, feature_size, lexical_hidden=32,
                 use_lexical_features=True, use_nli_pooler=True):
        super().__init__(
            backbone, tokenizer, feature_size, lexical_hidden, use_lexical_features,
        )
        self.use_nli_pooler = use_nli_pooler
        # 新模块的初始化不改变原四分类、Alignment、Lexical及后续训练的随机序列。
        with torch.random.fork_rng(devices=[]):
            self.nli_pooler = ContextPooler(backbone.config) if use_nli_pooler else None

    def forward(self, input_ids, attention_mask, lexical_features):
        hidden = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state

        cls_feature = self.nli_pooler(hidden) if self.use_nli_pooler else hidden[:, 0, :]
        cls_logits = self.output(self.dropout(cls_feature))

        # 以下与旧版完全相同，pooler不会替换Alignment所用的token表示。
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
        if self.use_lexical_features:
            logits = logits + self.lexical_output(lexical_features.to(logits.dtype))
        return logits


def read_nli_pooler(model_name, revision):
    # 只读取本地已有checkpoint中的两块小权重，不加载旧三分类头。
    path = cached_file(
        model_name, 'model.safetensors', revision=revision,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    with safe_open(path, framework='pt', device='cpu') as checkpoint:
        state = {
            'dense.weight': checkpoint.get_tensor('pooler.dense.weight'),
            'dense.bias': checkpoint.get_tensor('pooler.dense.bias'),
        }
    return state


def load_pretrained_model(tokenizer, model_name, revision, use_nli_pooler=True,
                          lexical_hidden=32, use_lexical_features=True, feature_size=20):
    backbone = AutoModel.from_pretrained(
        model_name, revision=revision, cache_dir=get_model_cache_dir(),
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaPoolerModel(
        backbone, tokenizer, feature_size, lexical_hidden,
        use_lexical_features, use_nli_pooler,
    )
    if use_nli_pooler:
        model.nli_pooler.load_state_dict(read_nli_pooler(model_name, revision), strict=True)
        print('NLI pooler已从同一固定revision单独载入；原三分类输出层不使用。')
    return model
