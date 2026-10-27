"""Row CE架构对照：CLS，或者CLS加原Token Alignment。"""
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_manual_aug.relation_lora_model import add_lora, load_trainable_weights


class DebertaRowModel(DebertaDiCoModel):
    def __init__(self, backbone, tokenizer, head_type):
        # 先按原顺序初始化，保证两组共享的编码器和CLS参数相同。
        # 原Alignment初始化在fork_rng里，不改变后续LoRA初始化的随机数。
        super().__init__(backbone, tokenizer, feature_size=0, use_lexical_features=False)
        self.head_type = head_type
        if head_type == 'cls':
            # 真正删除模块，既不计算，也不放进优化器或保存权重。
            del self.align_projection, self.compare_layer, self.match_output
        elif head_type != 'alignment':
            raise ValueError('head_type只能为cls或alignment。')

    def forward(self, input_ids, attention_mask, lexical_features=None):
        if self.head_type == 'alignment':
            return super().forward(input_ids, attention_mask, lexical_features)
        hidden = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.output(self.dropout(hidden[:, 0, :]))


def build_model(model_name, tokenizer, head_type, lora_settings, revision=None):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), local_files_only=True,
        revision=revision, dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaRowModel(backbone, tokenizer, head_type)
    return add_lora(model, lora_settings)


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(
        config['model_name'], tokenizer, config['head_type'], config['lora'],
        revision=config['base_revision'],
    )
    load_trainable_weights(model, model_dir / 'best_model.pth')
    model.to(device).eval()
    return model, tokenizer, config
