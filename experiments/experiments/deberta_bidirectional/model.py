"""两方向先做可训练交互，再进行一次四分类；没有Pair CE。"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional.data import REVERSE


class BidirectionalDeberta(nn.Module):
    def __init__(self, backbone, dimension=128, head_dropout=0.2):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = backbone.config.hidden_size
        self.dimension = dimension

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dimension), nn.GELU(),
        )
        self.joint_layer = nn.Sequential(
            nn.LayerNorm(3 * dimension), nn.Linear(3 * dimension, dimension),
            nn.GELU(), nn.Dropout(head_dropout),
        )
        # 三个对称分数：单向蕴含的共同分数、E分数、N分数。
        self.symmetric_output = nn.Linear(dimension, 3)
        # 根据共同信息，学习该怎样读取正反表征的差异。
        self.direction_weight = nn.Linear(dimension, dimension)
        self.register_buffer('reverse_labels', torch.tensor(REVERSE), persistent=False)

    def joint_logits(self, first, second):
        symmetric = torch.cat([
            first + second, torch.abs(first - second), first * second,
        ], dim=-1)
        joint = self.joint_layer(symmetric)
        common, equivalence, negative = self.symmetric_output(joint).unbind(dim=-1)
        direction = (self.direction_weight(joint) * (first - second)).sum(dim=-1)
        direction = direction / math.sqrt(self.dimension)

        # 交换first/second：共同特征不变，direction变号，F/B自然交换。
        # N与其他三类始终一起参与softmax，不依赖“这行是不是单例”。
        return torch.stack([
            common + direction, common - direction, equivalence, negative,
        ], dim=-1)

    def forward(self, input_ids, attention_mask, row_units=None, row_swapped=None):
        # 本批依次为AB、BA、CD、DC……；同一个编码器一次合批前向。
        hidden = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        features = self.projection(hidden)
        canonical_logits = self.joint_logits(features[0::2], features[1::2])
        if row_units is None:
            return canonical_logits

        # 计算复用后，还原到每条原始行。这里不读取标签或pair_id。
        logits = canonical_logits[row_units]
        reversed_logits = logits.index_select(1, self.reverse_labels)
        return torch.where(row_swapped.unsqueeze(-1), reversed_logits, logits)


def build_model(model_name, revision, lora_settings, dimension=128, head_dropout=0.2):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), revision=revision,
        local_files_only=True, dtype=torch.float32, use_safetensors=True,
    )
    model = BidirectionalDeberta(backbone, dimension, head_dropout)
    with torch.random.fork_rng(devices=[]):
        model.deberta_model = inject_adapter_in_model(LoraConfig(**lora_settings), backbone)
    return model


def parameter_report(model):
    records = []
    for name, parameter in model.named_parameters():
        if 'lora_A.' in name or 'lora_B.' in name:
            group = 'lora'
        elif name.startswith('deberta_model.'):
            group = 'frozen_encoder'
        else:
            group = 'joint_head'
        records.append({'name': name, 'group': group, 'parameters': parameter.numel(),
                        'trainable': parameter.requires_grad})
    return records


def save_weights(model, path):
    # 只保存LoRA与联合头，冻结底座从相同revision恢复，避免每轮保存700MB。
    path = Path(path)
    weights = {name: parameter.detach().cpu()
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    temporary = path.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)


def load_weights(model, path):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(weights) != expected:
        raise ValueError('权重与当前LoRA/联合头不一致，请使用本目录的加载函数。')
    model.load_state_dict(weights, strict=False)


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(
        config['model_name'], config['base_revision'], config['lora'],
        config['dimension'], config['head_dropout'],
    )
    load_weights(model, model_dir / 'best_model.pth')
    return model.to(device).eval(), tokenizer, config
