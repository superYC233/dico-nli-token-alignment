"""在原Lexical模型的compare和pool之间增加共享CNN，不重复定义原forward。"""
import torch
from torch import nn
import torch.nn.functional as F

from experiments.deberta_lexical.model import DebertaDiCoModel as LexicalModel


class MaskedComposition(nn.Module):
    def __init__(self, dimension=128, layers=2, kernel_size=3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('卷积核大小需要是奇数，保持token位置和mask对齐。')
        self.layers = nn.ModuleList([
            nn.Conv1d(dimension, dimension, kernel_size, padding=kernel_size // 2)
            for _ in range(layers)
        ])

    def forward(self, features, mask):
        # 每次只保留当前文本。对方文本、CLS/SEP及padding不进入本分支卷积。
        valid = mask.unsqueeze(1).to(features.dtype)
        values = features.transpose(1, 2) * valid
        for layer in self.layers:
            values = F.relu(layer(values)) * valid
        return values.transpose(1, 2)


class DebertaDiCoModel(LexicalModel):
    def __init__(
        self, backbone, tokenizer, feature_size, lexical_hidden=32,
        use_composition=True, composition_layers=2, composition_kernel_size=3,
    ):
        super().__init__(
            backbone, tokenizer, feature_size=feature_size,
            lexical_hidden=lexical_hidden, use_lexical_features=True,
        )
        self.use_composition = use_composition
        if use_composition:
            # 新层初始化不推进原CLS/Alignment/Lexical及训练的随机序列。
            with torch.random.fork_rng(devices=[]):
                self.composition = MaskedComposition(
                    dimension=128, layers=composition_layers,
                    kernel_size=composition_kernel_size,
                )

    def _pool(self, features, mask):
        if self.use_composition:
            # 原forward对A/B分别调用_pool，所以两段文本共用同一套CNN参数。
            features = self.composition(features, mask)
        return super()._pool(features, mask)
