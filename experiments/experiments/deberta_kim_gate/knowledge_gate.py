"""对每个词对、每种关系学习 0~1 权重；不生成词典里没有的新关系。"""
import torch
from torch import nn


class KnowledgeGate(nn.Module):
    def __init__(self, dimension=128, mode='learned'):
        super().__init__()
        if mode not in ('learned', 'fixed', 'open'):
            raise ValueError('gate mode 必须是 learned / fixed / open。')
        self.mode = mode
        # 分别投影再相加，避免构造 [batch, length, length, 260] 大张量。
        self.source = nn.Linear(dimension, 4, bias=False)
        self.target = nn.Linear(dimension, 4, bias=False)
        self.relation = nn.Linear(4, 4, bias=True)
        for parameter in self.parameters():
            nn.init.zeros_(parameter)
        # sigmoid(0)=0.5，初始不因新增层的随机权重偏向某些词或关系。
        self.collect_stats = False
        self.register_buffer('gate_sum', torch.zeros(4), persistent=False)
        self.register_buffer('gate_count', torch.zeros(4), persistent=False)

    def reset_stats(self):
        self.gate_sum.zero_()
        self.gate_count.zero_()

    def forward(self, context, relations):
        if self.mode == 'learned':
            source = self.source(context).unsqueeze(2)  # [B, L, 1, 4]
            target = self.target(context).unsqueeze(1)  # [B, 1, L, 4]
            gates = torch.sigmoid(source + target + self.relation(relations))
        else:
            value = 0.5 if self.mode == 'fixed' else 1.0
            gates = torch.full_like(relations, value)
        if self.collect_stats:
            with torch.no_grad():
                present = relations > 0
                self.gate_sum.add_((gates.detach() * present).sum(dim=(0, 1, 2)))
                self.gate_count.add_(present.sum(dim=(0, 1, 2)))
        return relations * gates
