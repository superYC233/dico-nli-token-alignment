"""检查分组、联合梯度和小模型训练，不下载或加载真实权重。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_joint_pair import config as cfg
from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_joint_pair.train_deberta_joint_pair import compute_joint_loss, train_epoch
from experiments.deberta_lexical.model import DebertaDiCoModel


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, first, second, max_length, **kwargs):
        rows = []
        for a, b in zip(first, second):
            token_a = 3 + sum(map(ord, a)) % 20
            token_b = 3 + sum(map(ord, b)) % 20
            rows.append([1, token_a, 2, token_b, 2] + [0] * (max_length - 5))
        ids = torch.tensor(rows)
        return {'input_ids': ids, 'attention_mask': (ids != 0).long()}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # 两个 pair 的原始行故意不相邻；N 在第一行。
    frame = pd.DataFrame([
        ['n0', 'n', 'red cat', 'blue dog', 'NEGATIVE_OTHER'],
        ['f0', 'f', 'black dog', 'dog', 'FORWARD_ENTAILMENT'],
        ['e0', 'e', 'a dog', 'the dog', 'EQUIVALENCE'],
        ['f1', 'f', 'dog', 'black dog', 'BACKWARD_ENTAILMENT'],
        ['e1', 'e', 'the dog', 'a dog', 'EQUIVALENCE'],
    ], columns=['instance_id', 'pair_id', 'text1', 'text2', 'label'])
    dataset = GroupDataset(frame, TinyTokenizer(), max_length=8)
    batch = collate_groups([dataset[i] for i in range(len(dataset))])
    assert batch['source_rows'].tolist() == [0, 1, 3, 2, 4]
    assert batch['pair_indices'].tolist() == [[1, 2], [3, 4]]
    assert batch['pair_targets'].tolist() == [0, 2]
    assert batch['labels'].tolist() == [3, 0, 1, 2, 2]

    # 一轮混合组采样，所有行恰好出现一次，不丢失、不重复 N。
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(7),
    )
    visited = [index for item in loader for index in item['source_rows'].tolist()]
    assert sorted(visited) == list(range(len(frame)))

    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    # 两行都高分预测 N：Pair CE 对 N logit 没有直接梯度，联合目标则有。
    logits = torch.tensor([[3., 0., 0., 10.], [0., 3., 0., 10.]], requires_grad=True)
    total, row_loss, pair_loss = compute_joint_loss(
        logits, torch.tensor([0, 1]), torch.tensor([[0, 1]]), torch.tensor([0]), row_ce, pair_ce,
    )
    pair_gradient = torch.autograd.grad(pair_loss, logits, retain_graph=True)[0]
    gradient = torch.autograd.grad(total, logits)[0]
    assert torch.equal(pair_gradient[:, 3], torch.zeros(2))
    assert (gradient[:, 3] > 0).all()
    assert torch.allclose(total, row_loss + cfg.PAIR_WEIGHT * pair_loss)

    # 全部单例的 batch 应正常退化为 Row CE。
    logits_n = torch.tensor([[0., 0., 0., 1.]], requires_grad=True)
    total_n, row_n, pair_n = compute_joint_loss(
        logits_n, torch.tensor([3]), torch.empty((0, 2), dtype=torch.long),
        torch.empty(0, dtype=torch.long), row_ce, pair_ce,
    )
    assert pair_n.item() == 0 and torch.equal(total_n, row_n)
    total_n.backward()

    # 真正跑一个随机微型 DeBERTa，验证联合损失能更新主干和分类头。
    backbone = DebertaV2Model(DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        relative_attention=False,
    ))
    model = DebertaDiCoModel(backbone, TinyTokenizer(), feature_size=0, use_lexical_features=False)
    assert not any(name.startswith('lexical_output') for name, _ in model.named_parameters())
    old_weight = model.output.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = train_epoch(model, loader, optimizer, row_ce, pair_ce, torch.device('cpu'))
    assert result['train_rows'] == 5 and result['train_pairs'] == 2
    assert result['optimizer_steps'] == 2
    assert not torch.equal(old_weight, model.output.weight.detach())
    assert model.deberta_model.embeddings.word_embeddings.weight.grad is not None
    print('PASS: group alignment, every row once, N gradients, singleton batch, tiny-model training')


if __name__ == '__main__':
    main()
