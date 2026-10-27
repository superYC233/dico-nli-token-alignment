"""CPU小模型检查：只检查机制，不训练真实DeBERTa。"""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_frozen_n_head.model import FrozenNHeadModel
from experiments.deberta_frozen_n_head.model_top2 import Top2NHeadModel
from experiments.deberta_frozen_n_head.train_deberta_top2 import compute_losses, make_optimizer, train_epoch


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Dropout(0.1)) for _ in range(4)
        ])

    def forward(self, input_ids, attention_mask):
        values = self.embedding(input_ids)
        for layer in self.encoder.layer:
            values = layer(values)
        return SimpleNamespace(last_hidden_state=values)


def create_model():
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    base = DebertaDiCoModel(TinyBackbone(), tokenizer, 0, use_lexical_features=False)
    return Top2NHeadModel(base, hidden_size=4, n_prior=0.2, unfreeze_last_layers=2)


def main():
    torch.manual_seed(42)
    model = create_model()
    # 模拟已经训练过的N头，避免零初始化最后一层阻断首步特征梯度。
    nn.init.normal_(model.n_head[-1].weight, std=0.1)
    ids = torch.tensor([[1, 3, 2, 4, 2], [1, 4, 2, 3, 2], [1, 5, 2, 6, 2]])
    mask = torch.ones_like(ids)
    model.eval()
    with torch.no_grad():
        logits = model(ids, mask)
        old_base, old_features = FrozenNHeadModel.extract_features(model, ids, mask)
        expected = model.logits_from_features(old_base, old_features)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    model.train()
    layers = model.relation_model.deberta_model.encoder.layer
    assert not layers[0].training and not layers[1].training
    assert layers[2].training and layers[3].training
    assert not model.relation_model.dropout.training and not model.relation_model.match_output.training
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    # 单独使用N金标也必须能给顶部两层梯度；这是本版本最关键的检查。
    n_labels = torch.full((3,), 3, dtype=torch.long)
    teacher = old_base.detach()
    loss, row, keep, kept = compute_losses(model(ids, mask), n_labels, teacher, 0.5, 2.0)
    assert kept == 0 and keep.item() == 0.0
    loss.backward()
    for layer in layers[-2:]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    assert all(p.grad is None for p in model.relation_model.output.parameters())
    assert all(p.grad is None for p in model.relation_model.compare_layer.parameters())
    assert all(p.grad is None for p in model.relation_model.deberta_model.embedding.parameters())

    optimizer = make_optimizer(model, 0.001, 0.01, 0.01)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert optimizer_ids == {id(p) for p in model.parameters() if p.requires_grad}
    # 完整训练函数只在这个小模型上运行一次，检查真实更新范围。
    train_epoch(model, [(ids, mask, n_labels, teacher)], optimizer, torch.device('cpu'), 0.5, 2.0, 1.0)
    changed = []
    for name, parameter in model.named_parameters():
        different = not torch.equal(before[name], parameter.detach())
        if not parameter.requires_grad:
            assert not different and parameter.grad is None
        if different:
            changed.append(name)
    assert any('.encoder.layer.2.' in name for name in changed)
    assert any('.encoder.layer.3.' in name for name in changed)
    assert any(name.startswith('n_head.') for name in changed)

    # 教师N分数无论多大都不影响条件保持；教师错误与N样本不参与KL。
    student = torch.tensor([[2., 0., 0., 3.], [0., 2., 0., 4.], [0., 0., 2., 5.]], requires_grad=True)
    teacher = torch.tensor([[3., 0., 0., -100.], [0., 0., 3., -100.], [3., 0., 0., -100.]], requires_grad=True)
    labels = torch.tensor([0, 1, 3])
    _, _, keep1, kept = compute_losses(student, labels, teacher, 0.5, 2.0)
    assert kept == 1
    other_teacher = teacher.detach().clone()
    other_teacher[:, 3] = 10000.0
    _, _, keep2, _ = compute_losses(student, labels, other_teacher, 0.5, 2.0)
    torch.testing.assert_close(keep1, keep2, rtol=0, atol=0)
    keep1.backward()
    assert teacher.grad is None
    assert student.grad[:, 3].abs().sum() == 0
    assert student.grad[1:].abs().sum() == 0
    assert student.grad[0, :3].abs().sum() > 0

    # KL权重为0时，损失退化为普通四分类CE。
    loss, row, _, _ = compute_losses(student, labels, teacher, 0.0, 2.0)
    torch.testing.assert_close(loss, row)
    model.eval()
    with torch.no_grad():
        expected = model(ids, mask)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = create_model()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, mask), expected, rtol=0, atol=0)
    print('通过：起点前向一致、N梯度到顶层、冻结参数不变、优化器范围、KL不读N、完整保存恢复。')


if __name__ == '__main__':
    main()
