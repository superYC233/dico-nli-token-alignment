"""CPU小模型检查：不加载真实权重，不进行真实DeBERTa训练。"""
import sys
import io
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import get_pair_logits, pair_aware
from experiments.deberta_frozen_n_head.model import FrozenNHeadModel
from experiments.deberta_frozen_n_head.train_deberta_n_head import selection_record, train_head, predict_cache


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(20, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def main():
    torch.manual_seed(42)
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    base = DebertaDiCoModel(TinyBackbone(), tokenizer, 0, use_lexical_features=False)
    model = FrozenNHeadModel(base, hidden_size=4, n_prior=0.2)
    ids = torch.tensor([[1, 3, 2, 4, 2], [1, 4, 2, 3, 2], [1, 5, 2, 6, 2]])
    mask = torch.ones_like(ids)
    model.train()
    assert not model.relation_model.training and not model.relation_model.deberta_model.dropout.training
    old_logits, features = model.extract_features(ids, mask)
    torch.testing.assert_close(old_logits, base(ids, mask, None), rtol=0, atol=0)
    assert not features.requires_grad
    logits = model.logits_from_features(old_logits, features)
    torch.testing.assert_close(logits[:, :3], old_logits[:, :3], rtol=0, atol=0)
    probabilities = logits.softmax(dim=1)
    n_score = model.n_head(features).squeeze(-1)
    torch.testing.assert_close(probabilities[:, 3], n_score.sigmoid())
    torch.testing.assert_close(probabilities[:, :3], (1 - n_score.sigmoid()).unsqueeze(1) * old_logits[:, :3].softmax(1))
    torch.testing.assert_close(get_pair_logits(logits[:1], logits[1:2]),
                               get_pair_logits(old_logits[:1], old_logits[1:2]), rtol=0, atol=0)

    # 验证：新头的二元CE与四类NLL只差固定的F/B/E条件分类项。
    labels = torch.tensor([0, 1, 3])
    binary = F.binary_cross_entropy_with_logits(n_score, labels.eq(3).float(), reduction='none')
    conditional = torch.zeros(3)
    conditional[:2] = F.cross_entropy(old_logits[:2, :3], labels[:2], reduction='none')
    full = F.cross_entropy(logits, labels, reduction='none')
    torch.testing.assert_close(full, binary + conditional)

    frozen_before = {name: value.clone() for name, value in base.state_dict().items()}
    head_before = {name: value.clone() for name, value in model.n_head.state_dict().items()}
    loader = [(features, labels.eq(3).float())]
    optimizer = torch.optim.AdamW(model.n_head.parameters(), lr=0.01)
    train_head(model, loader, optimizer, nn.BCEWithLogitsLoss(), torch.device('cpu'))
    assert all(torch.equal(frozen_before[name], value) for name, value in base.state_dict().items())
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(not torch.equal(head_before[name], value) for name, value in model.n_head.state_dict().items())

    model.eval()
    frame = pd.DataFrame({'pair_id': ['p', 'p', 'single']})
    after = model(ids, mask)
    cached = predict_cache(model, {'base_logits': old_logits, 'features': features}, torch.device('cpu'))
    torch.testing.assert_close(cached, after, rtol=0, atol=0)
    assert torch.equal(pair_aware(after, frame)[:2], pair_aware(old_logits, frame)[:2])
    # 完整checkpoint可独立恢复，两个指标使用的是同一个模型输出。
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored_base = DebertaDiCoModel(TinyBackbone(), tokenizer, 0, use_lexical_features=False)
    restored = FrozenNHeadModel(restored_base, hidden_size=4, n_prior=0.2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(ids, mask), after, rtol=0, atol=0)

    for independent, paired, expected in [(0.87, 0.94, True), (0.90, 0.93, False), (0.86, 0.96, False)]:
        summary = pd.DataFrame({'method': ['independent', 'pair_aware'], 'weighted_f1': [independent, paired]})
        record = selection_record(summary, 3, 0.87, 0.94)
        assert record['targets_met'] == expected
        assert (record['joint_score'] >= 1.0) == expected
    print('通过：原前向一致、概率公式、冻结参数、梯度、Pair不变、完整保存恢复、同轮选模。')


if __name__ == '__main__':
    main()
