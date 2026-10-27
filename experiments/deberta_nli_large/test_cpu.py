"""CPU 数据与损失检查：不加载预训练模型，不反向传播，不更新参数。"""
import ast
import io
import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.deberta_nli_large import config as cfg
from experiments.deberta_nli_large.data import read_data, GroupDataset, collate_groups
from experiments.deberta_nli_large.train_deberta_large import compute_joint_loss, make_dev_loader
from experiments.deberta_nli_large.data import build_feature_tensor, DiCoDataset
from experiments.deberta_nli_large.evaluation import predict_logits, pair_aware
from experiments.deberta_nli_large.model import DebertaDiCoModel


class TinyTokenizer:
    def __call__(self, first, second, **kwargs):
        if isinstance(first, str):
            first, second = [first], [second]
        values = []
        for a, b in zip(first, second):
            values.append([1, 3 + sum(map(ord, a)) % 25, 2, 3 + sum(map(ord, b)) % 25, 2, 0])
        return {
            'input_ids': torch.tensor(values),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0]] * len(values)),
        }


def check_full_epoch():
    train, dev = read_data()
    tokenizer = TinyTokenizer()
    features = build_feature_tensor(train)
    dataset = GroupDataset(train, tokenizer, features)
    assert len(train) == 3042 and len(dev) == 660
    assert len(dataset) == 1760 and dataset.pair_count == 1282
    loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    source_rows = []
    counts = torch.zeros(4, dtype=torch.long)
    pair_count = 0
    batches_without_n = 0
    combinations = [(0, 1), (1, 0), (2, 2)]
    for batch in loader:
        source = batch['source_rows']
        source_rows.extend(source.tolist())
        assert len(source) <= cfg.GROUP_BATCH_SIZE * 2
        assert torch.equal(batch['features'], features[source])
        assert torch.equal(batch['input_ids'], dataset.encoded['input_ids'][source])
        assert torch.equal(batch['labels'], dataset.labels[source])
        counts += torch.bincount(batch['labels'], minlength=4)
        batches_without_n += int(not (batch['labels'] == 3).any())
        pair_count += len(batch['pair_targets'])
        for indices, target in zip(batch['pair_indices'], batch['pair_targets']):
            first, second = indices.tolist()
            assert tuple(batch['labels'][indices].tolist()) == combinations[target.item()]
            a, b = train.iloc[source[first].item()], train.iloc[source[second].item()]
            assert a.text1 == b.text2 and a.text2 == b.text1 and a.pair_id == b.pair_id
    assert sorted(source_rows) == list(range(len(train)))
    assert counts.tolist() == [881, 881, 802, 478]
    assert pair_count == 1282 and len(loader) == 220

    # 提前整批编码与旧版逐条编码一致，方向特征也保持原行顺序。
    old_rows = DiCoDataset(train, tokenizer, features)
    for index in [0, 1, 100, 3041]:
        ids, mask, row_features, label = old_rows[index]
        assert torch.equal(ids, dataset.encoded['input_ids'][index])
        assert torch.equal(mask, dataset.encoded['attention_mask'][index])
        assert torch.equal(row_features, features[index]) and label == dataset.labels[index]
    print('完整一轮：3042 行各一次，1282 对各一次，220 步；四类数量：', counts.tolist())
    print('本轮没有 N 的批次数：', batches_without_n, '；没有连续排除 N 的第二阶段。')
    return dev, tokenizer


@torch.inference_mode()
def check_losses():
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float64),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    # 三个真实方向组合，穿插两个 N 单例，不按两行一组的固定位置猜索引。
    labels = torch.tensor([3, 0, 1, 3, 1, 0, 2, 2])
    indices = torch.tensor([[1, 2], [4, 5], [6, 7]])
    targets = torch.tensor([0, 1, 2])
    logits = torch.zeros(8, 4, dtype=torch.float64)
    logits[torch.arange(8), labels] = 3.0
    total, row, pair = compute_joint_loss(
        logits, labels, indices, targets, row_criterion, pair_criterion, 0.5,
    )
    assert torch.allclose(row, row_criterion(logits, labels))
    assert torch.allclose(total, row + 0.5 * pair)
    zero_total, zero_row, _ = compute_joint_loss(
        logits, labels, indices, targets, row_criterion, pair_criterion, 0.0,
    )
    assert torch.equal(zero_total, zero_row)

    # N 单例同样参与监督：提高真实 N 的 N 分数，应降低联合损失。
    better_n = logits.clone()
    better_n[0, 3] += 1.0
    n_total, _, n_pair = compute_joint_loss(
        better_n, labels, indices, targets, row_criterion, pair_criterion, 0.5,
    )
    assert n_total < total and torch.equal(n_pair, pair)
    # 对 F 行错误提高 N 分数，损失应升高。
    wrong_n = logits.clone()
    wrong_n[1, 3] += 10.0
    assert compute_joint_loss(
        wrong_n, labels, indices, targets, row_criterion, pair_criterion, 0.5,
    )[0] > total

    # 全 N 批次是正常输入，不产生虚构的 Pair 目标。
    only_n = compute_joint_loss(
        torch.zeros(2, 4, dtype=torch.float64), torch.tensor([3, 3]),
        torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long),
        row_criterion, pair_criterion, 0.5,
    )
    assert torch.equal(only_n[0], only_n[1]) and only_n[2].item() == 0.0
    assert all(torch.isfinite(value) for value in only_n)
    print('联合损失、零 Pair 权重对照、N 正例监督和全 N 批次均通过。')


class DeterministicModel(nn.Module):
    def forward(self, input_ids, attention_mask, features):
        return features[:, :4]


def check_dev_rng(dev, tokenizer):
    loader = make_dev_loader(dev, tokenizer, build_feature_tensor(dev), 128, 16)
    model = DeterministicModel()
    state = torch.get_rng_state().clone()
    first = predict_logits(model, loader, torch.device('cpu'))
    assert torch.equal(state, torch.get_rng_state())
    assert model.training
    second = predict_logits(model, loader, torch.device('cpu'))
    assert torch.equal(first, second) and first.shape == (660, 4)
    assert torch.equal(state, torch.get_rng_state())
    print('Dev 预测不消耗训练用全局随机数，预测顺序和模型模式恢复通过。')


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1024)
        self.embedding = nn.Embedding(40, 1024)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@torch.inference_mode()
def check_model_and_serialization():
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    model = DebertaDiCoModel(TinyBackbone(), tokenizer, feature_size=20).eval()
    ids = torch.tensor([[1, 3, 4, 2, 5, 2, 0], [1, 5, 2, 3, 4, 2, 0]])
    mask = (ids != 0).long()
    features = torch.zeros(2, 20)
    logits = model(ids, mask, features)
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
    assert model.output.in_features == 1024
    assert model.align_projection.in_features == 1024
    assert all(parameter.requires_grad for parameter in model.parameters())

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DebertaDiCoModel(TinyBackbone(), tokenizer, feature_size=20).eval()
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    assert torch.equal(logits, restored(ids, mask, features))
    print('Model: 1024 hidden size, four-class output, no frozen parameters, save/load match.')


def check_source_consistency():
    experiment = PROJECT_ROOT / 'experiments/deberta_nli_large'
    original_model = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe/model.py'
    old_class = next(node for node in ast.parse(original_model.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.ClassDef))
    new_class = next(node for node in ast.parse((experiment / 'model.py').read_text(encoding='utf-8')).body
                     if isinstance(node, ast.ClassDef))
    assert ast.dump(old_class) == ast.dump(new_class)
    original_features = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe/lexical_features.py'
    assert (experiment / 'lexical_features.py').read_text(encoding='utf-8') == original_features.read_text(encoding='utf-8')
    assert cfg.SELECT_BY == 'independent'
    assert cfg.LEARNING_RATE == 2e-5 and cfg.LEXICAL_LEARNING_RATE == 1e-3
    assert cfg.PAIR_WEIGHT == 0.5 and cfg.BF16 and cfg.GRADIENT_CHECKPOINTING
    print('Source: unchanged model class and lexical features; intended configuration matches.')


@torch.inference_mode()
def check_decoding(dev):
    labels = torch.tensor(dev['label'].map(cfg.LABEL_TO_ID).tolist())
    logits = torch.zeros(len(dev), 4)
    logits[torch.arange(len(dev)), labels] = 4.0
    assert torch.equal(pair_aware(logits, dev), labels)

    singleton = torch.tensor(dev.groupby('pair_id')['pair_id'].transform('size').eq(1).to_numpy())
    logits[:, 3] = 100.0
    paired = pair_aware(logits, dev)
    assert torch.equal(paired, labels)
    assert (logits.argmax(dim=1) == 3).all()
    assert (paired[singleton] == 3).all() and (paired[~singleton] != 3).all()
    print('Decoding: paired rows use FB/BF/EE; singleton decisions are unchanged.')


if __name__ == '__main__':
    check_source_consistency()
    check_model_and_serialization()
    dev, tokenizer = check_full_epoch()
    check_losses()
    check_dev_rng(dev, tokenizer)
    check_decoding(dev)
    print('全部通过；没有加载预训练模型，没有 backward 或 optimizer.step。')
