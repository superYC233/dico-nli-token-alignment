"""参数锚定检查；只记录梯度/验证预设位移，不执行训练优化器更新。"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

from project_paths import get_model_cache_dir
from experiments.deberta_lexical import train_deberta_lexical_anchor as cfg
from experiments.deberta_lexical import train_deberta_lexical as legacy
from experiments.deberta_lexical.parameter_anchor import RowParameterAnchor
from experiments.deberta_lexical.test_cpu import TinyTokenizer


class Recorder:
    def __init__(self, model):
        self.model = model
        self.param_groups = cfg.build_optimizer(model).param_groups
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.gradients.append({name: p.grad.clone() for name, p in self.model.named_parameters()
                               if p.grad is not None})


def toy_model():
    backbone = DebertaV2Model(DebertaV2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=128, pad_token_id=0,
        hidden_dropout_prob=.1, attention_probs_dropout_prob=.1, relative_attention=False,
    ))
    return cfg.DebertaDiCoModel(backbone, TinyTokenizer(), feature_size=20,
                                lexical_hidden=32, use_lexical_features=True)


def compare_gradients(first, second):
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def check_formula():
    model = nn.Module()
    model.original_weight = nn.Parameter(torch.tensor([1., -1.]))
    model.lexical_output = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.lexical_output.weight.copy_(torch.tensor([[2., -2.]]))
    optimizer = cfg.build_optimizer(model)
    anchor = RowParameterAnchor(model)
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        model.original_weight.add_(torch.tensor([.1, .2]))
        model.lexical_output.weight.add_(torch.tensor([[.4, .6]]))
    candidate = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.original_weight.grad = None
    model.lexical_output.weight.grad = torch.ones_like(model.lexical_output.weight)
    for p in model.parameters():
        optimizer.state[p]['exp_avg'] = torch.ones_like(p)
        optimizer.state[p]['exp_avg_sq'] = 2 * torch.ones_like(p)
    stats = anchor.apply(optimizer, cfg.ANCHOR_STRENGTH)
    for group in optimizer.param_groups:
        retention = 1 / (1 + group['lr'] * cfg.ANCHOR_STRENGTH)
        for p in group['params']:
            name, reference = anchor.references[id(p)]
            expected = reference + retention * (candidate[name] - reference)
            torch.testing.assert_close(p, expected, rtol=1e-6, atol=1e-7)
            assert torch.equal(reference, saved[name])
            assert optimizer.state[p]['exp_avg'].eq(1).all()
            assert optimizer.state[p]['exp_avg_sq'].eq(2).all()
    assert model.original_weight.grad is None
    assert model.lexical_output.weight.grad.eq(1).all()
    assert abs(stats['original_retention'] - 1/1.02) < 1e-12
    assert stats['lexical_retention'] == .5
    assert stats['anchor_distance_after_pull'] < stats['anchor_distance_before_pull']
    before_zero = {name: p.detach().clone() for name, p in model.named_parameters()}
    anchor.apply(optimizer, 0)
    assert all(torch.equal(p, before_zero[name]) for name, p in model.named_parameters())
    fresh = RowParameterAnchor(model)
    stats = fresh.apply(optimizer, cfg.ANCHOR_STRENGTH)
    assert stats['anchor_distance_before_pull'] == stats['anchor_distance_after_pull'] == 0
    assert all(torch.equal(p, before_zero[name]) for name, p in model.named_parameters())
    print('公式通过：分组近端回拉、None梯度参数覆盖、零强度无变化、锚点刷新、不修改梯度/Adam动量。')


def check_data(train, dev):
    features = cfg.build_feature_tensor(train)
    assert features.shape == (3042, 20)
    assert train.label.value_counts().to_dict() == {
        'FORWARD_ENTAILMENT': 881, 'BACKWARD_ENTAILMENT': 881,
        'EQUIVALENCE': 802, 'NEGATIVE_OTHER': 478,
    }
    old = pd.read_csv(cfg.REFERENCE_DIR / 'train_features.csv', keep_default_na=False)
    assert old.instance_id.tolist() == train.instance_id.tolist()
    old_features = torch.tensor(old[cfg.FEATURE_NAMES].to_numpy(), dtype=torch.float32)
    torch.testing.assert_close(features, old_features, rtol=0, atol=0)
    torch.testing.assert_close(features, cfg.build_feature_tensor(train[['text1', 'text2']]), rtol=0, atol=0)
    pairs = cfg.PairDataset(train, TinyTokenizer(), features)
    assert len(pairs) == 1282 and len(dev) == 660
    sources = [index for pair in pairs.pairs for index in pair[:2]]
    assert len(sources) == len(set(sources)) == 2564
    assert set(sources) == set(train.index[train.label.ne('NEGATIVE_OTHER')])
    for i in range(5):
        item = pairs[i]
        first, second, _ = pairs.pairs[i]
        assert torch.equal(item[2], features[first])
        assert torch.equal(item[5], features[second])
    print('数据通过：官方3042/660、1282组Pair、478条N；20维特征逐项匹配旧0.9306，不读取标签。')


def check_training():
    frame = pd.DataFrame([
        {'instance_id': 'f', 'pair_id': 'fb', 'text1': 'a black car', 'text2': 'a car', 'label': 'FORWARD_ENTAILMENT'},
        {'instance_id': 'b', 'pair_id': 'fb', 'text1': 'a car', 'text2': 'a black car', 'label': 'BACKWARD_ENTAILMENT'},
        {'instance_id': 'e0', 'pair_id': 'ee', 'text1': 'a small cat', 'text2': 'a little cat', 'label': 'EQUIVALENCE'},
        {'instance_id': 'e1', 'pair_id': 'ee', 'text1': 'a little cat', 'text2': 'a small cat', 'label': 'EQUIVALENCE'},
        {'instance_id': 'n', 'pair_id': 'n', 'text1': 'a dog', 'text2': 'a girl', 'label': 'NEGATIVE_OTHER'},
    ])
    features = cfg.build_feature_tensor(frame)
    rows = list(DataLoader(cfg.DiCoDataset(frame, TinyTokenizer(), features), batch_size=3))
    pairs = list(DataLoader(cfg.PairDataset(frame, TinyTokenizer(), features), batch_size=1))
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    device = torch.device('cpu')
    model = toy_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = cfg.build_optimizer(model)
    original_optimizer = legacy.build_optimizer(model)
    assert len(optimizer.param_groups) == 2
    for new, old in zip(optimizer.param_groups, original_optimizer.param_groups):
        assert new['lr'] == old['lr'] and new['weight_decay'] == old['weight_decay']
        assert [id(p) for p in new['params']] == [id(p) for p in old['params']]
    assert sum(p.numel() for p in model.lexical_output.parameters()) == 804
    assert cfg.train_rows is legacy.train_rows
    model.train()
    cfg.train_rows(model, rows, Recorder(model), row_ce, device)
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())

    # 零强度完整Pair路径与原Lexical逐参数梯度、dropout随机序列一致。
    other = copy.deepcopy(model)
    old_recorder, new_recorder = Recorder(model), Recorder(other)
    torch.manual_seed(90)
    old_loss = legacy.train_pairs(model, pairs, old_recorder, pair_ce, device)
    old_rng = torch.get_rng_state()
    anchor = RowParameterAnchor(other)
    torch.manual_seed(90)
    result, records = cfg.train_anchored_pairs(other, pairs, new_recorder, pair_ce, device, anchor, strength=0)
    assert result['pair_loss'] == old_loss and result['pair_ce_groups'] == 2
    assert torch.equal(old_rng, torch.get_rng_state())
    for first, second in zip(old_recorder.gradients, new_recorder.gradients):
        compare_gradients(first, second)
    assert all(torch.equal(p, before[name]) for name, p in other.named_parameters())

    # 在锚点外设置一个已知候选位置，验证先算Pair梯度、再回拉，不是相反。
    anchor = RowParameterAnchor(model)
    with torch.no_grad():
        model.output.bias[0].add_(.2)
        model.lexical_output[-1].bias[3].add_(.3)
    candidate = {name: p.detach().clone() for name, p in model.named_parameters()}
    expected_recorder = Recorder(model)
    torch.manual_seed(91)
    legacy.train_pairs(model, pairs[:1], expected_recorder, pair_ce, device)
    actual_recorder = Recorder(model)
    torch.manual_seed(91)
    result, records = cfg.train_anchored_pairs(model, pairs[:1], actual_recorder, pair_ce, device, anchor)
    compare_gradients(expected_recorder.gradients[0], actual_recorder.gradients[0])
    for group in actual_recorder.param_groups:
        retention = 1 / (1 + group['lr'] * cfg.ANCHOR_STRENGTH)
        for p in group['params']:
            name, reference = anchor.references[id(p)]
            torch.testing.assert_close(p, reference + retention * (candidate[name] - reference), rtol=1e-6, atol=1e-7)
    assert records[0]['original_distance_before_pull'] > 0
    assert records[0]['lexical_distance_before_pull'] > 0
    print('训练路径通过：Row直接复用旧版；分组LR相同；锚定0复现Pair梯度；回拉顺序及0.5系数正确。')


class NoUpdate:
    def __init__(self, model):
        self.model = model
        self.param_groups = cfg.build_optimizer(model).param_groups
        self.before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        assert all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        self.steps += 1


def check_gpu(train):
    cfg.set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(),
                                              revision=cfg.BASE_REVISION, local_files_only=True)
    features = cfg.build_feature_tensor(train)
    rows = cfg.DiCoDataset(train, tokenizer, features)
    pairs = cfg.PairDataset(train, tokenizer, features)
    row_batch = next(iter(DataLoader(rows, batch_size=cfg.ROW_BATCH_SIZE)))
    pair_batch = next(iter(DataLoader(pairs, batch_size=cfg.PAIR_BATCH_SIZE)))
    torch.cuda.reset_peak_memory_stats()
    model = cfg.build_model(tokenizer).to(device).train()
    recorder = NoUpdate(model)
    # 模拟两份Adam矩缓冲的显存，不运行Adam更新。
    moment_buffers = [(torch.zeros_like(p), torch.zeros_like(p)) for p in model.parameters() if p.requires_grad]
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    cfg.train_rows(model, [row_batch], recorder, row_ce, device)
    anchor = RowParameterAnchor(model)
    result, records = cfg.train_anchored_pairs(model, [pair_batch], recorder, pair_ce, device, anchor)
    assert recorder.steps == 2 and len(moment_buffers) == len(anchor.references)
    assert all(torch.equal(p.detach().cpu(), recorder.before[name]) for name, p in model.named_parameters())
    torch.cuda.synchronize()
    print('真实GPU检查通过：未更新模型；固定128 padding、两个LR、词汇分支、锚点及等量Adam矩显存均覆盖。')
    print({'parameters': sum(p.numel() for p in model.parameters()),
           'row_shape': list(row_batch[0].shape), 'pair_shape': list(pair_batch[0].shape),
           'peak_allocated_GiB_with_moment_buffers': round(torch.cuda.max_memory_allocated()/2**30, 3),
           **result, 'anchor_stats': records[0]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, dev = cfg.read_official_data()
    check_formula()
    check_data(train, dev)
    check_training()
    if args.gpu:
        check_gpu(train)
    print('全部检查通过；没有执行正式训练。')
