"""margin约束检查：CPU公式/梯度；--gpu只做真实前后向，不更新模型。"""
import argparse
import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical import train_deberta_lexical_margin as cfg
from experiments.deberta_lexical.margin_constraints import MarginProtection
from experiments.deberta_lexical.check_lexical_anchor import toy_model
from experiments.deberta_lexical.test_cpu import TinyTokenizer
from experiments.deberta_lexical.train_deberta_lexical import train_pairs as old_train_pairs


class Recorder:
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        values = {name: p.grad.detach().clone() for name, p in self.model.named_parameters()
                  if p.grad is not None}
        assert all(torch.isfinite(value).all() for value in values.values())
        self.gradients.append(values)


def synthetic_reference(labels):
    values = torch.zeros(len(labels), 4)
    values[torch.arange(len(labels)), labels] = 3.0
    return values


def check_formula():
    labels = torch.tensor([0, 3, 0])
    reference = torch.tensor([[4., 0., 0., -1.], [0., 0., 0., 3.], [0., 0., 3., 0.]], requires_grad=True)
    protection = MarginProtection(reference, labels, torch.device('cpu'))
    assert protection.protected.tolist() == [True, True, False]
    assert protection.floors.tolist() == [1., 1., 0.]
    assert not protection.reference_logits.requires_grad
    ids = torch.arange(3)
    current = torch.tensor([[0., 0., 0., 2.], [0., 0., 0., 3.], [0., 0., 0., 10.]], requires_grad=True)
    violations = protection.constraint_parts(current, ids, protection.counts(ids))
    torch.testing.assert_close(violations, torch.tensor([3., 0., 0., 0.]))
    penalty = protection.primal_term(violations)
    penalty.backward()
    torch.testing.assert_close(current.grad[0], torch.tensor([-.25, 0., 0., .25]))
    assert current.grad[1:].eq(0).all() and reference.grad is None
    # 梯度用lambda_t=.25，随后才更新lambda_{t+1}=.25+.1*3=.55。
    protection.dual_step(violations.detach())
    torch.testing.assert_close(protection.multipliers, torch.tensor([.55, .25, .25, .25]))
    audit = protection.audit(current)
    assert audit[0]['protected_broken'] == 1 and audit[0]['previously_wrong_fixed'] == 0
    assert audit[3]['protected_retained'] == 1

    labels2 = torch.tensor([0, 0])
    other = MarginProtection(synthetic_reference(labels2), labels2, torch.device('cpu'))
    # 一条更自信不能抵消另一条损坏。
    values = torch.tensor([[100., 0., 0., 0.], [0., 0., 0., 2.]], requires_grad=True)
    violation = other.constraint_parts(values, torch.arange(2), other.counts(torch.arange(2)))
    torch.testing.assert_close(violation[0], torch.tensor(1.5))
    tie = MarginProtection(torch.zeros(1, 4), torch.tensor([0]), torch.device('cpu'))
    assert not tie.protected.any()
    print('公式通过：正确Train行筛选、margin下限、错误行不锁定、N纠正梯度、先primal后dual、损坏不抵消。')


def check_microbatch_formula():
    labels = torch.tensor([0, 0, 3, 2, 1])
    protection = MarginProtection(synthetic_reference(labels), labels, torch.device('cpu'))
    indices = torch.arange(5)
    values = torch.tensor([[0., 0., 0., 2.], [4., 0., 0., 0.], [3., 0., 0., 0.],
                           [0., 3., 0., 0.], [0., 0., 0., 3.]], requires_grad=True)
    denominator = protection.counts(indices)
    full = protection.constraint_parts(values, indices, denominator)
    protection.primal_term(full).backward()
    expected = values.grad.clone()
    copy_values = values.detach().clone().requires_grad_()
    parts = []
    for start in range(0, len(values), 2):
        part = protection.constraint_parts(copy_values[start:start + 2], indices[start:start + 2], denominator)
        protection.primal_term(part).backward()
        parts.append(part.detach())
    torch.testing.assert_close(torch.stack(parts).sum(0), full.detach(), rtol=0, atol=0)
    torch.testing.assert_close(copy_values.grad, expected, rtol=0, atol=0)
    print('分块通过：microbatch约束值/梯度等于整批，类别分母不重复归一化。')


def check_data():
    train, dev = cfg.read_official_data()
    features = cfg.build_feature_tensor(train)
    old = pd.read_csv(cfg.REFERENCE_DIR / 'train_features.csv', keep_default_na=False)
    assert old.instance_id.tolist() == train.instance_id.tolist()
    torch.testing.assert_close(features, torch.tensor(old[cfg.FEATURE_NAMES].to_numpy(), dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(features, cfg.build_feature_tensor(train[['text1', 'text2']]), rtol=0, atol=0)
    rows = cfg.DiCoDataset(train, TinyTokenizer(), features)
    pairs = cfg.PairDataset(train, TinyTokenizer(), features)
    steps = math.ceil(len(pairs) / cfg.PAIR_BATCH_SIZE)
    guard_batch_size = math.ceil(len(rows) / steps)
    loader = DataLoader(cfg.IndexedRows(rows), batch_size=guard_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.SEED + 1009))
    seen, n = [], 0
    for *_, labels, indices in loader:
        seen.extend(indices.tolist())
        n += int(labels.eq(3).sum())
    assert len(loader) == steps == 81 and guard_batch_size == 38
    assert sorted(seen) == list(range(3042)) and n == 478
    assert len(pairs) == 1282 and len(dev) == 660
    print('数据通过：3042/660、1282组；每轮保护遍历3042行一次，含478条N；词汇特征匹配原版。')
    return train


def small_frame():
    return pd.DataFrame([
        {'instance_id': 'f', 'pair_id': 'fb', 'text1': 'a black car', 'text2': 'a car', 'label': 'FORWARD_ENTAILMENT'},
        {'instance_id': 'b', 'pair_id': 'fb', 'text1': 'a car', 'text2': 'a black car', 'label': 'BACKWARD_ENTAILMENT'},
        {'instance_id': 'e0', 'pair_id': 'ee', 'text1': 'a small cat', 'text2': 'a little cat', 'label': 'EQUIVALENCE'},
        {'instance_id': 'e1', 'pair_id': 'ee', 'text1': 'a little cat', 'text2': 'a small cat', 'label': 'EQUIVALENCE'},
        {'instance_id': 'n', 'pair_id': 'n', 'text1': 'a dog', 'text2': 'a girl', 'label': 'NEGATIVE_OTHER'},
    ])


def check_paths():
    frame = small_frame()
    features = cfg.build_feature_tensor(frame)
    rows = cfg.DiCoDataset(frame, TinyTokenizer(), features)
    pairs = list(DataLoader(cfg.PairDataset(frame, TinyTokenizer(), features), batch_size=1))
    guard_loader = DataLoader(cfg.IndexedRows(rows), batch_size=3)
    labels = torch.tensor([0, 1, 2, 2, 3])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    model = toy_model().train()
    other = copy.deepcopy(model)
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    first, second = Recorder(model), Recorder(other)
    protection = MarginProtection(synthetic_reference(labels), labels, torch.device('cpu'))
    torch.manual_seed(93)
    baseline_loss = old_train_pairs(model, pairs, first, criterion, torch.device('cpu'))
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(93)
    result, _ = cfg.train_protected_pairs(other, pairs, None, second, criterion,
                                         torch.device('cpu'), protection, enabled=False)
    assert result['pair_loss'] == baseline_loss and torch.equal(baseline_rng, torch.get_rng_state())
    for old, new in zip(first.gradients, second.gradients):
        assert old.keys() == new.keys()
        for name in old:
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
    # 参照捕获不改变train模式或随机序列。
    before_rng = torch.get_rng_state()
    values = cfg.predict_preserving_rng(model, DataLoader(rows, batch_size=3), torch.device('cpu'))
    assert model.training and torch.equal(before_rng, torch.get_rng_state())
    assert values.shape == (5, 4)

    recorder = Recorder(model)
    protection = MarginProtection(synthetic_reference(labels), labels, torch.device('cpu'))
    result, records = cfg.train_protected_pairs(model, pairs, guard_loader, recorder, criterion,
                                               torch.device('cpu'), protection)
    assert result['protection_rows_seen'] == 5 and result['guard_lagrangian'] > 0
    assert len(records) == len(recorder.gradients) == 2 and model.training
    assert protection.multipliers.gt(cfg.INITIAL_DUAL).any()
    assert all(torch.equal(p, saved[name]) for name, p in model.named_parameters())
    print('路径通过：关闭保护逐参数复现原Pair梯度/RNG；非零保护可反传；模式恢复；未更新模型。')


def check_gpu(train):
    cfg.set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(),
                                              revision=cfg.BASE_REVISION, local_files_only=True)
    features = cfg.build_feature_tensor(train)
    rows = cfg.DiCoDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    pairs = cfg.PairDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    pair_batch = next(iter(DataLoader(pairs, batch_size=cfg.PAIR_BATCH_SIZE)))
    chosen = []
    for k, full in enumerate(['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']):
        chosen.extend(train.index[train.label.eq(full)][:10 if k < 3 else 8].tolist())
    guards = DataLoader(Subset(cfg.IndexedRows(rows), chosen), batch_size=38)
    labels = torch.tensor(train.label.map(dict(zip([
        'FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER',
    ], range(4)))).to_numpy())
    model = cfg.build_model(tokenizer).to(device).train()
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    # 构造参考确保非零保护反传被覆盖，不声称真实Row模型已经判对这些行。
    protection = MarginProtection(synthetic_reference(labels), labels, device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
                                    label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    recorder = Recorder(model)
    # GPU检查不保留整份GPU梯度副本，仅检查有限性。
    def no_update():
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        recorder.gradients.append({})
    recorder.step = no_update
    torch.cuda.reset_peak_memory_stats()
    moment_buffers = [(torch.zeros_like(p), torch.zeros_like(p)) for p in model.parameters() if p.requires_grad]
    result, records = cfg.train_protected_pairs(model, [pair_batch], guards, recorder,
                                               criterion, device, protection)
    assert all(torch.equal(p.detach().cpu(), before[name]) for name, p in model.named_parameters())
    assert len(moment_buffers) > 0 and len(recorder.gradients) == 1
    torch.cuda.synchronize()
    print('真实GPU前后向通过；没有执行AdamW或更新真实模型参数。')
    print({'pair_rows': 32, 'guard_rows': result['protection_rows_seen'],
           'guard_microbatch': cfg.PROTECTION_MICROBATCH,
           'peak_allocated_GiB_with_Adam_moment_buffers': round(torch.cuda.max_memory_allocated()/2**30, 3),
           'guard_lagrangian': result['guard_lagrangian'], 'multipliers': protection.multipliers.tolist()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    check_formula()
    check_microbatch_formula()
    train = check_data()
    check_paths()
    if args.gpu:
        check_gpu(train)
