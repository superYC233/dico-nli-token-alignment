"""检查BLOOP公式、梯度、真实数据覆盖；step仅记录梯度，不更新参数。"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_manual_aug import train_relation_bloop as cfg
from experiments.deberta_manual_aug.bloop_gradients import BloopGradients
from experiments.deberta_bidirectional.check_original_staged_replay import (
    ToyRows, GradientRecorder, toy_model,
)
from experiments.deberta_stage_average import train_deberta_stage_average as legacy


def check_formula():
    for pair in [torch.tensor([2., 3.]), torch.tensor([-2., 3.])]:
        parameter = nn.Parameter(torch.zeros(2))
        state = BloopGradients([parameter])
        row = torch.tensor([1., 0.])
        expected = row + pair - torch.dot(row, pair) / (row.square().sum() + state.eps) * row
        stats = state.assign([row.clone()], [pair.clone()])
        torch.testing.assert_close(parameter.grad, expected)
        assert stats['projection_applied']
        assert abs(stats['ema_dot_projected_pair']) < 1e-6
        torch.testing.assert_close(state.ema[0], row)

    # 第二次的EMA不等于当前Row，必须减去EMA方向。
    parameter = nn.Parameter(torch.zeros(2))
    state = BloopGradients([parameter], ema_decay=.9)
    state.update_ema([torch.tensor([1., 0.])])
    row, pair, average = torch.tensor([0., 1.]), torch.tensor([1., 0.]), torch.tensor([.9, .1])
    expected = row + pair - torch.dot(average, pair) / (average.square().sum() + state.eps) * average
    state.assign([row.clone()], [pair.clone()])
    torch.testing.assert_close(parameter.grad, expected)
    torch.testing.assert_close(state.ema[0], average)
    assert state.updates == 2 and not state.ema[0].requires_grad

    # 当前某参数两项梯度皆None，但历史EMA非0，全局投影仍需覆盖它。
    parameters = [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))]
    state = BloopGradients(parameters)
    state.update_ema([torch.ones(1), torch.ones(1)])
    row, pair, average = torch.tensor([1., 0.]), torch.tensor([2., 0.]), torch.tensor([1., .9])
    expected = row + pair - torch.dot(average, pair) / (average.square().sum() + state.eps) * average
    state.assign([torch.ones(1), None], [2 * torch.ones(1), None])
    torch.testing.assert_close(torch.cat([p.grad for p in parameters]), expected)

    parameter = nn.Parameter(torch.zeros(2))
    state = BloopGradients([parameter])
    state.assign([torch.zeros(2)], [torch.tensor([2., 3.])])
    torch.testing.assert_close(parameter.grad, torch.tensor([2., 3.]))
    state.assign([torch.tensor([1., 2.])], [None])
    torch.testing.assert_close(parameter.grad, torch.tensor([1., 2.]))
    print('BLOOP公式通过：正/负内积均投影、EMA方向、零梯度和None梯度。')


def check_coverage(train):
    rows = ToyRows(train)
    pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(pairs.pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(123))
    batches = list(cfg.pair_batches(rows, loader, torch.Generator().manual_seed(456)))
    source = torch.cat([batch['source_rows'] for batch in batches])
    assert len(batches) == 87 and len(pairs) == 1380
    assert torch.equal(source.sort().values, torch.arange(3290))
    assert sum(batch['n_count'] for batch in batches) == 530
    assert max(len(batch['labels']) for batch in batches) == 39
    targets = torch.tensor([[0, 1], [1, 0], [2, 2]])
    for batch in batches:
        count = batch['pair_count']
        expected = targets[batch['pair_targets']]
        assert torch.equal(batch['labels'][:count], expected[:, 0])
        assert torch.equal(batch['labels'][count:2 * count], expected[:, 1])
        assert batch['labels'][2 * count:].eq(3).all()
    print('真实Train覆盖通过：3290行各一次、1380个Pair、530条N、87批。')


def check_losses(train):
    all_rows = ToyRows(train)
    all_pairs = cfg.RealPairDataset(all_rows)
    indices = [index for pair in all_pairs.pairs[:2] for index in pair[:2]]
    indices += torch.where(all_rows.labels.eq(3))[0][:3].tolist()
    rows = ToyRows(train.iloc[indices])
    pairs = cfg.RealPairDataset(rows)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS),
                                  label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    device = torch.device('cpu')

    # 新Row阶段和原版有相同逐参数梯度，维护EMA不消耗随机数。
    model = toy_model()
    original = copy.deepcopy(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    old_recorder, recorder = GradientRecorder(original), GradientRecorder(model)
    batches = list(DataLoader(rows, batch_size=3, collate_fn=cfg.collate_rows))
    state = BloopGradients(list(model.parameters()))
    torch.manual_seed(88)
    legacy.train_rows(original, batches, old_recorder, row_ce, device)
    torch.manual_seed(88)
    row_result = cfg.train_rows(model, batches, recorder, row_ce, device, state)
    assert row_result['row_ce_rows'] == 7 and state.updates == len(batches)
    for old, new in zip(old_recorder.gradients, recorder.gradients):
        assert old.keys() == new.keys()
        for name in old:
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
    assert all(torch.equal(old, new) for old, new in zip(old_recorder.states, recorder.states))

    # 完整训练函数运行两批，但step仅记录，不更新权重。
    pair_loader = DataLoader(pairs.pairs, batch_size=1, shuffle=False,
                             generator=torch.Generator().manual_seed(4))
    result, records = cfg.train_pair_bloop(
        model, rows, pair_loader, torch.Generator().manual_seed(5), recorder,
        row_ce, pair_ce, device, state,
    )
    assert result['pair_stage_rows'] == 7 and result['pair_stage_n_rows'] == 3
    assert result['pair_ce_groups'] == 2 and len(records) == 2
    assert state.updates == len(batches) + 2
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(torch.isfinite(value).all() for step in recorder.gradients for value in step.values())

    # mixed对照必须等于(Row CE + 0.5*Pair CE).backward()；0.5只能乘一次。
    batch = next(cfg.pair_batches(rows, pair_loader, torch.Generator().manual_seed(6)))
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    logits = model(batch['input_ids'], batch['attention_mask'], None)
    row_loss, pair_loss = cfg.compute_losses(logits, batch['labels'], batch['pair_targets'], row_ce, pair_ce)
    row_gradients, pair_gradients = cfg.separate_gradients(row_loss, pair_loss, list(model.parameters()), cfg.PAIR_WEIGHT)
    BloopGradients(list(model.parameters())).assign(row_gradients, pair_gradients, mode='mixed')
    actual = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    logits = model(batch['input_ids'], batch['attention_mask'], None)
    row_loss, pair_loss = cfg.compute_losses(logits, batch['labels'], batch['pair_targets'], row_ce, pair_ce)
    (row_loss + cfg.PAIR_WEIGHT * pair_loss).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, actual[name], rtol=2e-5, atol=2e-6)

    # 检查非reentrant检查点能同时支持两次autograd.grad。
    checkpointed = copy.deepcopy(model)
    checkpointed.deberta_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    checkpointed.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    logits = checkpointed(batch['input_ids'], batch['attention_mask'], None)
    row_loss, pair_loss = cfg.compute_losses(logits, batch['labels'], batch['pair_targets'], row_ce, pair_ce)
    grads = cfg.separate_gradients(row_loss, pair_loss, list(checkpointed.parameters()), cfg.PAIR_WEIGHT)
    BloopGradients(list(checkpointed.parameters())).assign(*grads, mode='mixed')
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad, rtol=2e-5, atol=2e-6)
    print('损失及流程通过：Row与原版一致、Pair仅乘一次0.5、EMA跨阶段保留、权重未更新。')


def check_gpu(train):
    device = torch.device('cuda')
    cfg.set_seed(cfg.SEED)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(),
                                              revision=cfg.BASE_REVISION, local_files_only=True)
    rows = cfg.RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.RealPairDataset(rows)
    pair_loader = DataLoader(pairs.pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=False)
    # 检查这个完整批次方案中序列最长的一批，不只取通常较短的第一批。
    batch = max(cfg.pair_batches(rows, pair_loader, torch.Generator().manual_seed(7)),
                key=lambda item: (item['input_ids'].shape[1], len(item['labels'])))
    torch.cuda.reset_peak_memory_stats()
    model = cfg.build_model(tokenizer).to(device).train()
    model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    # Tensor版本号用于确认从未对权重做原地写入，不复制整套大模型。
    versions = [p._version for p in parameters]
    state = BloopGradients(parameters, cfg.EMA_DECAY, cfg.PROJECTION_EPS)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
                                  label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), None)
    row_loss, pair_loss = cfg.compute_losses(logits, batch['labels'].to(device),
                                             batch['pair_targets'].to(device), row_ce, pair_ce)
    grads = cfg.separate_gradients(row_loss, pair_loss, parameters, cfg.PAIR_WEIGHT)
    stats = state.assign(*grads)
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert versions == [p._version for p in parameters]
    torch.cuda.synchronize()
    print('真实GPU前后向通过：没有optimizer.step，没有训练或保存模型。')
    print({'rows': len(batch['labels']), 'tokens': batch['input_ids'].shape[1],
           'parameters': sum(p.numel() for p in parameters),
           'peak_allocated_GiB_without_Adam_state': round(torch.cuda.max_memory_allocated() / 2**30, 3),
           'row_loss': float(row_loss.detach()), 'pair_loss': float(pair_loss.detach()),
           **stats})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, dev, digest = cfg.read_data()
    assert len(dev) == 660
    check_formula()
    check_coverage(train)
    check_losses(train)
    if args.gpu:
        check_gpu(train)
    print('全部检查通过。Train SHA256：', digest)
