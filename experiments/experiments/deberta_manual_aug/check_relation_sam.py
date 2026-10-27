"""SAM公式和训练路径检查；optimizer.step只核对梯度，不做真实更新。"""
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
from experiments.deberta_manual_aug import train_relation_sam as cfg
from experiments.deberta_manual_aug.sam_perturbation import SAMParameters
from experiments.deberta_bidirectional.check_original_staged_replay import (
    ToyRows, GradientRecorder, toy_model,
)
from experiments.deberta_stage_average import train_deberta_stage_average as legacy


def compare_gradients(first, second):
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=2e-5, atol=2e-6)


def check_formula():
    first = nn.Parameter(torch.tensor([3., 4.]))
    unused = nn.Parameter(torch.tensor([7.]))
    original = first.detach().clone()
    (.5 * first.square().sum()).backward()
    sam = SAMParameters([first, unused], rho=.05)
    stats = sam.perturb()
    torch.testing.assert_close(first, torch.tensor([3.03, 4.04]))
    assert abs(stats['sam_radius'] - .05) < 1e-10
    torch.testing.assert_close(unused, torch.tensor([7.]))
    first.grad = None
    sam.restore()
    assert torch.equal(first, original) and not sam.original
    # 梯度为0时扰动也是0，不产生NaN。
    first.grad = torch.zeros_like(first)
    stats = sam.perturb()
    assert stats['sam_radius'] == 0
    sam.restore()
    assert torch.equal(first, original)
    print('SAM公式通过：全局L2半径、零梯度、None梯度、精确恢复。')


def check_data(train):
    rows = ToyRows(train)
    pairs = cfg.RealPairDataset(rows)
    indices = torch.tensor([index for pair in pairs.pairs for index in pair[:2]])
    assert len(rows) == 3290 and len(pairs) == 1380
    assert torch.equal(indices.sort().values, torch.where(rows.labels.ne(3))[0])
    assert rows.labels.bincount(minlength=4).tolist() == [933, 933, 894, 530]
    assert (len(rows) + cfg.ROW_BATCH_SIZE - 1) // cfg.ROW_BATCH_SIZE == 206
    assert (len(pairs) + cfg.PAIR_BATCH_SIZE - 1) // cfg.PAIR_BATCH_SIZE == 87
    print('数据通过：3290行、1380组Pair、530条N，原Row/Pair曝光不变。')


def check_training(train):
    all_rows = ToyRows(train)
    all_pairs = cfg.RealPairDataset(all_rows)
    selected = [next(pair for pair in all_pairs.pairs if pair[2] == target) for target in [0, 2]]
    indices = [index for pair in selected for index in pair[:2]]
    indices += torch.where(all_rows.labels.eq(3))[0][:3].tolist()
    rows = ToyRows(train.iloc[indices])
    pairs = cfg.RealPairDataset(rows)
    device = torch.device('cpu')
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS),
                                  label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    batches = list(DataLoader(rows, batch_size=3, collate_fn=cfg.collate_rows))

    model = toy_model()
    original = copy.deepcopy(model)
    clean = {name: p.detach().clone() for name, p in model.named_parameters()}
    old_recorder, zero_recorder = GradientRecorder(original), GradientRecorder(model)
    torch.manual_seed(88)
    legacy.train_rows(original, batches, old_recorder, row_ce, device)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(88)
    result, records = cfg.train_rows_sam(model, batches, zero_recorder, row_ce, device, rho=0)
    assert result['row_ce_rows'] == 7 and result['row_forward_rows'] == 7
    assert torch.equal(baseline_rng, torch.get_rng_state())
    for old, new in zip(old_recorder.gradients, zero_recorder.gradients):
        compare_gradients(old, new)

    # SAM结束后参数必须精确还原；第二次前向不应额外推进dropout随机序列。
    recorder = GradientRecorder(model)
    torch.manual_seed(88)
    result, records = cfg.train_rows_sam(model, batches, recorder, row_ce, device)
    assert result['row_ce_rows'] == 7 and result['row_forward_rows'] == 14
    assert len(recorder.gradients) == len(batches)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    assert all(torch.equal(p, clean[name]) for name, p in model.named_parameters())

    # 独立手算第一批扰动点梯度，检查没有累加第一次的干净梯度。
    batch = batches[0]
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[0])
    row_ce(model(batch['input_ids'], batch['attention_mask'], None), batch['label']).backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1) for p in parameters])).item()
    with torch.no_grad():
        for p in parameters:
            p.add_(p.grad, alpha=cfg.SAM_RHO / (norm + 1e-12))
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[0])
    row_ce(model(batch['input_ids'], batch['attention_mask'], None), batch['label']).backward()
    manual = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    compare_gradients(manual, recorder.gradients[0])
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(clean[name])

    # 原Pair循环的逐参数梯度必须相同，0.5只乘一次。
    pair_batches = list(DataLoader(pairs, batch_size=1, collate_fn=cfg.collate_pairs))
    other = copy.deepcopy(model)
    first_recorder, second_recorder = GradientRecorder(model), GradientRecorder(other)
    torch.manual_seed(99)
    legacy.train_pairs(model, pair_batches, first_recorder, pair_ce, device)
    torch.manual_seed(99)
    cfg.train_pairs(other, pair_batches, second_recorder, pair_ce, device)
    for first, second in zip(first_recorder.gradients, second_recorder.gradients):
        compare_gradients(first, second)

    # 非reentrant检查点与两次SAM反向可以共同使用。
    checkpointed = copy.deepcopy(model)
    checkpointed.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    first_recorder, second_recorder = GradientRecorder(model), GradientRecorder(checkpointed)
    torch.manual_seed(100)
    cfg.train_rows_sam(model, batches[:1], first_recorder, row_ce, device)
    torch.manual_seed(100)
    cfg.train_rows_sam(checkpointed, batches[:1], second_recorder, row_ce, device)
    compare_gradients(first_recorder.gradients[0], second_recorder.gradients[0])
    assert all(torch.equal(p, clean[name]) for name, p in model.named_parameters())
    assert all(torch.equal(p, clean[name]) for name, p in checkpointed.named_parameters())
    print('训练路径通过：rho0等于原Row、SAM仅用扰动点梯度、原Pair梯度一致、检查点兼容。')


def check_pair_sam(train):
    all_rows = ToyRows(train)
    all_pairs = cfg.RealPairDataset(all_rows)
    selected = [next(pair for pair in all_pairs.pairs if pair[2] == target) for target in [0, 2]]
    rows = ToyRows(train.iloc[[index for pair in selected for index in pair[:2]]])
    batches = list(DataLoader(cfg.RealPairDataset(rows), batch_size=1, collate_fn=cfg.collate_pairs))
    forward = next(batch for batch in batches if int(batch[-1][0]) == 0)
    equivalence = next(batch for batch in batches if int(batch[-1][0]) == 2)
    a, ma, b, mb, target = forward
    # 包括FB、BF、EE三类，BF用FB的两个方向互换构造检查输入。
    batches = [forward, (b, mb, a, ma, torch.ones_like(target)), equivalence]
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS),
                                     label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    model = toy_model()
    original = copy.deepcopy(model)
    clean = {name: p.detach().clone() for name, p in model.named_parameters()}
    old_recorder, zero_recorder = GradientRecorder(original), GradientRecorder(model)
    torch.manual_seed(333)
    old_loss = cfg.train_pairs(original, batches, old_recorder, criterion, device)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(333)
    result, _ = cfg.train_pairs_sam(model, batches, zero_recorder, criterion, device, rho=0)
    assert result['pair_loss'] == old_loss
    assert result['pair_ce_groups'] == 3 and result['pair_forward_rows'] == 6
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for old, new in zip(old_recorder.gradients, zero_recorder.gradients):
        compare_gradients(old, new)

    class RestoredRecorder(GradientRecorder):
        def step(self):
            # 在模拟AdamW调用的位置检查，而不只在整个函数返回后检查。
            assert all(torch.equal(p, clean[name]) for name, p in self.model.named_parameters())
            super().step()

    recorder = RestoredRecorder(model)
    torch.manual_seed(333)
    result, records = cfg.train_pairs_sam(model, batches, recorder, criterion, device)
    assert result['pair_ce_groups'] == 3 and result['pair_forward_rows'] == 12
    assert len(recorder.gradients) == 3 and len(records) == 3
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert all(torch.equal(p, clean[name]) for name, p in model.named_parameters())

    # 手工计算第一批的扰动点梯度，两个方向完整重放，0.5仅乘一次。
    ids0, mask0, ids1, mask1, targets = batches[0]
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[0])
    first, second = model(ids0, mask0, None), model(ids1, mask1, None)
    (cfg.PAIR_WEIGHT * criterion(cfg.get_pair_logits(first, second), targets)).backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1) for p in parameters])).item()
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(parameter.grad, alpha=cfg.PAIR_SAM_RHO / (norm + 1e-12))
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[0])
    first, second = model(ids0, mask0, None), model(ids1, mask1, None)
    (cfg.PAIR_WEIGHT * criterion(cfg.get_pair_logits(first, second), targets)).backward()
    manual = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    compare_gradients(manual, recorder.gradients[0])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(clean[name])

    # 非reentrant检查点在两个方向的两次Pair反向中也必须可用。
    checkpointed = copy.deepcopy(model)
    checkpointed.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    first_recorder, second_recorder = RestoredRecorder(model), RestoredRecorder(checkpointed)
    torch.manual_seed(334)
    cfg.train_pairs_sam(model, batches[:1], first_recorder, criterion, device)
    torch.manual_seed(334)
    cfg.train_pairs_sam(checkpointed, batches[:1], second_recorder, criterion, device)
    compare_gradients(first_recorder.gradients[0], second_recorder.gradients[0])

    # 第二次的第二个方向出错时，也必须恢复临时参数，不能遗留扰动。
    failed = toy_model()
    before = {name: p.detach().clone() for name, p in failed.named_parameters()}
    calls = [0]

    def fail_fourth_forward(module, inputs):
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError('intentional Pair-SAM restore check')

    handle = failed.register_forward_pre_hook(fail_fourth_forward)
    failed_recorder = GradientRecorder(failed)
    try:
        cfg.train_pairs_sam(failed, batches[:1], failed_recorder, criterion, device)
    except RuntimeError as error:
        assert str(error) == 'intentional Pair-SAM restore check'
    else:
        raise AssertionError('检查没有进入第二遍的第二个方向。')
    finally:
        handle.remove()
    assert not failed_recorder.gradients
    assert all(torch.equal(p, before[name]) for name, p in failed.named_parameters())
    print('Pair-SAM通过：rho0与旧版一致、完整双向RNG重放、0.5仅一次、更新前恢复、检查点与异常恢复。')


class NoUpdate:
    def __init__(self, model):
        self.model = model
        self.original = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        # 验证交给AdamW之前参数已经还原，但不执行AdamW。
        for name, parameter in self.model.named_parameters():
            assert torch.equal(parameter.detach().cpu(), self.original[name]), name
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        self.steps += 1


def check_gpu(train):
    cfg.set_seed(cfg.SEED)
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(),
                                              revision=cfg.BASE_REVISION, local_files_only=True)
    rows = cfg.RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.RealPairDataset(rows)
    # 用训练集中最长的16行检查，不只检查短样本。
    indices = rows.encoded['attention_mask'].sum(1).argsort(descending=True)[:16].tolist()
    row_batch = cfg.collate_rows([rows[index] for index in indices])
    pair_loader = DataLoader(pairs, batch_size=cfg.PAIR_BATCH_SIZE, collate_fn=cfg.collate_pairs)
    pair_batch = max(pair_loader, key=lambda b: max(b[0].shape[1], b[2].shape[1]))
    torch.cuda.reset_peak_memory_stats()
    model = cfg.build_model(tokenizer).to(device)
    model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    recorder = NoUpdate(model)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS, device=device),
                                  label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    result, diagnostics = cfg.train_rows_sam(model, [row_batch], recorder, row_ce, device)
    pair_result, pair_diagnostics = cfg.train_pairs_sam(model, [pair_batch], recorder, pair_ce, device)
    assert recorder.steps == 2
    torch.cuda.synchronize()
    print('真实GPU检查通过：临时SAM扰动已精确恢复，无真实优化器更新。')
    print({'parameters': sum(p.numel() for p in model.parameters()),
           'row_shape': list(row_batch['input_ids'].shape),
           'pair_shapes': [list(pair_batch[0].shape), list(pair_batch[2].shape)],
           'peak_allocated_GiB_without_Adam_state': round(torch.cuda.max_memory_allocated() / 2**30, 3),
           **result, **pair_result, 'sam_batch': diagnostics[0], 'pair_sam_batch': pair_diagnostics[0]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, dev, digest = cfg.read_data()
    assert len(dev) == 660
    check_formula()
    check_data(train)
    check_training(train)
    check_pair_sam(train)
    if args.gpu:
        check_gpu(train)
    print('全部检查通过；未正式训练。Train SHA256：', digest)
