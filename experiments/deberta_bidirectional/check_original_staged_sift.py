"""SiFT机制检查；不运行真实训练或优化器参数更新。"""
import argparse
import copy
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.deberta_bidirectional import train_original_staged_sift as cfg
from experiments.deberta_bidirectional.sift_regularizer import SiFT, capture_rng, symmetric_kl
from experiments.deberta_bidirectional.check_original_staged_replay import ToyRows, GradientRecorder, toy_model
from experiments.deberta_stage_average import train_deberta_stage_average as legacy


def add_sift(model, **kwargs):
    model.sift = SiFT(model, **kwargs)
    return model


def check_toy(train, folder):
    rows = ToyRows(train.iloc[:5])
    batch = cfg.collate_rows([rows[i] for i in range(len(rows))])
    ids, mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))

    # 中心化的KL与官方双向stop-target交叉熵梯度相同。
    a, b = torch.randn(5, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True)
    ours = symmetric_kl(a, b)
    original = (-(F.softmax(a.detach(), -1) * F.log_softmax(b, -1)).sum(-1)
                -(F.softmax(b.detach(), -1) * F.log_softmax(a, -1)).sum(-1)).mean()
    first = torch.autograd.grad(ours, [a, b], retain_graph=True)
    second = torch.autograd.grad(original, [a, b])
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)

    # zero weight与旧Row CE逐参数梯度相同，不注册归一化/扰动hook。
    model = toy_model()
    model.sift = None
    other = copy.deepcopy(model)
    old_recorder, new_recorder = GradientRecorder(model), GradientRecorder(other)
    torch.manual_seed(33)
    legacy.train_rows(model, [batch], old_recorder, criterion, torch.device('cpu'))
    torch.manual_seed(33)
    cfg.train_rows(other, [batch], new_recorder, criterion, torch.device('cpu'), sift_weight=0)
    for name, gradient in old_recorder.gradients[0].items():
        torch.testing.assert_close(gradient, new_recorder.gradients[0][name], rtol=0, atol=0)

    # 采用相同dropout后，零扰动必须得到零正则，不能偷偷变成R-Drop。
    zero = add_sift(toy_model(), init_perturbation=0, step_size=0)
    zero.train()
    state = capture_rng(torch.device('cpu'))
    clean = zero(ids, mask, None)
    after = torch.get_rng_state()
    regularizer = zero.sift.loss(clean, lambda: zero(ids, mask, None), mask, state)
    torch.testing.assert_close(regularizer, torch.zeros_like(regularizer), atol=1e-6, rtol=0)
    assert torch.equal(after, torch.get_rng_state())
    assert all(p.grad is None and p.requires_grad for p in zero.parameters())

    # 非零扰动：delta搜索不写模型grad；最终损失向encoder和分类头反传。
    model = add_sift(toy_model(), noise_seed=91)
    model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state = capture_rng(torch.device('cpu'))
    clean = model(ids, mask, None)
    row_loss = criterion(clean, labels)
    regularizer = model.sift.loss(clean, lambda: model(ids, mask, None), mask, state)
    assert all(p.grad is None and p.requires_grad for p in model.parameters())
    assert model.sift.last_stats['delta_max'] > 0
    assert model.sift.last_stats['delta_gradient_norm'] > 0
    (row_loss + regularizer).backward()
    assert torch.isfinite(regularizer) and regularizer > 0
    assert model.output.weight.grad.abs().sum() > 0
    assert model.deberta_model.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert not model.sift.active and model.sift.delta is None and model.sift.cached is None

    # 真正的训练函数只调用一次step；recorder.step只记录梯度，不更新参数。
    model.zero_grad(set_to_none=True)
    recorder = GradientRecorder(model)
    result, diagnostics = cfg.train_rows(model, [batch], recorder, criterion, torch.device('cpu'), 1.0)
    assert len(recorder.gradients) == 1 and result['row_ce_rows'] == 5
    assert len(diagnostics) == 1 and diagnostics[0]['delta_max'] > 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())

    # Pair仍为原三组合损失，且不触发扰动。
    first, second = toy_model(), toy_model()
    second.load_state_dict(first.state_dict())
    add_sift(first)
    add_sift(second)
    pair_batch = (ids[:2], mask[:2], ids[2:4], mask[2:4], torch.tensor([0, 2]))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    old_recorder, new_recorder = GradientRecorder(first), GradientRecorder(second)
    torch.manual_seed(66)
    legacy.train_pairs(first, [pair_batch], old_recorder, pair_ce, torch.device('cpu'))
    torch.manual_seed(66)
    cfg.train_pairs(second, [pair_batch], new_recorder, pair_ce, torch.device('cpu'))
    for name, gradient in old_recorder.gradients[0].items():
        torch.testing.assert_close(gradient, new_recorder.gradients[0][name], rtol=0, atol=0)
    assert second.sift.last_stats == {} and not second.sift.active

    # 保存的是完整原模型参数；恢复时重新安装相同的无参数归一化。
    model.eval()
    with torch.inference_mode():
        expected = model(ids, mask, None)
    cfg.save_weights(model, folder / 'toy.pth')
    restored = add_sift(toy_model()).eval()
    cfg.load_weights(restored, folder / 'toy.pth')
    with torch.inference_mode():
        actual = restored(ids, mask, None)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert all((ROOT / name).is_file() for name in cfg.source_hashes())
    print('CPU检查通过：零权重原Row梯度、扰动搜索与梯度恢复、相同dropout、checkpoint反向、原Pair梯度、保存重载。')


def check_real(train, folder, gpu=False):
    device = torch.device('cuda' if gpu else 'cpu')
    options = cfg.sift_settings(1.0)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=cfg.get_model_cache_dir(),
                                              revision=cfg.BASE_REVISION, local_files_only=True)
    rows = cfg.RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    sample = list(range(cfg.ROW_BATCH_SIZE if gpu else 2))
    batch = cfg.collate_rows([rows[i] for i in sample])
    if gpu:
        width = int(rows.encoded['attention_mask'].sum(1).max())
        batch['input_ids'] = rows.encoded['input_ids'][sample, :width].contiguous()
        batch['attention_mask'] = rows.encoded['attention_mask'][sample, :width].contiguous()
    model = cfg.build_model(tokenizer, sift_options=options).to(device)
    model.deberta_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 184001800
    state_buffers = [(torch.zeros_like(p), torch.zeros_like(p)) for p in model.parameters()] if gpu else []
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
    state = capture_rng(device)
    clean = model(ids, mask, None)
    row_loss = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, device=device))(clean, labels)
    adv = model.sift.loss(clean, lambda: model(ids, mask, None), mask, state)
    assert all(p.grad is None and p.requires_grad for p in model.parameters())
    (row_loss + adv).backward()
    assert torch.isfinite(adv) and model.output.weight.grad is not None
    assert model.deberta_model.embeddings.word_embeddings.weight.grad is not None
    if gpu:
        torch.cuda.synchronize()
        print('GPU单批SiFT检查：', {'rows': len(ids), 'tokens': ids.shape[1], 'sift_loss': float(adv.detach()),
              'peak_allocated_GiB': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
              'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / 1024**3, 3), 'optimizer_steps': 0})
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        expected = model(ids, mask, None).cpu()
    model_dir = folder / 'real'
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    (model_dir / 'run_config.json').write_text(json.dumps({
        'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION, 'sift': options,
    }), encoding='utf-8')
    cfg.save_weights(model, model_dir / 'best_model.pth')
    del model, clean, row_loss, adv, state_buffers
    if gpu:
        torch.cuda.empty_cache()
    restored, _, _ = cfg.load_saved_model(model_dir, device)
    with torch.inference_mode():
        actual = restored(ids, mask, None).cpu()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print('真实NLI模型SiFT与完整权重重载通过；没有优化器参数更新。')


def main(real=False, gpu=False):
    torch.set_num_threads(4)
    train, _, _ = cfg.read_data()
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'original_staged_sift_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_toy(train, folder)
        if real or gpu:
            check_real(train, folder, gpu)
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('original_staged_sift_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args.real, args.gpu)
