"""CPU检查原模型、原Pair梯度与Row阶段PAWS；没有优化器参数更新。"""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_joint_pair.data import RowDataset
from experiments.deberta_manual_aug.train_boundary_manual import RealPairDataset
from experiments.deberta_stage_average import train_deberta_stage_average as legacy
from experiments.deberta_bidirectional.check_alignment_joint import AlignmentToyTokenizer, toy_backbone
from experiments.deberta_bidirectional.check_bidirectional import toy_frame
from experiments.deberta_bidirectional.check_paws_auxiliary import GradientRecorder, paws_frame
from experiments.deberta_bidirectional.paws_auxiliary import PawsDataset, equivalence_loss
from experiments.deberta_bidirectional import train_original_staged_paws as cfg


def toy_model():
    return DebertaDiCoModel(toy_backbone(), AlignmentToyTokenizer(), feature_size=0, use_lexical_features=False)


def compare_gradients(first, second):
    assert len(first.gradients) == len(second.gradients)
    for a, b in zip(first.gradients, second.gradients):
        assert set(a) == set(b)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=1e-6)


def check_toy(folder):
    tokenizer = AlignmentToyTokenizer()
    frame = toy_frame().iloc[:5].copy()
    dataset = RowDataset(frame, tokenizer, max_length=16)
    batch = next(iter(DataLoader(dataset, batch_size=5, shuffle=False,
                                  generator=torch.Generator().manual_seed(1))))
    row_batches = [batch] * 5
    pair_dataset = RealPairDataset(dataset)
    assert len(pair_dataset) == 2
    pair_batch = next(iter(DataLoader(pair_dataset, batch_size=2, shuffle=False,
                                      generator=torch.Generator().manual_seed(2))))
    pair_batches = [pair_batch] * 2
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS),
                                         label_smoothing=cfg.PAIR_LABEL_SMOOTHING)

    # PAWS权重0时，Row逐参数梯度与原版Row实现一致。
    first, second = toy_model(), toy_model()
    recorder_first, recorder_second = GradientRecorder(first), GradientRecorder(second)
    class ForbiddenLoader:
        def __iter__(self):
            raise AssertionError('权重0不应取PAWS批次。')
    torch.manual_seed(123)
    old_loss = legacy.train_rows(first, row_batches, recorder_first, criterion, torch.device('cpu'))
    torch.manual_seed(123)
    result, exposure = cfg.train_rows(second, row_batches, ForbiddenLoader(), recorder_second,
                                      criterion, torch.device('cpu'), paws_weight=0)
    compare_gradients(recorder_first, recorder_second)
    assert abs(old_loss - result['row_loss']) < 1e-6 and not exposure
    assert result['row_ce_rows'] == 25 and result['paws_rows'] == 0

    # Pair路径的原始三组合分数、权重、平滑与梯度必须保持相同。
    first, second = toy_model(), toy_model()
    recorder_first, recorder_second = GradientRecorder(first), GradientRecorder(second)
    torch.manual_seed(456)
    old_loss = legacy.train_pairs(first, pair_batches, recorder_first, pair_criterion, torch.device('cpu'))
    torch.manual_seed(456)
    result = cfg.train_pairs(second, pair_batches, recorder_second, pair_criterion, torch.device('cpu'))
    compare_gradients(recorder_first, recorder_second)
    assert abs(old_loss - result['pair_loss']) < 1e-6
    assert result['pair_ce_groups'] == 4 and result['pair_optimizer_steps'] == 2

    auxiliary_dataset = PawsDataset(paws_frame(), tokenizer, max_length=16)
    auxiliary_loader = DataLoader(
        auxiliary_dataset, batch_size=2, shuffle=False, collate_fn=cfg.collate_original_units,
        generator=torch.Generator().manual_seed(3),
    )
    model = toy_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    recorder = GradientRecorder(model)
    result, exposure = cfg.train_rows(model, row_batches, auxiliary_loader, recorder,
                                      criterion, torch.device('cpu'), paws_weight=.1, paws_every=4)
    assert result['row_ce_rows'] == 25 and result['paws_rows'] == 2
    assert result['paws_batches'] == 1 and exposure[0]['row_step'] == 4
    assert len(recorder.gradients) == 5
    assert all(torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters())

    # 同一次更新的梯度确实是Row + 0.1*(PAWS两个方向CE的平均)。
    auxiliary = next(iter(auxiliary_loader))
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[3])
    criterion(model(batch['input_ids'], batch['attention_mask'], None), batch['label']).backward()
    view_logits = model(auxiliary['input_ids'], auxiliary['attention_mask'], None)
    index, target = auxiliary['row_units'], auxiliary['labels']
    manual = (equivalence_loss(view_logits[2 * index], target)
              + equivalence_loss(view_logits[2 * index + 1], target)) / 2
    (.1 * manual).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, recorder.gradients[3][name], rtol=1e-5, atol=1e-6)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        expected = model(batch['input_ids'], batch['attention_mask'], None)
    cfg.save_weights(model, folder / 'original_toy.pth')
    restored = toy_model().eval()
    cfg.load_weights(restored, folder / 'original_toy.pth')
    with torch.inference_mode():
        torch.testing.assert_close(expected, restored(batch['input_ids'], batch['attention_mask'], None), rtol=0, atol=0)
    assert cfg.PAIR_WEIGHT == legacy.PAIR_WEIGHT
    assert cfg.PAIR_WEIGHTS == legacy.PAIR_WEIGHTS and cfg.PAIR_LABEL_SMOOTHING == legacy.PAIR_LABEL_SMOOTHING
    assert all((ROOT / name).is_file() for name in cfg.source_hashes())
    print('CPU检查通过：原Row/Pair参考梯度、PAWS只进Row且双向取平均、曝光与更新次数、完整权重重载、源码路径。')


def check_real(folder):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=cfg.BASE_REVISION, local_files_only=True,
    )
    model = cfg.build_model(tokenizer)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert count == 184001800 and all(parameter.requires_grad for parameter in model.parameters())
    assert not any('lora_' in name for name, _ in model.named_parameters())
    dataset = PawsDataset(paws_frame(), tokenizer, cfg.MAX_LENGTH)
    batch = cfg.collate_original_units([dataset[i] for i in range(len(dataset))])
    model.train()
    loss = cfg.paws_loss_original(model, batch, torch.device('cpu'))
    loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad is not None
    assert model.deberta_model.embeddings.word_embeddings.weight.grad is not None
    for parameter in model.parameters():
        parameter.grad = None
    with torch.no_grad():
        model.deberta_model.embeddings.word_embeddings.weight[1, 0].add_(.001)
    model.eval()
    with torch.inference_mode():
        expected = model(batch['input_ids'], batch['attention_mask'], None)
    model_dir = folder / 'real'
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION}
    (model_dir / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
    cfg.save_weights(model, model_dir / 'best_model.pth')
    del model
    restored, _, _ = cfg.load_saved_model(model_dir, torch.device('cpu'))
    with torch.inference_mode():
        actual = restored(batch['input_ids'], batch['attention_mask'], None)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print('真实原模型检查通过：184001800个全量参数、PAWS梯度、完整encoder重载；没有执行参数更新。')


def main(real=False):
    torch.set_num_threads(4)
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'original_staged_paws_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_toy(folder)
        if real:
            check_real(folder)
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('original_staged_paws_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
