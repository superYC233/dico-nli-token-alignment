"""只检查数据、Fisher掩码及梯度；不执行optimizer.step，不启动真实训练。"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, DebertaV2Config, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_child_tuning import config as cfg
from experiments.deberta_lexical_child_tuning.child_tuning import (
    ENCODER_PREFIX, select_fisher_mask, calculate_fisher_mask, apply_gradient_mask,
)
from experiments.deberta_lexical_child_tuning.train_deberta_child_tuning import (
    read_data, load_model, make_fisher_loader,
)
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2


def check_mask_logic():
    masks, _, _ = select_fisher_mask({'test': torch.tensor([1., 2., 3., 4.])}, 0.5)
    assert masks['test'].tolist() == [False, False, True, True]
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=32, relative_attention=False,
    )
    set_seed(42)
    model = DebertaDiCoModel(AutoModel.from_config(config), TinyTokenizer(), 20).eval()
    ids = torch.tensor([[1, 3, 2, 4, 2, 0], [1, 4, 2, 3, 2, 0]]).repeat(4, 1)
    features = torch.arange(160, dtype=torch.float32).reshape(8, 20) / 160
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    dataset = TensorDataset(ids, ids.ne(0).long(), features, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(142))
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    masks, stats, _ = calculate_fisher_mask(model, loader, criterion, torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert not model.training and all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
    assert stats['train_rows'] == 8 and stats['label_counts_F_B_E_N'] == [2, 2, 2, 2]
    assert all(name.startswith(ENCODER_PREFIX) for name in masks)
    assert 0 < stats['actual_retained_fraction'] < 1
    assert not any(name.startswith(('output.', 'compare_layer.', 'lexical_output.')) for name in masks)

    # 固定全1梯度，核对遮罩发生在encoder坐标，embedding和新增头不被误遮。
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    apply_gradient_mask(model, masks)
    for name, parameter in model.named_parameters():
        expected = masks[name].to(parameter.dtype) if name in masks else torch.ones_like(parameter)
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    print('小模型通过：只遮encoder层内梯度；Fisher不改权重、不改变后续随机数，模式和梯度已还原。')

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DebertaDiCoModel(AutoModel.from_config(config), TinyTokenizer(), 20).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, ids.ne(0).long(), features), model(ids, ids.ne(0).long(), features), rtol=0, atol=0)
    print('保存重载通过：普通完整state_dict即可推理，不需要梯度掩码。')


def check_real_model():
    set_seed(cfg.SEED)
    train, dev = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    dataset = GroupDataset(train, tokenizer, build_feature_tensor(train), cfg.MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    rows = []
    pair_count = 0
    for batch in loader:
        rows.extend(batch['source_rows'].tolist())
        pair_count += len(batch['pair_targets'])
    fisher_loader = make_fisher_loader(dataset)
    assert len(train) == 3042 and len(dev) == 660
    assert sorted(rows) == list(range(3042)) and pair_count == 1282 and len(loader) == 220
    assert len(fisher_loader.dataset) == 3042 and len(fisher_loader) == 191
    print('数据通过：正式每轮3042行/1282对/220步；Fisher另用Train3042行/191批。')

    # 真模型只检查2行的Fisher前后向及筛选，完整准备由用户训练入口执行。
    model = load_model(tokenizer).eval()
    assert not hasattr(model, 'nli_pooler')
    sample = TensorDataset(
        dataset.encoded['input_ids'][:2], dataset.encoded['attention_mask'][:2],
        dataset.features[:2], dataset.labels[:2],
    )
    probe = DataLoader(sample, batch_size=2, generator=torch.Generator().manual_seed(cfg.FISHER_SEED))
    state_before = torch.get_rng_state().clone()
    sample_name, sample_parameter = next((n, p) for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIX))
    sample_before = sample_parameter.detach().clone()
    masks, stats, _ = calculate_fisher_mask(
        model, probe, nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS)),
        torch.device('cpu'), cfg.RESERVE_P, cfg.FISHER_GRAD_CLIP, cfg.FISHER_SEED,
    )
    assert torch.equal(torch.get_rng_state(), state_before) and not model.training
    assert torch.equal(sample_parameter, sample_before)
    assert all(p.grad is None for p in model.parameters()) and sample_name in masks
    assert all(name.startswith(ENCODER_PREFIX) for name in masks)
    print('真实CPU检查通过；encoder参与筛选：', stats['eligible_coordinates'],
          '；保留：', stats['retained_coordinates'], '；比例：', stats['actual_retained_fraction'])
    with torch.inference_mode():
        logits = model(sample.tensors[0], sample.tensors[1], sample.tensors[2])
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()


if __name__ == '__main__':
    torch.set_num_threads(4)
    check_mask_logic()
    check_real_model()
    print('全部检查通过。未执行optimizer.step，没有启动正式训练，未写入模型目录。')
