"""随机微型模型的冻结检查，不下载权重，不使用 GPU。"""

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_joint_pair import config as reference_cfg
from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
from experiments.deberta_joint_pair.test_cpu import TinyTokenizer
from experiments.deberta_joint_pair_frozen import config as cfg
from experiments.deberta_joint_pair_frozen.train_deberta_frozen import freeze_backbone, train_epoch
from experiments.deberta_lexical.model import DebertaDiCoModel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # 对照中数据、优化超参数、选模规则全部不变。
    for name in [
        'SEED', 'MODEL_NAME', 'EPOCHS', 'MAX_LENGTH', 'LEARNING_RATE', 'WEIGHT_DECAY',
        'ROW_WEIGHTS', 'PAIR_WEIGHTS', 'PAIR_LABEL_SMOOTHING', 'PAIR_WEIGHT',
        'GROUP_BATCH_SIZE', 'DEV_BATCH_SIZE', 'SELECT_BY', 'TRAIN_PATH', 'DEV_PATH', 'GOLD_PATH',
    ]:
        assert getattr(cfg, name) == getattr(reference_cfg, name), name
    assert cfg.MODEL_DIR != reference_cfg.MODEL_DIR
    assert cfg.RESULTS_DIR != reference_cfg.RESULTS_DIR

    tokenizer = TinyTokenizer()
    backbone = DebertaV2Model(DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=8,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
        relative_attention=True, max_relative_positions=8,
        pos_att_type=['c2p', 'p2c'], norm_rel_ebd='layer_norm',
        conv_kernel_size=3,
    ))
    reference = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    model = freeze_backbone(copy.deepcopy(reference))
    assert reference.state_dict().keys() == model.state_dict().keys()

    for name, parameter in model.named_parameters():
        if name.startswith('deberta_model.embeddings.'):
            assert not parameter.requires_grad
        elif name.startswith('deberta_model.encoder.layer.'):
            index = int(name.split('.')[3])
            assert parameter.requires_grad == (index >= 6)
        elif name.startswith(('deberta_model.encoder.rel_embeddings.',
                              'deberta_model.encoder.LayerNorm.',
                              'deberta_model.encoder.conv.')):
            assert not parameter.requires_grad
        else:
            assert parameter.requires_grad

    frame = pd.DataFrame([
        ['f0', 'f', 'black dog', 'dog', 'FORWARD_ENTAILMENT'],
        ['n0', 'n', 'red cat', 'blue dog', 'NEGATIVE_OTHER'],
        ['f1', 'f', 'dog', 'black dog', 'BACKWARD_ENTAILMENT'],
        ['e0', 'e', 'a dog', 'the dog', 'EQUIVALENCE'],
        ['e1', 'e', 'the dog', 'a dog', 'EQUIVALENCE'],
    ], columns=['instance_id', 'pair_id', 'text1', 'text2', 'label'])
    dataset = GroupDataset(frame, tokenizer, max_length=8)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_groups)
    batch = next(iter(loader))

    reference.eval()
    model.eval()
    with torch.inference_mode():
        expected = reference(batch['input_ids'], batch['attention_mask'], None)
        actual = model(batch['input_ids'], batch['attention_mask'], None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01,
    )
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all((id(p) in optimized) == p.requires_grad for p in model.parameters())
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    result = train_epoch(model, loader, optimizer, row_ce, pair_ce, torch.device('cpu'))
    assert result['train_rows'] == 5 and result['train_pairs'] == 2

    changed = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
            assert torch.equal(before[name], parameter.detach()), name
        elif not torch.equal(before[name], parameter.detach()):
            changed.append(name)
    assert any(name.startswith('deberta_model.encoder.layer.6.') for name in changed)
    assert any(name.startswith('deberta_model.encoder.layer.7.') for name in changed)
    assert 'output.weight' in changed
    assert 'match_output.2.weight' in changed
    print('PASS: same recipe, initial predictions equal, frozen weights unchanged, upper layers and heads updated')


if __name__ == '__main__':
    main()
