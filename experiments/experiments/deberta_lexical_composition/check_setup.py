"""检查配置、mask、梯度和保存加载；不更新参数，--gpu可检查真实权重前后向。"""
import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_lexical_composition import config as cfg
from experiments.deberta_lexical_composition import train_deberta_composition as current
from experiments.deberta_lexical_composition.data import read_data, build_feature_tensor, DiCoDataset, PairDataset
from experiments.deberta_lexical_composition.model import DebertaDiCoModel, MaskedComposition
from experiments.deberta_lexical_composition.evaluation import get_pair_logits, evaluate_without_changing_rng
from experiments.deberta_lexical.model import DebertaDiCoModel as OriginalModel
from experiments.deberta_lexical import train_deberta_lexical as original
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES


class GradientRecorder:
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self):
        # 保留梯度作对照，不更新参数。
        self.gradients.append({
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in self.model.named_parameters()
        })


def check_data():
    reference = json.loads((cfg.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    fields = [
        'model_name', 'seed', 'epochs', 'row_batch_size', 'pair_batch_size', 'dev_batch_size',
        'max_length', 'learning_rate', 'lexical_learning_rate', 'weight_decay',
        'row_weights', 'pair_weights', 'pair_weight', 'pair_label_smoothing', 'lexical_hidden_size',
    ]
    for field in fields:
        assert getattr(cfg, field.upper()) == reference[field], field
    assert current.file_hash(cfg.TRAIN_PATH) == reference['train_sha256']
    assert current.file_hash(cfg.DEV_PATH) == reference['dev_sha256']
    train, dev = read_data()
    features = build_feature_tensor(train)
    assert len(train) == 3042 and len(dev) == 660 and features.shape == (3042, 20)
    assert len(PairDataset(train, None, features, cfg.MAX_LENGTH)) == 1282
    print('配置与历史0.930569一致；官方3042/660，1282组Pair，原20维特征，无增强。')
    return train, dev, features


def check_mask():
    layer = MaskedComposition()
    assert sum(p.numel() for p in layer.parameters()) == 98560
    features = torch.randn(2, 10, 128, requires_grad=True)
    mask = torch.tensor([
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    ], dtype=torch.bool)
    changed = features.detach().clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 10000
    values = layer(features, mask)
    torch.testing.assert_close(values, layer(changed, mask), rtol=0, atol=0)
    assert torch.count_nonzero(values[~mask]) == 0
    values.sum().backward()
    assert torch.count_nonzero(features.grad[~mask]) == 0
    extended = torch.cat([features.detach(), torch.randn(2, 5, 128)], dim=1)
    longer_mask = torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(values.detach(), layer(extended, longer_mask)[:, :10], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(layer(features.detach(), torch.zeros_like(mask))) == 0
    print('CNN通过：98560参数，其他文本及padding无串扰，延长padding不改变有效结果。')


def check_models(output_dir):
    tiny_config = DebertaV2Config(
        vocab_size=64, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    backbone = DebertaV2Model(tiny_config)
    before = torch.get_rng_state()
    old = OriginalModel(copy.deepcopy(backbone), tokenizer, len(FEATURE_NAMES))
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    control = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, len(FEATURE_NAMES), use_composition=False)
    assert torch.equal(torch.get_rng_state(), after)
    torch.set_rng_state(before)
    model = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, len(FEATURE_NAMES))
    assert torch.equal(torch.get_rng_state(), after)
    for name, value in old.state_dict().items():
        torch.testing.assert_close(control.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    ids = torch.tensor([[1, 4, 5, 2, 6, 7, 2, 0], [1, 6, 7, 2, 4, 5, 2, 0]])
    masks = (ids != 0).long()
    features = torch.randn(2, 20)
    rows = [(ids, masks, features, torch.tensor([0, 1]))]
    pairs = [(ids[:1], masks[:1], features[:1], ids[1:], masks[1:], features[1:], torch.tensor([0]))]
    row_ce, pair_ce = current.build_criteria(torch.device('cpu'))
    old.train()
    control.train()
    first, second = GradientRecorder(old), GradientRecorder(control)
    initial = {name: value.detach().clone() for name, value in control.named_parameters()}
    rng = torch.get_rng_state()
    old_losses = [
        original.train_rows(old, rows, first, row_ce, torch.device('cpu')),
        original.train_pairs(old, pairs, first, pair_ce, torch.device('cpu')),
    ]
    old_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    new_losses = [
        current.train_rows(control, rows, second, row_ce, torch.device('cpu')),
        current.train_pairs(control, pairs, second, pair_ce, torch.device('cpu')),
    ]
    assert old_losses == new_losses and torch.equal(torch.get_rng_state(), old_rng)
    for old_grad, new_grad in zip(first.gradients, second.gradients):
        for name in old_grad:
            if old_grad[name] is None:
                assert new_grad[name] is None
            else:
                torch.testing.assert_close(old_grad[name], new_grad[name], rtol=0, atol=0)
    assert all(torch.equal(p, initial[name]) for name, p in control.named_parameters())

    optimizer = current.build_optimizer(model)
    assert optimizer.param_groups[0]['lr'] == cfg.LEARNING_RATE
    assert optimizer.param_groups[1]['lr'] == cfg.LEXICAL_LEARNING_RATE
    groups = [{id(p) for p in group['params']} for group in optimizer.param_groups]
    assert not groups[0] & groups[1]
    assert groups[0] | groups[1] == {id(p) for p in model.parameters()}
    assert {id(p) for p in model.composition.parameters()} <= groups[0]
    assert sum(p.numel() for p in optimizer.param_groups[1]['params']) == 804
    assert not optimizer.state
    del optimizer
    print('无CNN对照通过：原参数、Row/Pair梯度及RNG一致；CNN初始化不推进原RNG。')

    # 只在微型检查模型上模拟输出头已学习的状态，检验CNN反向通路。
    # 正式模型仍保留原版输出层零初始化；本脚本没有optimizer.step。
    with torch.no_grad():
        model.match_output[-1].weight.normal_(0, 0.02)
    seen_masks = []
    hook = model.composition.register_forward_pre_hook(lambda module, inputs: seen_masks.append(inputs[1].clone()))
    model.train()
    logits = model(ids, masks, features)
    hook.remove()
    assert len(seen_masks) == 2
    assert torch.equal(seen_masks[0], torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool))
    assert torch.equal(seen_masks[1], torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]] * 2, dtype=torch.bool))
    loss = row_ce(logits, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.composition.parameters())
    model.zero_grad(set_to_none=True)
    first_logits = model(ids[:1], masks[:1], features[:1])
    second_logits = model(ids[1:], masks[1:], features[1:])
    loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(first_logits, second_logits), torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.composition.parameters())
    model.eval()
    with torch.inference_mode():
        expected = model(ids, masks, features)
    path = output_dir / 'tiny_state.pth'
    torch.save(model.state_dict(), path)
    restored = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, len(FEATURE_NAMES))
    restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, masks, features), expected, rtol=0, atol=0)
    print('CNN通过：A/B共享且mask分别正确，Row/Pair均能反传；完整保存重载logits一致。')


def check_scoring(dev, output_dir):
    history = pd.read_csv(cfg.REFERENCE_PREDICTIONS, keep_default_na=False)
    aligned = dev.set_index('instance_id').loc[history['instance_id']].reset_index()
    logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)

    class HistoricalModel(nn.Module):
        def forward(self, input_ids, attention_mask, lexical_features):
            return logits[input_ids[:, 0]]

    size = len(aligned)
    dataset = TensorDataset(
        torch.arange(size).unsqueeze(1), torch.ones(size, 1, dtype=torch.long),
        torch.zeros(size, 20), torch.zeros(size, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=cfg.DEV_BATCH_SIZE)
    model = HistoricalModel().train()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    summary = evaluate_without_changing_rng(model, loader, aligned, torch.device('cpu'), output_dir)
    assert model.training and random.getstate() == states[0]
    now = np.random.get_state()
    assert now[0] == states[1][0] and np.array_equal(now[1], states[1][1]) and now[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    scores = summary.set_index('method')['weighted_f1']
    assert abs(scores['independent'] - 0.46819069545516523) < 1e-12
    assert abs(scores['pair_aware'] - 0.9305693600964176) < 1e-12
    predictions = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
    assert predictions['prediction'].tolist() == history['prediction'].tolist()
    assert predictions['pair_prediction'].tolist() == history['pair_prediction'].tolist()
    singleton = aligned.groupby('pair_id')['pair_id'].transform('size').eq(1)
    assert predictions.loc[singleton, 'prediction'].equals(predictions.loc[singleton, 'pair_prediction'])
    print('官方评分通过：历史660行预测、0.930569结构分、单例原argmax和评估RNG完全一致。')


def check_gpu(train, features):
    device = torch.device('cuda')
    tokenizer = current.load_tokenizer()
    current.set_seed(cfg.SEED)
    model = current.build_model(tokenizer, device)
    model.train()
    row_ce, pair_ce = current.build_criteria(device)
    assert all(p.requires_grad for p in model.parameters())
    row_loader = DataLoader(DiCoDataset(train, tokenizer, features, cfg.MAX_LENGTH), batch_size=cfg.ROW_BATCH_SIZE)
    pair_loader = DataLoader(PairDataset(train, tokenizer, features, cfg.MAX_LENGTH), batch_size=cfg.PAIR_BATCH_SIZE)
    torch.cuda.reset_peak_memory_stats()
    ids, masks, values, labels = next(iter(row_loader))
    logits = model(ids.to(device), masks.to(device), values.to(device))
    loss = row_ce(logits, labels.to(device))
    loss.backward()
    assert logits.shape == (cfg.ROW_BATCH_SIZE, 4) and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    ids0, masks0, values0, ids1, masks1, values1, targets = next(iter(pair_loader))
    first = model(ids0.to(device), masks0.to(device), values0.to(device))
    second = model(ids1.to(device), masks1.to(device), values1.to(device))
    loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(first, second), targets.to(device))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('真实NLI主干+CNN，Row16行、Pair16组前后向通过。峰值GiB：',
          round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print('峰值不含Adam更新状态；未执行optimizer.step。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    current.set_seed(cfg.SEED)
    train, dev, features = check_data()
    check_mask()
    with tempfile.TemporaryDirectory(prefix='_check_', dir=cfg.EXPERIMENT_DIR) as temporary:
        output_dir = Path(temporary)
        check_models(output_dir)
        check_scoring(dev, output_dir / 'scoring')
    if args.gpu:
        check_gpu(train, features)
    print('全部检查完成；没有训练或更新参数。')
