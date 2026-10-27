"""核对配置、梯度及保存加载；不更新参数。--gpu检查真实模型前后向。"""
import argparse
import copy
import io
import json
import random
import sys
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

from experiments.deberta_feature_fusion import config as cfg
from experiments.deberta_feature_fusion import train_deberta_feature_fusion as current
from experiments.deberta_feature_fusion.data import read_data, build_feature_tensor, DiCoDataset, PairDataset
from experiments.deberta_feature_fusion.model import DebertaDiCoModel
from experiments.deberta_feature_fusion.evaluation import get_pair_logits, evaluate_without_changing_rng
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
        # 只记录梯度，不更新参数。
        self.gradients.append({
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in self.model.named_parameters()
        })


def check_data():
    reference = json.loads((cfg.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    for field in [
        'model_name', 'seed', 'epochs', 'row_batch_size', 'pair_batch_size', 'dev_batch_size',
        'max_length', 'learning_rate', 'lexical_learning_rate', 'weight_decay',
        'row_weights', 'pair_weights', 'pair_weight', 'pair_label_smoothing', 'lexical_hidden_size',
    ]:
        assert getattr(cfg, field.upper()) == reference[field], field
    assert current.file_hash(cfg.TRAIN_PATH) == reference['train_sha256']
    assert current.file_hash(cfg.DEV_PATH) == reference['dev_sha256']
    feature_file = PROJECT_ROOT / 'experiments/deberta_lexical/lexical_features.py'
    assert current.file_hash(feature_file) == reference['feature_source_sha256']
    train, dev = read_data()
    features = build_feature_tensor(train)
    assert len(train) == 3042 and len(dev) == 660 and features.shape == (3042, 20)
    assert len(PairDataset(train, None, features, cfg.MAX_LENGTH)) == 1282
    assert train.label.value_counts().to_dict() == {
        'FORWARD_ENTAILMENT': 881, 'BACKWARD_ENTAILMENT': 881,
        'EQUIVALENCE': 802, 'NEGATIVE_OTHER': 478,
    }
    print('数据/训练参数通过：官方3042/660、1282组Pair、原20维特征；没有增强。')
    return train, dev, features


def check_alignment():
    tokens = torch.randn(2, 7, 128)
    scores = torch.randn(2, 7, 7)
    mask = torch.tensor([[0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)
    old = OriginalModel._align(None, tokens, scores, mask)
    new = DebertaDiCoModel._align(None, tokens, scores, mask)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    changed = tokens.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    torch.testing.assert_close(new, DebertaDiCoModel._align(None, changed, scores, mask), rtol=0, atol=0)
    values = torch.randn(2, 7, 128)
    torch.testing.assert_close(
        DebertaDiCoModel._pool(None, values, mask), OriginalModel._pool(None, values, mask), rtol=0, atol=0,
    )
    empty = torch.zeros_like(mask)
    assert torch.count_nonzero(DebertaDiCoModel._pool(None, values, empty)) == 0
    print('对齐公式通过：soft alignment和mean/max pooling与原版相同，目标mask有效。')


def assert_gradients(model):
    parameters = dict(model.named_parameters())
    for name in [
        'deberta_model.embeddings.word_embeddings.weight',
        'align_projection.weight', 'compare_layer.0.weight',
        'cls_norm.weight', 'match_norm.weight', 'lexical_encoder.0.weight',
        'fusion_output.0.weight', 'fusion_output.3.weight',
    ]:
        gradient = parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def check_model():
    tiny_config = DebertaV2Config(
        vocab_size=64, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    model = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, len(FEATURE_NAMES))
    assert model.fusion_input_size == 16 + 512 + 32
    assert len([module for module in model.modules() if isinstance(module, nn.Linear) and module.out_features == 4]) == 1
    assert not hasattr(model, 'output') and not hasattr(model, 'match_output') and not hasattr(model, 'lexical_output')
    assert not any(isinstance(module, nn.Conv1d) for module in model.modules())
    optimizer = current.build_optimizer(model)
    groups = [{id(p) for p in group['params']} for group in optimizer.param_groups]
    assert not groups[0] & groups[1]
    assert groups[0] | groups[1] == {id(p) for p in model.parameters()}
    assert groups[1] == {id(p) for p in model.lexical_encoder.parameters()}
    assert optimizer.param_groups[0]['lr'] == cfg.LEARNING_RATE
    assert optimizer.param_groups[1]['lr'] == cfg.LEXICAL_LEARNING_RATE
    assert sum(p.numel() for p in model.lexical_encoder.parameters()) == 736
    assert not optimizer.state
    del optimizer

    ids = torch.tensor([[1, 4, 5, 2, 6, 7, 2, 0], [1, 6, 7, 2, 4, 5, 2, 0]])
    masks = (ids != 0).long()
    features = torch.randn(2, 20)
    model.eval()
    captured = {}
    handles = []
    def capture(name):
        def hook(module, inputs, value):
            captured[name] = value
        return hook
    for name, module in [('cls', model.cls_norm), ('match', model.match_norm), ('lexical', model.lexical_encoder)]:
        handles.append(module.register_forward_hook(capture(name)))
    handles.append(model.fusion_output[0].register_forward_pre_hook(
        lambda module, inputs: captured.update(fused=inputs[0])
    ))
    with torch.inference_mode():
        logits = model(ids, masks, features)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        captured['fused'], torch.cat([captured['cls'], captured['match'], captured['lexical']], dim=-1),
        rtol=0, atol=0,
    )
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
    changed_ids = ids.clone()
    changed_ids[masks == 0] = 25
    with torch.inference_mode():
        torch.testing.assert_close(model(changed_ids, masks, features), logits, rtol=1e-5, atol=1e-6)

    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    row_ce, pair_ce = current.build_criteria(torch.device('cpu'))
    model.train()
    row_loss = row_ce(model(ids, masks, features), torch.tensor([0, 1]))
    row_loss.backward()
    assert_gradients(model)
    model.zero_grad(set_to_none=True)
    first = model(ids[:1], masks[:1], features[:1])
    second = model(ids[1:], masks[1:], features[1:])
    pair_loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(first, second), torch.tensor([0]))
    pair_loss.backward()
    assert_gradients(model)
    assert all(torch.equal(p, initial[name]) for name, p in model.named_parameters())
    print('融合/梯度通过：只有一个四分类头，三个特征分支首批Row/Pair均能反传；未更新参数。')

    old = copy.deepcopy(model)
    first, second = GradientRecorder(old), GradientRecorder(model)
    rows = [(ids, masks, features, torch.tensor([0, 1]))]
    pairs = [(ids[:1], masks[:1], features[:1], ids[1:], masks[1:], features[1:], torch.tensor([0]))]
    rng = torch.get_rng_state()
    old_losses = [original.train_rows(old, rows, first, row_ce, torch.device('cpu')),
                  original.train_pairs(old, pairs, first, pair_ce, torch.device('cpu'))]
    old_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    new_losses = [current.train_rows(model, rows, second, row_ce, torch.device('cpu')),
                  current.train_pairs(model, pairs, second, pair_ce, torch.device('cpu'))]
    assert old_losses == new_losses and torch.equal(torch.get_rng_state(), old_rng)
    for old_grad, new_grad in zip(first.gradients, second.gradients):
        for name in old_grad:
            if old_grad[name] is None:
                assert new_grad[name] is None
            else:
                torch.testing.assert_close(old_grad[name], new_grad[name], rtol=0, atol=0)
    assert all(torch.equal(p, initial[name]) for name, p in model.named_parameters())
    run_config = current.make_run_config(model, 3042, 660, 1282)
    assert run_config['fusion_input_size'] == 560 and not run_config['data_augmentation']
    assert not run_config['use_cnn'] and not run_config['use_lora']
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, len(FEATURE_NAMES))
    restored.load_state_dict(torch.load(stream, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, masks, features), model(ids, masks, features), rtol=0, atol=0)
    print('训练循环/重载通过：原Row/Pair梯度与损失权重保留，完整state重载后logits一致。')


def check_scoring(dev):
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
    loader = DataLoader(dataset, batch_size=16)
    model = HistoricalModel().train()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    output_dir = cfg.EXPERIMENT_DIR / 'outputs/_checks/historical_scoring'
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
    print('评分通过：历史660行独立/结构结果完全复现；单例不改答案，评估不推进训练RNG。')


def check_gpu(train, features):
    tokenizer = current.load_tokenizer()
    current.set_seed(cfg.SEED)
    model = current.build_model(tokenizer, torch.device('cuda')).train()
    assert all(p.requires_grad for p in model.parameters())
    assert model.fusion_input_size == 1312
    row_ce, pair_ce = current.build_criteria(torch.device('cuda'))
    row_loader = DataLoader(DiCoDataset(train, tokenizer, features, cfg.MAX_LENGTH), batch_size=cfg.ROW_BATCH_SIZE)
    pair_loader = DataLoader(PairDataset(train, tokenizer, features, cfg.MAX_LENGTH), batch_size=cfg.PAIR_BATCH_SIZE)
    torch.cuda.reset_peak_memory_stats()
    ids, masks, values, labels = next(iter(row_loader))
    loss = row_ce(model(ids.cuda(), masks.cuda(), values.cuda()), labels.cuda())
    loss.backward()
    assert torch.isfinite(loss)
    assert_gradients(model)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    ids0, masks0, values0, ids1, masks1, values1, targets = next(iter(pair_loader))
    first = model(ids0.cuda(), masks0.cuda(), values0.cuda())
    second = model(ids1.cuda(), masks1.cuda(), values1.cuda())
    loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(first, second), targets.cuda())
    loss.backward()
    assert torch.isfinite(loss)
    assert_gradients(model)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('GPU通过：真实NLI主干，1312维融合，Row16行/Pair16组前后向及各分支梯度正常。')
    print('峰值GiB：', round(torch.cuda.max_memory_allocated() / 1024**3, 3), '；不含Adam状态。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    current.set_seed(cfg.SEED)
    train, dev, features = check_data()
    check_alignment()
    check_model()
    check_scoring(dev)
    if args.gpu:
        check_gpu(train, features)
    print('全部检查完成；没有启动训练或执行参数更新。')
