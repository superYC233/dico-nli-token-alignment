"""使用随机微型 DeBERTa 验证；不加载真实大权重，不运行 GPU。"""
import ast
import copy
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_lexical import config as cfg
from experiments.deberta_lexical.check_setup import main as check_setup
from experiments.deberta_lexical.data import DiCoDataset, PairDataset, build_feature_tensor, read_data
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES, build_lexical_features
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.evaluation import pair_aware, score_logits, predict_logits, compare_with_original
from experiments.deberta_lexical.train_deberta_lexical import (
    build_optimizer, train_rows, train_pairs, evaluate_without_changing_rng,
)


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, first, second, **kwargs):
        a = 3 + sum(map(ord, first)) % 25
        b = 3 + sum(map(ord, second)) % 25
        return {
            'input_ids': torch.tensor([[1, a, 2, b, 2, 0, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }


def test_features():
    first = build_lexical_features('two black cars', 'a car without wheels')
    second = build_lexical_features('a car without wheels', 'two black cars')
    direction_pairs = [(0, 1), (3, 4), (5, 6), (7, 8), (10, 11), (12, 13), (16, 17)]
    for i, j in direction_pairs:
        assert first[i] == second[j] and first[j] == second[i]
    assert first[9] == -second[9]
    for index in [2, 14, 15, 18, 19]:
        assert first[index] == second[index]
    numbers = dict(zip(FEATURE_NAMES, build_lexical_features('two cats', '2 cats')))
    assert numbers['number_jaccard'] == 1.0
    assert numbers['different_nonempty_number_sets'] == 0.0
    numbers = dict(zip(FEATURE_NAMES, build_lexical_features('3.0 cats', '3 cats')))
    assert numbers['number_jaccard'] == 1.0
    negatives = dict(zip(FEATURE_NAMES, build_lexical_features("can't go", 'can go')))
    assert negatives['has_negation_1'] == 1.0 and negatives['negation_presence_differs'] == 1.0
    assert np.isfinite(build_lexical_features('', '')).all()
    assert build_lexical_features('DOG', 'the dog') == build_lexical_features('dog', 'the dog')
    print('词汇特征：方向互换、数字、否定、空文本检查通过。')


def original_model_class(backbone, tokenizer):
    # 只编译旧文件里的模型类，不执行顶层代码，不触发真实权重加载。
    tree = ast.parse((PROJECT_ROOT / 'experiments/deberta_original/train.py').read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'debertaDiCoModel')
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'my_model': backbone, 'my_tokenizer': tokenizer}
    exec(compile(module, 'original_model_only', 'exec'), namespace)
    return namespace['debertaDiCoModel']


def main():
    torch.set_num_threads(2)
    check_setup()
    test_features()
    torch.manual_seed(42)
    tokenizer = TinyTokenizer()
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    backbone = DebertaV2Model(tiny_config)
    original_class = original_model_class(copy.deepcopy(backbone), tokenizer)
    rng = torch.get_rng_state()
    old_model = original_class()
    torch.set_rng_state(rng)
    model = DebertaDiCoModel(backbone, tokenizer, len(FEATURE_NAMES))
    # 新增层的初始化没有推进原随机序列。
    after_lexical_init = torch.get_rng_state()
    torch.set_rng_state(rng)
    control = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, len(FEATURE_NAMES), use_lexical_features=False)
    assert torch.equal(torch.get_rng_state(), after_lexical_init)
    control.load_state_dict(old_model.state_dict(), strict=True)
    for name, value in old_model.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    batch = tokenizer('a black car', 'a car')
    features = torch.tensor([build_lexical_features('a black car', 'a car')])
    old_model.eval()
    model.eval()
    torch.testing.assert_close(old_model(**batch), model(**batch, lexical_features=features), rtol=0, atol=0)
    assert sum(p.numel() for p in model.lexical_output.parameters()) == 804

    old_model.train()
    model.train()
    rng = torch.get_rng_state()
    old_logits = old_model(**batch)
    old_logits.sum().backward()
    torch.set_rng_state(rng)
    new_logits = model(**batch, lexical_features=features)
    new_logits.sum().backward()
    torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
    new_parameters = dict(model.named_parameters())
    for name, parameter in old_model.named_parameters():
        if parameter.grad is None:
            assert new_parameters[name].grad is None
        else:
            torch.testing.assert_close(parameter.grad, new_parameters[name].grad, rtol=0, atol=0)
    print('原模型与新增零初始化分支：初始参数、预测和原有参数梯度一致。')

    tiny_frame = pd.DataFrame([
        {'instance_id': 'p0', 'pair_id': 'p', 'text1': 'a black car', 'text2': 'a car', 'label': cfg.FULL_LABELS[0]},
        {'instance_id': 'p1', 'pair_id': 'p', 'text1': 'a car', 'text2': 'a black car', 'label': cfg.FULL_LABELS[1]},
        {'instance_id': 'n0', 'pair_id': 'n', 'text1': 'cat', 'text2': 'dog', 'label': cfg.FULL_LABELS[3]},
    ])
    tiny_features = build_feature_tensor(tiny_frame)
    row_loader = DataLoader(DiCoDataset(tiny_frame, tokenizer, tiny_features), batch_size=3, shuffle=True)
    pair_data = PairDataset(tiny_frame, tokenizer, tiny_features)
    pair_loader = DataLoader(pair_data, batch_size=1, shuffle=True)
    item = pair_data[0]
    assert torch.equal(item[2], tiny_features[0]) and torch.equal(item[5], tiny_features[1])
    assert not torch.equal(item[2], item[5])

    optimizer = build_optimizer(model)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['lr'] == 2e-5
    assert optimizer.param_groups[1]['lr'] == 1e-3
    original_ids = {id(p) for name, p in model.named_parameters() if not name.startswith('lexical_output.')}
    lexical_ids = {id(p) for p in model.lexical_output.parameters()}
    assert {id(p) for p in optimizer.param_groups[0]['params']} == original_ids
    assert {id(p) for p in optimizer.param_groups[1]['params']} == lexical_ids
    all_optimizer_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(all_optimizer_ids) == len(set(all_optimizer_ids))
    baseline_optimizer = build_optimizer(control)
    assert len(baseline_optimizer.param_groups) == 1
    assert baseline_optimizer.param_groups[0]['lr'] == 2e-5
    print('优化器分组检查通过：原模型 2e-5，词汇分支 1e-3，无遗漏或重复参数。')
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    device = torch.device('cpu')
    row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
    assert torch.count_nonzero(model.lexical_output[-1].weight).item() > 0
    pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
    assert np.isfinite(row_loss) and np.isfinite(pair_loss)
    assert model.lexical_output[0].weight.grad.abs().sum().item() > 0
    print('Row CE / Pair CE 反向更新通过，词汇分支两层均能学习。')

    train, dev = read_data()
    train_features = build_feature_tensor(train)
    real_pairs = PairDataset(train, tokenizer, train_features)
    assert len(real_pairs) == 1282
    for index0, index1, target in real_pairs.pairs:
        assert train.iloc[index0]['pair_id'] == train.iloc[index1]['pair_id']
        assert target in (0, 1, 2)

    # 完整 660 条 Dev 只用微型随机模型验证接口，不代表实际成绩。
    dev_features = build_feature_tensor(dev)
    dev_loader = DataLoader(DiCoDataset(dev, tokenizer, dev_features), batch_size=16, shuffle=False)
    output_dir = cfg.EXPERIMENT_DIR / 'outputs' / '_cpu_check'
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    evaluate_without_changing_rng(model, dev_loader, dev, device, output_dir / 'tiny_dev')
    assert random.getstate() == states[0]
    numpy_state = np.random.get_state()
    assert np.array_equal(numpy_state[1], states[1][1]) and numpy_state[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2]) and model.training
    saved = output_dir / 'tiny_state.pth'
    torch.save(model.state_dict(), saved)
    reloaded = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, len(FEATURE_NAMES))
    reloaded.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
    logits_before = predict_logits(model, dev_loader, device)
    logits_after = predict_logits(reloaded, dev_loader, device)
    torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)
    saved.unlink()
    print('评估随机状态恢复、完整权重保存加载、重载后 logits 一致检查通过。')

    history_path = cfg.PROJECT_ROOT / 'experiments/qwen_lora/outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
    if history_path.exists():
        history = pd.read_csv(history_path, keep_default_na=False)
        aligned = dev.set_index('instance_id').loc[history['instance_id']].reset_index()
        logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)
        prediction = pair_aware(logits, aligned)
        assert [cfg.LABELS[i] for i in prediction.tolist()] == history['pair_prediction'].tolist()
        report = score_logits(logits, aligned, output_dir / 'historical_qwen')
        score = float(report.loc[report['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
        assert abs(score - 0.8991367586828434) < 1e-12
        compare_with_original(
            output_dir / 'historical_qwen/predictions_with_scores.csv', output_dir / 'comparison_check',
        )
        print('历史预测逐条一致、官方评分与错误对照文件通过。')
    print('所有 CPU 检查通过；未加载真实大模型权重，未进行 GPU 训练。')


if __name__ == '__main__':
    main()
