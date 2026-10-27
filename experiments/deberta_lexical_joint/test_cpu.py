"""CPU 微型检查，不加载真实预训练权重，不运行 GPU 训练。"""
import ast
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_lexical_joint import config as cfg
from experiments.deberta_lexical_joint.data import (
    read_data, build_feature_tensor, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical_joint.model import DebertaDiCoModel
from experiments.deberta_lexical_joint.evaluation import get_pair_logits, score_logits
from experiments.deberta_lexical_joint.analyze_stages import (
    compare_with_reference, compare_best_with_reference,
)
from experiments.deberta_lexical_joint.train_deberta_joint import (
    set_seed, build_optimizer, build_auxiliary_loader, train_rows, train_pairs,
    save_best_weights, make_run_config,
)
from experiments.deberta_lexical_stage_probe import train_deberta_stage_probe as original


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


class TinyFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(20, 4, dtype=torch.float64)

    def forward(self, input_ids, attention_mask, features):
        return self.output(features)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def test_unchanged_source():
    old_dir = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe'
    for name in ['model.py', 'lexical_features.py', 'data.py', 'evaluation.py']:
        before = (old_dir / name).read_text(encoding='utf-8-sig')
        after = (cfg.EXPERIMENT_DIR / name).read_text(encoding='utf-8-sig')
        after = after.replace('experiments.deberta_lexical_joint', 'experiments.deberta_lexical_stage_probe')
        assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)), name
    old_tree = ast.parse((old_dir / 'train_deberta_stage_probe.py').read_text(encoding='utf-8-sig'))
    new_tree = ast.parse((cfg.EXPERIMENT_DIR / 'train_deberta_joint.py').read_text(encoding='utf-8-sig'))
    for name in ['set_seed', 'build_optimizer', 'train_rows', 'evaluate_without_changing_rng']:
        before = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        after = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(before) == ast.dump(after), name
    saved = json.loads((cfg.REFERENCE_RESULTS_DIR / 'run_config.json').read_text(encoding='utf-8'))
    for name, key in [
        ('SEED', 'seed'), ('MODEL_NAME', 'model_name'), ('LEXICAL_HIDDEN_SIZE', 'lexical_hidden_size'),
        ('LEXICAL_LEARNING_RATE', 'lexical_learning_rate'), ('LEARNING_RATE', 'learning_rate'),
        ('EPOCHS', 'epochs'), ('ROW_BATCH_SIZE', 'row_batch_size'), ('PAIR_BATCH_SIZE', 'pair_batch_size'),
        ('DEV_BATCH_SIZE', 'dev_batch_size'), ('MAX_LENGTH', 'max_length'),
        ('WEIGHT_DECAY', 'weight_decay'), ('ROW_WEIGHTS', 'row_weights'),
        ('PAIR_WEIGHTS', 'pair_weights'), ('PAIR_WEIGHT', 'pair_weight'),
        ('PAIR_LABEL_SMOOTHING', 'pair_label_smoothing'),
    ]:
        assert getattr(cfg, name) == saved[key], name
    print('原模型、特征、Row 阶段、优化器和原参数一致。')


def test_auxiliary_sampling():
    train, _ = read_data()
    features = build_feature_tensor(train)
    tokenizer = TinyTokenizer()
    state = torch.get_rng_state().clone()
    loader = build_auxiliary_loader(train, tokenizer, features)
    assert torch.equal(state, torch.get_rng_state())
    assert len(loader.dataset) == 3042
    assert loader.dataset.frame['label'].value_counts().to_dict() == train['label'].value_counts().to_dict()
    iterator = iter(loader)
    assert torch.equal(state, torch.get_rng_state())
    counts = torch.zeros(4, dtype=torch.long)
    first_batches = []
    for _ in range(81):
        batch = next(iterator)
        counts += torch.bincount(batch[-1], minlength=4)
        first_batches.append(batch[-1].clone())
    assert counts.sum().item() == 1296 and (counts > 0).all()
    assert torch.equal(state, torch.get_rng_state())
    second = iter(build_auxiliary_loader(train, tokenizer, features))
    for labels in first_batches:
        assert torch.equal(labels, next(second)[-1])
    with patch.object(cfg, 'PAIR_ROW_WEIGHT', 0.0):
        assert build_auxiliary_loader(train, tokenizer, features) is None
    print('辅助样本只来自完整 Train，含 N；独立抽样可复现，不消耗全局 RNG。')
    print('81 个辅助批次的 F/B/E/N 数量：', counts.tolist())


def test_zero_weight():
    tokenizer = TinyTokenizer()
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=128,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    set_seed(42)
    initial = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, feature_size=20)
    train, _ = read_data()
    train = pd.concat([train.iloc[:4], train[train['label'] == cfg.FULL_LABELS[3]].iloc[:1]], ignore_index=True)
    features = build_feature_tensor(train)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    outcomes = []
    for use_new_function in [False, True]:
        set_seed(123)
        model = copy.deepcopy(initial)
        optimizer = build_optimizer(model)
        rows = DataLoader(DiCoDataset(train, tokenizer, features), batch_size=3, shuffle=True)
        pairs = DataLoader(PairDataset(train, tokenizer, features), batch_size=2, shuffle=True)
        losses = []
        for _ in range(2):
            model.train()
            row_loss = train_rows(model, rows, optimizer, row_criterion, torch.device('cpu'))
            if use_new_function:
                with patch.object(cfg, 'PAIR_ROW_WEIGHT', 0.0):
                    result = train_pairs(model, pairs, optimizer, pair_criterion, torch.device('cpu'))
                pair_loss = result['pair_loss']
                assert result['aux_rows'] == result['aux_batches'] == result['aux_row_loss'] == 0
            else:
                pair_loss = original.train_pairs(model, pairs, optimizer, pair_criterion, torch.device('cpu'))
            losses.append((row_loss, pair_loss))
        outcomes.append((copy.deepcopy(model.state_dict()), optimizer.state_dict(), losses))
    assert outcomes[0][2] == outcomes[1][2]
    for name, tensor in outcomes[0][0].items():
        torch.testing.assert_close(tensor, outcomes[1][0][name], rtol=0, atol=0)
    for index, state in outcomes[0][1]['state'].items():
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, outcomes[1][1]['state'][index][name], rtol=0, atol=0)
    print('辅助权重为 0：两轮训练的 loss、全部参数与 Adam 状态完全退化为旧版。')


def test_joint_gradient():
    set_seed(7)
    model = TinyFeatureModel()
    manual = copy.deepcopy(model)
    ids = torch.zeros(2, 4, dtype=torch.long)
    masks = torch.ones_like(ids)
    first_features = torch.linspace(-0.8, 0.7, 40, dtype=torch.float64).reshape(2, 20)
    second_features = first_features.flip(1)
    targets = torch.tensor([0, 2])
    pair_batch = (ids, masks, first_features, ids, masks, second_features, targets)
    row_ids = torch.zeros(3, 4, dtype=torch.long)
    row_masks = torch.ones_like(row_ids)
    row_features = torch.linspace(-1.0, 1.0, 60, dtype=torch.float64).reshape(3, 20)
    labels = torch.tensor([0, 2, 3])
    row_batch = (row_ids, row_masks, row_features, labels)
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float64),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64))
    optimizer = CountingSGD(model.parameters())
    manual_optimizer = CountingSGD(manual.parameters())
    manual_optimizer.zero_grad()
    expected_pair = pair_criterion(get_pair_logits(
        manual(ids, masks, first_features), manual(ids, masks, second_features),
    ), targets)
    expected_row = row_criterion(manual(row_ids, row_masks, row_features), labels)
    total = cfg.PAIR_WEIGHT * expected_pair + cfg.PAIR_ROW_WEIGHT * expected_row
    total.backward()
    manual_optimizer.step()
    result = train_pairs(
        model, [pair_batch], optimizer, pair_criterion, torch.device('cpu'), [row_batch], row_criterion,
    )
    assert optimizer.steps == 1
    assert result['aux_batches'] == 1 and result['aux_rows'] == 3
    assert result['aux_label_counts'] == [1, 0, 1, 1]
    assert abs(result['total_loss'] - total.item()) < 1e-12
    for first, second in zip(model.parameters(), manual.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    # 辅助 loader 耗尽后需要继续取；每个 pair batch 仍只 step 一次。
    optimizer.steps = 0
    result = train_pairs(
        model, [pair_batch, pair_batch], optimizer, pair_criterion,
        torch.device('cpu'), [row_batch], row_criterion,
    )
    assert optimizer.steps == 2 and result['aux_rows'] == 6
    assert result['aux_label_counts'] == [2, 0, 2, 2]
    print('顺序反传与损失之和的梯度一致；每批只 step 一次，辅助 loader 循环正确。')


def test_checkpoint(output_dir):
    folder = output_dir / 'checkpoint'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'best_model.pth'
    model = TinyFeatureModel()
    save_best_weights(model, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    old_bytes = path.read_bytes()
    def fail_save(state, temporary):
        Path(temporary).write_bytes(b'failed write')
        raise OSError('模拟磁盘写入失败')
    with patch('experiments.deberta_lexical_joint.train_deberta_joint.torch.save', side_effect=fail_save):
        try:
            save_best_weights(model, path)
        except OSError:
            pass
        else:
            raise AssertionError('应当报告写盘失败')
    assert path.read_bytes() == old_bytes
    assert not path.with_name(path.name + '.tmp').exists()
    print('权重严格重载通过；模拟保存失败时旧最佳权重仍完整，临时文件已清理。')


def main():
    torch.set_num_threads(2)
    output_dir = cfg.EXPERIMENT_DIR / 'outputs/_cpu_check'
    output_dir.mkdir(parents=True, exist_ok=True)
    test_unchanged_source()
    test_auxiliary_sampling()
    test_zero_weight()
    test_joint_gradient()
    test_checkpoint(output_dir)
    # 用历史 CSV 验证解码/评分/比较，不对真实模型重新推理。
    _, dev = read_data()
    history = pd.read_csv(cfg.REFERENCE_RESULTS_DIR / 'epoch_3/predictions_with_scores.csv', keep_default_na=False)
    dev = dev.set_index('instance_id').loc[history['instance_id']].reset_index()
    logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].to_numpy(), dtype=torch.float32)
    summary = score_logits(logits, dev, output_dir / 'epoch_3/after_pair')
    assert abs(float(summary.set_index('method').loc['pair_aware', 'weighted_f1']) - 0.9305693600964176) < 1e-12
    run_config = make_run_config(True, 3042, 660, 1282)
    (output_dir / 'run_config.json').write_text(json.dumps(run_config), encoding='utf-8')
    compare_with_reference(output_dir)
    protocol = json.loads((output_dir / 'reference_protocol_check.json').read_text(encoding='utf-8'))
    assert not protocol['same_protocol'] and 'pair_row_weight' in protocol['different_keys']
    compare_best_with_reference(output_dir, 3)
    changes = pd.read_csv(output_dir / 'comparison_with_lexical/summary.csv')
    assert (changes['fixed'] == 0).all() and (changes['broken'] == 0).all()
    print('历史评分、最佳模型对比和新损失的协议标记检查通过。')
    print('CPU 检查全部通过；未加载真实预训练权重，未运行 GPU 训练。')


if __name__ == '__main__':
    main()
