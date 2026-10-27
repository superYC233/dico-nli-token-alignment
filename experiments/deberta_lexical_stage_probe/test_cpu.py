"""CPU 微型模型检查：新增评估不能改变后续训练；不加载真实大权重。"""
import ast
import copy
import json
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
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_lexical_stage_probe import config as cfg
from experiments.deberta_lexical_stage_probe.data import (
    read_data, build_feature_tensor, DiCoDataset, PairDataset,
)
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.evaluation import score_logits, pair_aware
from experiments.deberta_lexical_stage_probe.analyze_stages import (
    compare_stages, print_stage_table, compare_with_reference, error_counts,
)
from experiments.deberta_lexical_stage_probe.train_deberta_stage_probe import (
    set_seed, build_optimizer, train_rows, train_pairs, evaluate_without_changing_rng,
)
from experiments.deberta_lexical import train_deberta_lexical as original


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


def test_unchanged_source():
    old_dir = PROJECT_ROOT / 'experiments/deberta_lexical'
    for name in ['model.py', 'lexical_features.py', 'data.py', 'evaluation.py']:
        old = (old_dir / name).read_text(encoding='utf-8-sig')
        new = (cfg.EXPERIMENT_DIR / name).read_text(encoding='utf-8-sig')
        new = new.replace('experiments.deberta_lexical_stage_probe', 'experiments.deberta_lexical')
        assert ast.dump(ast.parse(new)) == ast.dump(ast.parse(old)), name
    for name in ['build_optimizer', 'train_rows', 'train_pairs', 'evaluate_without_changing_rng']:
        new_tree = ast.parse((cfg.EXPERIMENT_DIR / 'train_deberta_stage_probe.py').read_text(encoding='utf-8-sig'))
        old_tree = ast.parse((old_dir / 'train_deberta_lexical.py').read_text(encoding='utf-8-sig'))
        new_node = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        old_node = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(new_node) == ast.dump(old_node), name
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
    print('模型、数据、特征、损失、优化器与旧版代码一致；参数匹配 0.930569 运行。')


def assert_rng_equal(before):
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0]
    assert np.array_equal(after_numpy[1], before[1][1])
    assert after_numpy[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def test_training_trajectory(output_dir):
    tokenizer = TinyTokenizer()
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=128,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    set_seed(42)
    initial = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, feature_size=20)
    train, dev = read_data()
    # 保留两个真实可逆组和一个 N 单例，仅用于微型更新测试。
    train = pd.concat([train.iloc[:4], train[train['label'] == cfg.FULL_LABELS[3]].iloc[:1]], ignore_index=True)
    features = build_feature_tensor(train)
    dev_features = build_feature_tensor(dev)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    outcomes = []
    for add_middle_evaluation in [False, True]:
        model = copy.deepcopy(initial)
        set_seed(123)
        optimizer = build_optimizer(model)
        row_loader = DataLoader(DiCoDataset(train, tokenizer, features), batch_size=3, shuffle=True)
        pair_loader = DataLoader(PairDataset(train, tokenizer, features), batch_size=2, shuffle=True)
        dev_loader = DataLoader(DiCoDataset(dev, tokenizer, dev_features), batch_size=16, shuffle=False)
        device = torch.device('cpu')
        losses = []
        for epoch in range(1, 3):
            model.train()
            # 对照流程使用原版函数；诊断流程使用新文件的函数。
            functions = (train_rows, train_pairs) if add_middle_evaluation else (original.train_rows, original.train_pairs)
            row_loss = functions[0](model, row_loader, optimizer, row_criterion, device)
            if add_middle_evaluation:
                states = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                evaluate_without_changing_rng(
                    model, dev_loader, dev, device, output_dir / f'probe_{epoch}/after_row',
                )
                assert model.training
                assert_rng_equal(states)
            pair_loss = functions[1](model, pair_loader, optimizer, pair_criterion, device)
            evaluate_without_changing_rng(
                model, dev_loader, dev, device,
                output_dir / f'{"probe" if add_middle_evaluation else "control"}_{epoch}/after_pair',
            )
            assert model.training
            losses.append((row_loss, pair_loss))
        outcomes.append((copy.deepcopy(model.state_dict()), optimizer.state_dict(), losses))
    assert outcomes[0][2] == outcomes[1][2]
    for name, value in outcomes[0][0].items():
        torch.testing.assert_close(value, outcomes[1][0][name], rtol=0, atol=0)
    for index, state in outcomes[0][1]['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, outcomes[1][1]['state'][index][name], rtol=0, atol=0)
    assert len(outcomes[1][1]['param_groups']) == 2
    assert outcomes[1][1]['param_groups'][0]['lr'] == 2e-5
    assert outcomes[1][1]['param_groups'][1]['lr'] == 1e-3
    print('插入中间评估后，两轮训练的 loss、全部参数、Adam 状态与原流程逐项完全一致。')

    # 打乱文件行序，诊断仍须按 instance_id 对齐；同一结果的 fixed/broken 应为 0。
    before_path = output_dir / 'control_2/after_pair/predictions_with_scores.csv'
    after_path = output_dir / 'probe_2/after_pair/predictions_with_scores.csv'
    frame = pd.read_csv(after_path, keep_default_na=False)
    frame.iloc[::-1].to_csv(after_path, index=False, encoding='utf-8-sig')
    summary0 = pd.read_csv(before_path.parent / 'summary.csv')
    summary1 = pd.read_csv(after_path.parent / 'summary.csv')
    records = compare_stages(before_path, after_path, summary0, summary1, output_dir / 'unchanged', 2, 'pair_ce')
    assert all(row['changed'] == row['fixed'] == row['broken'] == row['net_correct'] == 0 for row in records)


def test_known_changes(output_dir):
    gold = pd.Series(['F', 'B', 'E', 'E', 'N', 'N', 'F'])
    before = pd.Series(['E', 'B', 'E', 'F', 'N', 'E', 'N'])
    after = pd.Series(['F', 'B', 'B', 'E', 'F', 'F', 'N'])
    frame = pd.DataFrame({
        'instance_id': [f't{i}' for i in range(7)], 'pair_id': [f'p{i}' for i in range(7)],
        'text1': [f'a{i}' for i in range(7)], 'text2': [f'b{i}' for i in range(7)],
        'gold_label': gold.map(dict(zip(cfg.LABELS, cfg.FULL_LABELS))),
        'prediction': before, 'pair_prediction': before,
    })
    folder = output_dir / 'known'
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / 'before.csv', index=False)
    frame['prediction'] = frame['pair_prediction'] = after
    frame.to_csv(folder / 'after.csv', index=False)
    summary0 = pd.DataFrame({'method': ['independent', 'pair_aware'], 'weighted_f1': [0.5, 0.5]})
    summary1 = pd.DataFrame({'method': ['independent', 'pair_aware'], 'weighted_f1': [0.55, 0.55]})
    rows = compare_stages(folder / 'before.csv', folder / 'after.csv', summary0, summary1, folder, 1, 'pair_ce')
    for row in rows:
        assert (row['changed'], row['fixed'], row['broken'], row['changed_wrong'], row['net_correct']) == (5, 2, 2, 1, 0)
        assert row['F_delta_correct'] == 1 and row['E_delta_correct'] == 0 and row['N_delta_correct'] == -1
        assert abs(row['delta_f1_pp'] - 5) < 1e-10
        assert row['E_to_FB_before'] == row['E_to_FB_after'] == 1
    assert error_counts(gold, before)['nonN_to_N'] == 1
    assert len(pd.read_csv(folder / 'pair_aware/fixed.csv')) == 2
    assert len(pd.read_csv(folder / 'pair_aware/broken.csv')) == 2
    print('已知答案检查通过：fixed/broken/changed_wrong、类别变化、错误方向统计正确。')


def main():
    torch.set_num_threads(2)
    output_dir = cfg.EXPERIMENT_DIR / 'outputs/_cpu_check'
    output_dir.mkdir(parents=True, exist_ok=True)
    test_unchanged_source()
    test_training_trajectory(output_dir)
    test_known_changes(output_dir)
    # 验证新目录的解码和官方评分能复现真实历史结果，不重新跑真实模型。
    _, dev = read_data()
    historical = pd.read_csv(cfg.REFERENCE_RESULTS_DIR / 'epoch_3/predictions_with_scores.csv', keep_default_na=False)
    aligned = dev.set_index('instance_id').loc[historical['instance_id']].reset_index()
    logits = torch.tensor(historical[[f'logit_{label}' for label in cfg.LABELS]].to_numpy(), dtype=torch.float32)
    assert [cfg.LABELS[i] for i in pair_aware(logits, aligned).tolist()] == historical['pair_prediction'].tolist()
    summary = score_logits(logits, aligned, output_dir / 'historical')
    score = float(summary.set_index('method').loc['pair_aware', 'weighted_f1'])
    assert abs(score - 0.9305693600964176) < 1e-12
    print('历史 0.930569 的 660 条解码与官方评分复现通过。')
    print('CPU 检查全部通过，未加载真实大模型权重，未运行 GPU 训练。')


if __name__ == '__main__':
    main()
