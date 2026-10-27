"""只用随机微型 DeBERTa 检查，不加载真实大权重、不启动 GPU 训练。"""
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_altn import config as cfg
from experiments.deberta_altn.altn_features import (
    build_features, build_position_features, align_words, side_features,
)
from experiments.deberta_altn.check_setup import main as check_setup
from experiments.deberta_altn.data import read_data, build_feature_tensor, DiCoDataset, PairDataset
from experiments.deberta_altn.model import DebertaDiCoModel
from experiments.deberta_altn.evaluation import (
    pair_aware, predict_logits, score_logits, compare_with_lexical,
)
from experiments.deberta_altn.train_deberta_altn import (
    build_optimizer, train_rows, train_pairs, evaluate_without_changing_rng,
    save_best_weights, make_run_config,
)
from experiments.deberta_lexical.model import DebertaDiCoModel as OriginalLexicalModel
from experiments.deberta_lexical.train_deberta_lexical import build_optimizer as original_optimizer


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


def test_position_features():
    assert build_position_features('train station', 'station') == [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]
    assert build_position_features('a dog', 'a dog') == [0] * 8
    assert build_position_features('', '') == [0] * 8
    assert build_position_features('', 'cat') == [0, 0, 0, 0, 1, 1, 1, 1]
    assert side_features([True, False, True, False, True]) == [2/5, 1/5, 2/5, 4/5]
    assert side_features([True, False, False, True, True]) == [1/5, 2/5, 2/5, 3/5]
    # 词集合和长度完全相同，但差异片段的分布不同，新增特征确实提供新数值。
    first = build_features('a x b y c', 'a b c')
    second = build_features('a x y b c', 'a b c')
    assert first[:20] == second[:20] and first[20:] != second[20:]
    sequences = [list(items) for n in range(5) for items in itertools.product(['a', 'b'], repeat=n)]
    for first in sequences:
        for second in sequences:
            mask1, mask2, matches = align_words(first, second)
            reversed2, reversed1, reversed_matches = align_words(second, first)
            assert mask1 == reversed1 and mask2 == reversed2
            assert matches == [(j, i) for i, j in reversed_matches]
            assert sum(mask1) == sum(mask2) == len(matches)
            assert len({i for i, _ in matches}) == len(matches)
            assert len({j for _, j in matches}) == len(matches)
            assert all(first[i] == second[j] for i, j in matches)
            assert all(0 <= value <= 1 for value in side_features(mask1) + side_features(mask2))
    _, _, matches = align_words(['dog', 'dog'], ['dog'])
    assert len(matches) == 1
    print('LCS 重复词一对一、平局、方向互换、连续片段和边界检查通过。')


def loaders(frame, tokenizer, features):
    return (
        DataLoader(DiCoDataset(frame, tokenizer, features), batch_size=5, shuffle=True),
        DataLoader(PairDataset(frame, tokenizer, features), batch_size=2, shuffle=True),
    )


def main():
    torch.set_num_threads(2)
    check_setup()
    test_position_features()
    output_dir = cfg.EXPERIMENT_DIR / 'outputs' / f'_cpu_check_{time.time_ns()}'
    output_dir.mkdir(parents=True)
    tokenizer = TinyTokenizer()
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    torch.manual_seed(42)
    backbone = DebertaV2Model(tiny_config)
    before = torch.get_rng_state().clone()
    original = OriginalLexicalModel(copy.deepcopy(backbone), tokenizer, 20)
    after_original = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    model = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer)
    assert torch.equal(torch.get_rng_state(), after_original)
    torch.set_rng_state(before)
    control = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, use_position_features=False)
    assert torch.equal(torch.get_rng_state(), after_original)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(control.state_dict()[key], value, rtol=0, atol=0)
    assert sum(p.numel() for p in model.lexical_output.parameters()) == 804
    assert sum(p.numel() for p in model.position_projection.parameters()) == 256
    assert torch.count_nonzero(model.position_projection.weight) == 0
    assert not hasattr(control, 'position_projection')

    frame = pd.DataFrame([
        {'instance_id':'p0', 'pair_id':'p', 'text1':'a black car', 'text2':'a car', 'label':cfg.FULL_LABELS[0]},
        {'instance_id':'p1', 'pair_id':'p', 'text1':'a car', 'text2':'a black car', 'label':cfg.FULL_LABELS[1]},
        {'instance_id':'e0', 'pair_id':'e', 'text1':'a dog', 'text2':'the dog', 'label':cfg.FULL_LABELS[2]},
        {'instance_id':'e1', 'pair_id':'e', 'text1':'the dog', 'text2':'a dog', 'label':cfg.FULL_LABELS[2]},
        {'instance_id':'n0', 'pair_id':'n', 'text1':'cat', 'text2':'dog', 'label':cfg.FULL_LABELS[3]},
    ])
    features = build_feature_tensor(frame)
    # 原 Dataset 要求 20 维，使用原版 Dataset 单独构造。
    from experiments.deberta_lexical.data import DiCoDataset as OldDataset, PairDataset as OldPairs
    old_rows = DataLoader(OldDataset(frame, tokenizer, features[:, :20].contiguous()), batch_size=5, shuffle=True)
    old_pairs = DataLoader(OldPairs(frame, tokenizer, features[:, :20].contiguous()), batch_size=2, shuffle=True)
    control_rows, control_pairs = loaders(frame, tokenizer, features)
    zero_features = features.clone()
    zero_features[:, 20:] = 0
    zero_rows, zero_pairs = loaders(frame, tokenizer, zero_features)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    device = torch.device('cpu')
    # 对照在完整两轮 Row->Pair 后仍与原 Lexical 一致；零位置输入也不能偷偷改变旧分支。
    rng = torch.get_rng_state().clone()
    for network, row_loader, pair_loader, optimizer in [
        (original, old_rows, old_pairs, original_optimizer(original)),
        (control, control_rows, control_pairs, build_optimizer(control)),
        (model, zero_rows, zero_pairs, build_optimizer(model)),
    ]:
        torch.set_rng_state(rng)
        network.train()
        for _ in range(2):
            train_rows(network, row_loader, optimizer, row_criterion, device)
            train_pairs(network, pair_loader, optimizer, pair_criterion, device)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(control.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert torch.count_nonzero(model.position_projection.weight) == 0
    print('原 Lexical、关闭位置特征、位置输入为零：两轮训练的原有参数逐项一致。')

    optimizer = build_optimizer(model)
    original_ids = {id(p) for name, p in model.named_parameters()
                    if not name.startswith(('lexical_output.', 'position_projection.'))}
    branch_ids = {id(p) for name, p in model.named_parameters()
                  if name.startswith(('lexical_output.', 'position_projection.'))}
    assert optimizer.param_groups[0]['lr'] == 2e-5
    assert optimizer.param_groups[1]['lr'] == 1e-3
    assert {id(p) for p in optimizer.param_groups[0]['params']} == original_ids
    assert {id(p) for p in optimizer.param_groups[1]['params']} == branch_ids
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    row_loader, pair_loader = loaders(frame, tokenizer, features)
    model.train()
    train_rows(model, row_loader, optimizer, row_criterion, device)
    assert model.position_projection.weight.grad.abs().sum() > 0
    assert model.position_projection.weight.abs().sum() > 0
    train_pairs(model, pair_loader, optimizer, pair_criterion, device)
    assert model.position_projection.weight.grad.abs().sum() > 0
    item = pair_loader.dataset[0]
    assert torch.equal(item[2][20:24], item[5][24:28])
    assert torch.equal(item[2][24:28], item[5][20:24])
    print('学习率分组无遗漏、真实 Pair 的方向特征、Row/Pair 两阶段位置梯度检查通过。')

    _, dev = read_data()
    dev_features = build_feature_tensor(dev)
    dev_loader = DataLoader(DiCoDataset(dev, tokenizer, dev_features), batch_size=16, shuffle=False)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    evaluate_without_changing_rng(model, dev_loader, dev, device, output_dir / 'tiny_dev')
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2]) and model.training
    checkpoint = output_dir / 'tiny_best.pth'
    save_best_weights(model, checkpoint)
    reloaded = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(
        predict_logits(model, dev_loader, device), predict_logits(reloaded, dev_loader, device), rtol=0, atol=0,
    )
    old_bytes = checkpoint.read_bytes()

    def failed_save(state, temporary):
        Path(temporary).write_bytes(b'partial write')
        raise OSError('simulated disk failure')

    with patch('experiments.deberta_altn.train_deberta_altn.torch.save', side_effect=failed_save):
        try:
            save_best_weights(model, checkpoint)
        except OSError:
            pass
        else:
            raise AssertionError('模拟写盘失败未触发。')
    assert checkpoint.read_bytes() == old_bytes
    assert not checkpoint.with_name(checkpoint.name + '.tmp').exists()
    manifest = make_run_config(True, 3042, 660, 1282)
    assert manifest['use_position_features'] and manifest['lexical_learning_rate'] == 1e-3
    assert len(manifest['feature_names']) == 28
    (output_dir / 'run_config.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完整保存重载、评估 RNG 恢复、写盘失败保护检查通过。')

    # 用已保存的真实历史 logits 核验官方评分和四类错误统计，不加载大权重。
    if cfg.LEXICAL_REFERENCE_PATH.is_file():
        history = pd.read_csv(cfg.LEXICAL_REFERENCE_PATH, keep_default_na=False)
        aligned = dev.set_index('instance_id').loc[history['instance_id']].reset_index()
        logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)
        assert [cfg.LABELS[i] for i in pair_aware(logits, aligned).tolist()] == history['pair_prediction'].tolist()
        summary = score_logits(logits, aligned, output_dir / 'historical_lexical')
        best = summary[summary['method'] == 'pair_aware'].iloc[0]
        assert abs(best['weighted_f1'] - 0.9305693600964176) < 1e-12
        assert [best[key] for key in ['FB_to_E_errors', 'E_to_FB_errors', 'F_B_reversal_errors', 'N_errors']] == [14,16,8,8]
        compare_with_lexical(output_dir / 'historical_lexical/predictions_with_scores.csv', output_dir / 'self_comparison')
        assert pd.read_csv(output_dir / 'self_comparison/fixed.csv').empty
        assert pd.read_csv(output_dir / 'self_comparison/broken.csv').empty
        print('历史 0.930569 逐条解码、官方评分、四格错误计数与比较文件检查通过。')
    print('CPU 检查完成，没有加载真实大模型或运行 GPU 训练。')
    print('检查记录：', output_dir)


if __name__ == '__main__':
    main()
