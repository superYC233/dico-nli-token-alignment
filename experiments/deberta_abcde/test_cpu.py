"""CPU 校验：原版模型等价性、Row/Pair 反传、随机状态、历史评分。

不导入 train.py，不加载真实大模型权重，不运行 GPU。
"""
import ast
import copy
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_abcde import config as cfg
from experiments.deberta_abcde.check_setup import main as check_data
from experiments.deberta_abcde.data_variants import RowDataset, PairDataset
from experiments.deberta_abcde.model import DebertaDiCoModel
from experiments.deberta_abcde.evaluation import pair_aware, score_logits
from experiments.deberta_abcde.train_deberta_abcde import (
    train_rows, train_pairs, evaluate_without_changing_rng,
)


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, first, second, **kwargs):
        # 只服务于 CPU 烟雾测试，模拟 DeBERTa 的双文本分隔形式。
        a = 3 + sum(map(ord, first)) % 25
        b = 3 + sum(map(ord, second)) % 25
        ids = [1, a, 2, b, 2, 0, 0, 0]
        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }


def get_original_class(name):
    # 只编译原文件里的单个类；绝不执行原文件的顶层加载和训练代码。
    tree = ast.parse((PROJECT_ROOT / 'experiments/deberta_original/train.py').read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, str(PROJECT_ROOT / 'experiments/deberta_original/train.py'), 'exec')


def main():
    torch.set_num_threads(2)
    check_data()
    tokenizer = TinyTokenizer()
    torch.manual_seed(42)
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, relative_attention=True,
        pos_att_type=['p2c', 'c2p'],
    )
    backbone = DebertaV2Model(tiny_config)
    original_namespace = {
        'torch': torch, 'nn': nn, 'F': F,
        'my_model': copy.deepcopy(backbone), 'my_tokenizer': tokenizer,
    }
    exec(get_original_class('debertaDiCoModel'), original_namespace)
    rng = torch.get_rng_state()
    original = original_namespace['debertaDiCoModel']()
    torch.set_rng_state(rng)
    model = DebertaDiCoModel(backbone, tokenizer)
    assert original.state_dict().keys() == model.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    batch = tokenizer('a cat', 'a dog')
    original.eval()
    model.eval()
    torch.testing.assert_close(original(**batch), model(**batch), rtol=0, atol=0)

    # 训练模式也要核对：初始化、dropout、前向和梯度均与原模型一致。
    original.train()
    model.train()
    rng = torch.get_rng_state()
    old_logits = original(**batch)
    old_logits.sum().backward()
    torch.set_rng_state(rng)
    new_logits = model(**batch)
    new_logits.sum().backward()
    torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
    for (name0, param0), (name1, param1) in zip(original.named_parameters(), model.named_parameters()):
        assert name0 == name1
        if param0.grad is None:
            assert param1.grad is None
        else:
            torch.testing.assert_close(param0.grad, param1.grad, rtol=0, atol=0)
    print('原版与新版：初始化、前向、梯度完全一致。')

    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    train['label_id'] = train['label'].map(cfg.LABEL_TO_ID)
    dataset_namespace = {'torch': torch, 'Dataset': torch.utils.data.Dataset}
    exec(get_original_class('PairDataset'), dataset_namespace)
    old_pairs = dataset_namespace['PairDataset'](train, tokenizer)
    new_pairs = PairDataset(train, tokenizer)
    assert old_pairs.pairs == new_pairs.pairs
    print('原版与新版 A 组：1282 组 Pair 的顺序和目标一致。')

    frame = pd.DataFrame([
        {'instance_id': 'p0', 'pair_id': 'p', 'text1': 'a cat', 'text2': 'cat', 'label': cfg.FULL_LABELS[0]},
        {'instance_id': 'p1', 'pair_id': 'p', 'text1': 'cat', 'text2': 'a cat', 'label': cfg.FULL_LABELS[1]},
        {'instance_id': 'n0', 'pair_id': 'n', 'text1': 'cat', 'text2': 'dog', 'label': cfg.FULL_LABELS[3]},
    ])
    row_loader = DataLoader(RowDataset(frame, tokenizer), batch_size=3, shuffle=True)
    pair_loader = DataLoader(PairDataset(frame, tokenizer), batch_size=1, shuffle=True)
    device = torch.device('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    before = model.output.weight.detach().clone()
    row_loss = train_rows(model, row_loader, optimizer, row_criterion, device)
    pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, device)
    assert np.isfinite(row_loss) and np.isfinite(pair_loss)
    assert not torch.equal(before, model.output.weight)
    print('CPU 微型 DeBERTa：Row CE 和 Real Pair CE 均已完成反向更新。')

    # 真实 Dev 的小模型前向，只检查流程和随机状态，分数没有实验意义。
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    dev_loader = DataLoader(RowDataset(dev, tokenizer), batch_size=16, shuffle=False)
    temp = cfg.EXPERIMENT_DIR / 'outputs' / '_cpu_check'
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    evaluate_without_changing_rng(model, dev_loader, dev, device, temp / 'tiny_dev')
    assert random.getstate() == states[0]
    numpy_state = np.random.get_state()
    assert np.array_equal(numpy_state[1], states[1][1]) and numpy_state[2:] == states[1][2:]
    assert torch.equal(torch.get_rng_state(), states[2])
    assert model.training
    saved = temp / 'tiny_state.pth'
    torch.save(model.state_dict(), saved)
    model.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
    saved.unlink()
    print('评估后随机状态、训练模式恢复；权重保存与严格加载通过。')

    # 用已有 Qwen 原始 logits 检查相同的 pair-aware 算法和官方评分链路。
    path = PROJECT_ROOT / 'experiments/qwen_lora/outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
    if path.exists():
        history = pd.read_csv(path, keep_default_na=False)
        dev_indexed = dev.set_index('instance_id').loc[history['instance_id']].reset_index()
        logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)
        predictions = pair_aware(logits, dev_indexed)
        assert [cfg.LABELS[i] for i in predictions.tolist()] == history['pair_prediction'].tolist()
        result = score_logits(logits, dev_indexed, temp / 'historical_qwen')
        score = float(result.loc[result['method'] == 'pair_aware', 'weighted_f1'].iloc[0])
        assert abs(score - 0.8991367586828434) < 1e-12
        print('历史 660 条预测逐条一致；官方 F1 精确复现 0.899136758683。')
    print('所有 CPU 检查通过；没有加载真实大模型权重，没有进行 GPU 训练。')


if __name__ == '__main__':
    main()
