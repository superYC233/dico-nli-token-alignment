"""真实 WordNet + 随机微型 DeBERTa 的 CPU 验证，不加载大权重。"""
import copy
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

from experiments.deberta_wordnet import config as cfg
from experiments.deberta_wordnet.check_setup import main as check_setup
from experiments.deberta_wordnet.data import DiCoDataset, PairDataset, build_feature_tensor, read_data
from experiments.deberta_wordnet.lexical_features import build_lexical_features
from experiments.deberta_wordnet.wordnet_features import (
    FEATURE_NAMES, WORDNET_FEATURE_NAMES, build_features,
)
from experiments.deberta_wordnet.wordnet_resource import ensure_wordnet
from experiments.deberta_wordnet.model import DebertaDiCoModel
from experiments.deberta_lexical.model import DebertaDiCoModel as PreviousLexicalModel
from experiments.deberta_wordnet.evaluation import pair_aware, predict_logits, score_logits
from experiments.deberta_wordnet.train_deberta_wordnet import build_optimizer, train_rows, train_pairs


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


def test_relations():
    def named(first, second):
        values, _ = build_features(first, second)
        return dict(zip(FEATURE_NAMES, values))

    car = named('black car', 'black vehicle')
    assert car['wn_specific_1_to_general_2'] == 1.0
    assert car['wn_specific_2_to_general_1'] == 0.0
    assert named('car', 'automobile')['wn_synonym_coverage_1'] == 1.0
    assert named('hot', 'cold')['wn_antonym_coverage_1'] == 1.0
    assert named('dog', 'cat')['wn_specific_1_to_general_2'] == 0.0
    # 共同祖先 animal 不等于 dog 蕴含 cat。
    assert named('dog', 'cat')['wn_specific_2_to_general_1'] == 0.0
    for first, second in [
        ('black car', 'black vehicle'), ('cars', 'automobile'), ('hot', 'cold'),
        ('qzxxyzz', 'vehicle'), ('the', ''), ('a dog', 'a dog'),
    ]:
        a = named(first, second)
        b = named(second, first)
        for i in range(0, len(WORDNET_FEATURE_NAMES), 2):
            name0, name1 = WORDNET_FEATURE_NAMES[i:i + 2]
            assert a[name0] == b[name1] and a[name1] == b[name0]
        values, _ = build_features(first, second)
        assert np.isfinite(values).all()
    assert named('qzxxyzz', 'vehicle')['wn_lookup_coverage_1'] == 0.0
    print('真实 WordNet：同义、上下位方向、反义、共同祖先、未知词、空文本通过。')


def main():
    torch.set_num_threads(2)
    ensure_wordnet()
    check_setup()
    test_relations()
    tokenizer = TinyTokenizer()
    tiny_config = DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, relative_attention=True, pos_att_type=['p2c', 'c2p'],
    )
    torch.manual_seed(42)
    backbone = DebertaV2Model(tiny_config)
    old_backbone = copy.deepcopy(backbone)
    state = torch.get_rng_state()
    old = PreviousLexicalModel(old_backbone, tokenizer, 20)
    old_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    model = DebertaDiCoModel(backbone, tokenizer, len(FEATURE_NAMES))
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert torch.count_nonzero(model.wordnet_projection.weight).item() == 0

    batch = tokenizer('black car', 'black vehicle')
    values, _ = build_features('black car', 'black vehicle')
    features = torch.tensor([values])
    old.train()
    model.train()
    state = torch.get_rng_state()
    old_logits = old(**batch, lexical_features=features[:, :20])
    old_logits.sum().backward()
    torch.set_rng_state(state)
    new_logits = model(**batch, lexical_features=features)
    new_logits.sum().backward()
    torch.testing.assert_close(old_logits, new_logits, rtol=0, atol=0)
    new_parameters = dict(model.named_parameters())
    for name, parameter in old.named_parameters():
        if parameter.grad is None:
            assert new_parameters[name].grad is None
        else:
            torch.testing.assert_close(parameter.grad, new_parameters[name].grad, rtol=0, atol=0)
    print('与原词汇分支版相比：初始化、随机序列、初始输出和原参数梯度一致。')

    frame = pd.DataFrame([
        {'instance_id': 'p0', 'pair_id': 'p', 'text1': 'black car', 'text2': 'black vehicle', 'label': cfg.FULL_LABELS[0]},
        {'instance_id': 'p1', 'pair_id': 'p', 'text1': 'black vehicle', 'text2': 'black car', 'label': cfg.FULL_LABELS[1]},
        {'instance_id': 'n0', 'pair_id': 'n', 'text1': 'cat', 'text2': 'girl', 'label': cfg.FULL_LABELS[3]},
    ])
    features = build_feature_tensor(frame)
    original = torch.tensor([build_lexical_features(a, b) for a, b in zip(frame.text1, frame.text2)])
    torch.testing.assert_close(features[:, :20], original, rtol=0, atol=0)
    optimizer = build_optimizer(model)
    assert [group['lr'] for group in optimizer.param_groups] == [2e-5, 1e-3]
    expected = {id(p) for name, p in model.named_parameters() if name.startswith(('lexical_output.', 'wordnet_projection.'))}
    assert {id(p) for p in optimizer.param_groups[1]['params']} == expected
    all_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    row_loader = DataLoader(DiCoDataset(frame, tokenizer, features), batch_size=3, shuffle=True)
    pair_dataset = PairDataset(frame, tokenizer, features)
    assert torch.equal(pair_dataset[0][2], features[0])
    assert torch.equal(pair_dataset[0][5], features[1])
    device = torch.device('cpu')
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    train_rows(model, row_loader, optimizer, row_criterion, device)
    train_pairs(model, DataLoader(pair_dataset, batch_size=1), optimizer, pair_criterion, device)
    assert model.wordnet_projection.weight.grad.abs().sum().item() > 0
    assert torch.count_nonzero(model.wordnet_projection.weight).item() > 0
    print('原主干/新分支学习率分组正确，WordNet 投影已通过实际反传更新。')

    _, dev = read_data()
    dev_features = build_feature_tensor(dev)
    loader = DataLoader(DiCoDataset(dev, tokenizer, dev_features), batch_size=16, shuffle=False)
    before = predict_logits(model, loader, device)
    output_dir = cfg.EXPERIMENT_DIR / 'outputs/_cpu_check'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'tiny_state.pth'
    torch.save(model.state_dict(), path)
    reloaded = DebertaDiCoModel(DebertaV2Model(tiny_config), tokenizer, len(FEATURE_NAMES))
    reloaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    after = predict_logits(reloaded, loader, device)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    path.unlink()
    score_logits(before, dev, output_dir / 'tiny_dev')

    history_path = cfg.PROJECT_ROOT / 'experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv'
    if history_path.exists():
        history = pd.read_csv(history_path, keep_default_na=False)
        aligned = dev.set_index('instance_id').loc[history.instance_id].reset_index()
        logits = torch.tensor(history[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)
        pred = pair_aware(logits, aligned)
        assert [cfg.LABELS[i] for i in pred.tolist()] == history.pair_prediction.tolist()
        summary = score_logits(logits, aligned, output_dir / 'historical_lexical')
        score = float(summary.loc[summary.method == 'pair_aware', 'weighted_f1'].iloc[0])
        assert abs(score - 0.9305693600964176) < 1e-12
        print('0.930569 的历史预测和评分精确复现。')
    print('CPU 检查通过。没有加载真实大模型权重，没有进行 GPU 训练。')


if __name__ == '__main__':
    main()
