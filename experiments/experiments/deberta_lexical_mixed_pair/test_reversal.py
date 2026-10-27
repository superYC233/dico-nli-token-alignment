"""CPU检查：原架构、原数据、原Row CE，只替换反向辅助目标。"""
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DebertaV2Config, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_mixed_pair import config_reversal as cfg
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_mixed_pair.train_deberta_reversal import (
    read_data, load_model, make_dev_loader, compute_reversal_loss,
)
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.evaluation import predict_logits


def check_baseline_settings():
    reference = json.loads((cfg.EXPERIMENT_DIR / 'outputs/run01_mixed_pair/run_config.json').read_text(encoding='utf-8'))
    for local, saved in [
        ('LEARNING_RATE', 'learning_rate'), ('LEXICAL_LEARNING_RATE', 'lexical_learning_rate'),
        ('ROW_WEIGHTS', 'row_weights'), ('WEIGHT_DECAY', 'weight_decay'), ('EPOCHS', 'epochs'),
        ('GROUP_BATCH_SIZE', 'group_batch_size'), ('MAX_LENGTH', 'max_length'),
        ('SEED', 'seed'), ('LEXICAL_HIDDEN_SIZE', 'lexical_hidden_size'),
    ]:
        assert getattr(cfg, local) == reference[saved], local
    assert cfg.ROW_LABEL_SMOOTHING == 0.0
    assert hashlib.sha256(cfg.TRAIN_PATH.read_bytes()).hexdigest() == reference['train_sha256']
    assert hashlib.sha256(cfg.DEV_PATH.read_bytes()).hexdigest() == reference['dev_sha256']
    print('基线核对通过：原run01的训练参数、原3042/660数据保持一致；新目标按独立F1选模。')


def check_loss():
    torch.manual_seed(42)
    logits = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([3, 0, 1, 2, 3, 2])
    pairs = torch.tensor([[1, 2], [3, 5]])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64))
    total, row, kl = compute_reversal_loss(logits, labels, pairs, criterion, cfg.CONSISTENCY_WEIGHT)
    torch.testing.assert_close(row, criterion(logits, labels), rtol=0, atol=0)
    torch.testing.assert_close(total, row + cfg.CONSISTENCY_WEIGHT * kl)
    gradient = torch.autograd.grad(kl, logits, retain_graph=True)[0]
    assert gradient[[0, 4]].eq(0).all()
    assert gradient[[1, 2, 3, 5]].abs().sum(dim=1).gt(0).all()
    improved_n = logits.detach().clone()
    improved_n[0, 3] += 1
    better, _, unchanged_kl = compute_reversal_loss(improved_n, labels, pairs, criterion, cfg.CONSISTENCY_WEIGHT)
    assert better < total and unchanged_kl == kl
    zero = compute_reversal_loss(logits, labels, pairs, criterion, 0.0)[0]
    torch.testing.assert_close(zero, row, rtol=0, atol=0)
    only_n = compute_reversal_loss(
        torch.zeros(2, 4, dtype=torch.float64), torch.tensor([3, 3]),
        torch.empty(0, 2, dtype=torch.long), criterion,
    )
    assert only_n[0] == only_n[1] and only_n[2].item() == 0
    print('损失通过：保留原加权Row CE；KL双向反传；N单例仅进Row；无真实Pair时不造KL。')


def check_tiny_model():
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, max_position_embeddings=32, relative_attention=False,
    )
    class Tokenizer:
        cls_token_id = 1
        sep_token_id = 2
    model = DebertaDiCoModel(AutoModel.from_config(config), Tokenizer(), 20)
    ids = torch.tensor([[1, 3, 2, 4, 2, 0], [1, 4, 2, 3, 2, 0]])
    mask = ids.ne(0).long()
    features = torch.arange(40, dtype=torch.float32).reshape(2, 20) / 40
    logits = model(ids, mask, features)
    loss, _, _ = compute_reversal_loss(
        logits, torch.tensor([0, 1]), torch.tensor([[0, 1]]),
        nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS)), cfg.CONSISTENCY_WEIGHT,
    )
    loss.backward()
    for parameter in [model.output.weight, model.match_output[-1].weight,
                      model.lexical_output[-1].weight, model.deberta_model.embeddings.word_embeddings.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DebertaDiCoModel(AutoModel.from_config(config), Tokenizer(), 20).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, mask, features), restored(ids, mask, features), rtol=0, atol=0)
    print('小模型通过：CLS/Alignment/Lexical和编码器都有梯度，原结构保存重载一致。')


def check_real_data_and_forward():
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
    pairs = 0
    for batch in loader:
        rows.extend(batch['source_rows'].tolist())
        pairs += len(batch['pair_indices'])
        for first, second in batch['pair_indices']:
            assert batch['labels'][first] != 3 and batch['labels'][second] != 3
        assert torch.equal(batch['features'], dataset.features[batch['source_rows']])
    assert len(train) == 3042 and len(dev) == 660 and len(loader) == 220
    assert sorted(rows) == list(range(3042)) and pairs == 1282
    print('数据通过：3042行各一次、1282真实反向组各一次、220步、没有增强或反向N。')
    model = load_model(tokenizer).eval()
    assert type(model) is DebertaDiCoModel and not hasattr(model, 'nli_pooler')
    calls = []
    handle = model.deberta_model.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        result = model(dataset.encoded['input_ids'][:2], dataset.encoded['attention_mask'][:2], dataset.features[:2])
    handle.remove()
    assert len(calls) == 1 and result.shape == (2, 4) and torch.isfinite(result).all()
    print('真实CPU前向通过：直接复用原模型，一遍编码得到四类分数。')

    class FeatureModel(nn.Module):
        def forward(self, input_ids, attention_mask, features):
            return features[:, :4]
    dev_loader = make_dev_loader(dev, tokenizer, build_feature_tensor(dev), cfg.MAX_LENGTH, cfg.DEV_BATCH_SIZE)
    probe = FeatureModel().train()
    state = torch.get_rng_state().clone()
    first = predict_logits(probe, dev_loader, torch.device('cpu'))
    second = predict_logits(probe, dev_loader, torch.device('cpu'))
    assert probe.training and torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first, second) and first.shape == (660, 4)
    print('验证通过：预测次序固定，Dev评估不消耗正式训练的全局随机数。')


if __name__ == '__main__':
    torch.set_num_threads(4)
    check_baseline_settings()
    check_loss()
    check_tiny_model()
    check_real_data_and_forward()
    print('全部检查通过；没有optimizer.step，没有训练真实模型或写入模型目录。')
