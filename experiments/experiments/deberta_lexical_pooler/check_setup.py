"""CPU检查：核对真实NLI pooler、数据覆盖、梯度与保存重载；不更新参数。"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DebertaV2Config, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_lexical_pooler import config as cfg
from experiments.deberta_lexical_pooler.model import (
    DebertaPoolerModel, load_pretrained_model, read_nli_pooler,
)
from experiments.deberta_lexical_pooler.train_deberta_pooler import read_data, compute_joint_loss
from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2


def check_tiny_model():
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, pooler_hidden_size=24,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
        max_position_embeddings=32, relative_attention=False,
        pooler_dropout=0.0, pooler_hidden_act='gelu',
    )
    backbone = AutoModel.from_config(config)
    set_seed(42)
    original = DebertaDiCoModel(backbone, TinyTokenizer(), 20)
    # 非零残差检查，不能仅依赖两分支零初始化时的恒等结果。
    with torch.no_grad():
        original.match_output[-1].weight.normal_(std=0.02)
        original.lexical_output[-1].weight.normal_(std=0.02)
    set_seed(42)
    candidate = DebertaPoolerModel(backbone, TinyTokenizer(), 20, use_nli_pooler=True)
    missing, unexpected = candidate.load_state_dict(original.state_dict(), strict=False)
    assert set(missing) == {'nli_pooler.dense.weight', 'nli_pooler.dense.bias'}
    assert not unexpected
    ids = torch.tensor([[1, 3, 2, 4, 2, 0], [1, 4, 2, 3, 2, 0]])
    mask = ids.ne(0).long()
    features = torch.arange(40, dtype=torch.float32).reshape(2, 20) / 40
    original.eval()
    candidate.eval()
    with torch.inference_mode():
        candidate.use_nli_pooler = False
        torch.testing.assert_close(candidate(ids, mask, features), original(ids, mask, features), rtol=0, atol=0)
    candidate.use_nli_pooler = True
    candidate.train()
    logits = candidate(ids, mask, features)
    labels = torch.tensor([0, 1])
    loss, _, _ = compute_joint_loss(
        logits, labels, torch.tensor([[0, 1]]), torch.tensor([0]),
        nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), cfg.PAIR_WEIGHT,
    )
    loss.backward()
    gradient = candidate.nli_pooler.dense.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert candidate.deberta_model.embeddings.word_embeddings.weight.grad is not None

    candidate.eval()
    with torch.inference_mode():
        expected = candidate(ids, mask, features)
    buffer = io.BytesIO()
    torch.save(candidate.state_dict(), buffer)
    buffer.seek(0)
    restored = DebertaPoolerModel(AutoModel.from_config(config), TinyTokenizer(), 20)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, mask, features), expected, rtol=0, atol=0)
    print('小模型通过：关闭pooler等于旧版；pooler和编码器有梯度；完整保存重载一致。')


def check_real_model():
    set_seed(cfg.SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    train, dev = read_data()
    features = build_feature_tensor(train)
    dataset = GroupDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
        collate_fn=collate_groups, generator=torch.Generator().manual_seed(cfg.SEED),
    )
    rows = []
    pairs = 0
    for batch in loader:
        rows.extend(batch['source_rows'].tolist())
        pairs += len(batch['pair_targets'])
    assert len(train) == 3042 and len(dev) == 660
    assert sorted(rows) == list(range(3042)) and pairs == 1282 and len(loader) == 220
    print('真实数据通过：3042行各一次、1282组Pair各一次、每轮220步，没有增强。')

    model = load_pretrained_model(
        tokenizer, cfg.MODEL_NAME, cfg.BASE_REVISION, use_nli_pooler=True,
    ).eval()
    for name, value in read_nli_pooler(cfg.MODEL_NAME, cfg.BASE_REVISION).items():
        assert torch.equal(model.nli_pooler.state_dict()[name], value)
    print('NLI pooler逐元素核对通过；参数量：', sum(p.numel() for p in model.nli_pooler.parameters()))
    encoded = tokenizer(
        train['text1'].iloc[:2].tolist(), train['text2'].iloc[:2].tolist(),
        padding='max_length', truncation=True, max_length=cfg.MAX_LENGTH, return_tensors='pt',
    )
    calls = []
    handle = model.deberta_model.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        output = model(encoded['input_ids'], encoded['attention_mask'], features[:2])
    handle.remove()
    assert len(calls) == 1 and output.shape == (2, 4) and torch.isfinite(output).all()
    with torch.inference_mode():
        model.use_nli_pooler = False
        disabled = model(encoded['input_ids'], encoded['attention_mask'], features[:2])
        original = DebertaDiCoModel.forward(
            model, encoded['input_ids'], encoded['attention_mask'], features[:2],
        )
    torch.testing.assert_close(disabled, original, rtol=0, atol=0)
    print('真实CPU前向通过：一遍编码得到四类分数；关闭pooler与旧forward逐位一致。')


if __name__ == '__main__':
    torch.set_num_threads(4)
    check_tiny_model()
    check_real_model()
    print('全部检查通过。未启动训练，没有optimizer.step，也未写入模型目录。')
