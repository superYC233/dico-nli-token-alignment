"""CPU数据、损失、原生模型和评分检查；不更新参数。"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config
from transformers.utils.hub import cached_file

from project_paths import get_model_cache_dir
from experiments.deberta_reversal_kl import config as cfg
from experiments.deberta_reversal_kl.data import read_data, GroupDataset, collate_groups
from experiments.deberta_reversal_kl.train_deberta_reversal import compute_loss, load_model
from experiments.deberta_reversal_kl.evaluation import decode_twins, predict_logits, score_logits


def check_losses():
    torch.manual_seed(42)
    logits = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 2, 1, 3, 2, 3])
    pairs = torch.tensor([[0, 2], [4, 1]])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    total, ce, kl = compute_loss(logits, labels, pairs, criterion, cfg.CONSISTENCY_WEIGHT)
    lp = logits[pairs[:, 0]].log_softmax(-1)
    lq = logits[pairs[:, 1]][:, [1, 0, 2, 3]].log_softmax(-1)
    expected = (F.kl_div(lp, lq, log_target=True, reduction='batchmean') +
                F.kl_div(lq, lp, log_target=True, reduction='batchmean')) / 2
    torch.testing.assert_close(kl, expected)
    torch.testing.assert_close(total, ce + cfg.CONSISTENCY_WEIGHT * kl)
    gradients = torch.autograd.grad(kl, logits, retain_graph=True)[0]
    assert gradients[[3, 5]].eq(0).all()
    assert gradients[pairs.flatten()].abs().sum(1).gt(0).all()
    zero_weight = compute_loss(logits, labels, pairs, criterion, 0.0)[0]
    torch.testing.assert_close(zero_weight, ce, rtol=0, atol=0)

    correct = torch.tensor([[7., 0, 0, 0], [0, 7., 0, 0]])
    wrong_n = torch.tensor([[0., 0, 0, 7.], [0, 0, 0, 7.]])
    target = torch.tensor([0, 1])
    pair = torch.tensor([[0, 1]])
    good_loss, _, good_kl = compute_loss(correct, target, pair, criterion)
    bad_loss, _, bad_kl = compute_loss(wrong_n, target, pair, criterion)
    assert good_kl.item() == 0 and bad_kl.item() == 0 and bad_loss > good_loss
    # 只改变两方向N的相对概率，完整四维KL应感知到差异。
    unequal_n = torch.tensor([[3., 0, 0, -3.], [0, 3., 0, 3.]])
    assert compute_loss(unequal_n, target, pair, criterion)[2] > 0
    only_n = torch.zeros(2, 4, requires_grad=True)
    loss, ce, kl = compute_loss(only_n, torch.tensor([3, 3]), torch.empty(0, 2, dtype=torch.long), criterion)
    assert loss == ce and kl.item() == 0
    loss.backward()
    assert only_n.grad[:, 3].lt(0).all()
    print('损失通过：双向KL公式正确、两边都有梯度、N单例不进KL、CE会惩罚两边一起错答N。')


def check_tiny_model():
    config = DebertaV2Config(
        vocab_size=32, hidden_size=24, pooler_hidden_size=24,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
        max_position_embeddings=32, relative_attention=False,
        pooler_dropout=0.0, num_labels=4,
        id2label=dict(enumerate(cfg.FULL_LABELS)), label2id=cfg.LABEL_TO_ID,
    )
    model = AutoModelForSequenceClassification.from_config(config)
    ids = torch.tensor([[1, 3, 2, 4, 2, 0], [1, 4, 2, 3, 2, 0]])
    mask = ids.ne(0).long()
    logits = model(input_ids=ids, attention_mask=mask).logits
    loss, _, _ = compute_loss(logits, torch.tensor([0, 1]), torch.tensor([[0, 1]]), nn.CrossEntropyLoss())
    loss.backward()
    for parameter in [model.classifier.weight, model.pooler.dense.weight, model.deberta.embeddings.word_embeddings.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = AutoModelForSequenceClassification.from_config(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(input_ids=ids, attention_mask=mask).logits,
            restored(input_ids=ids, attention_mask=mask).logits, rtol=0, atol=0,
        )
    print('小模型通过：原生分类头、pooler及编码器有梯度，完整保存重载一致。')


def check_real_setup():
    train, dev = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    dataset = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(
        dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True, collate_fn=collate_groups,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    rows = []
    pairs = 0
    reverse = torch.tensor([1, 0, 2, 3])
    for batch in loader:
        rows.extend(batch['source_rows'].tolist())
        pairs += len(batch['pair_indices'])
        assert len(batch['labels']) <= 16
        for first, second in batch['pair_indices']:
            assert batch['labels'][first] != 3 and batch['labels'][second] == reverse[batch['labels'][first]]
    assert len(train) == 3042 and len(dev) == 660 and len(loader) == 220
    assert sorted(rows) == list(range(3042)) and pairs == 1282
    print('数据通过：3042行各一次、1282真实正反组各一次、220步；没有反转N或数据增强。')

    model = load_model().eval()
    assert model.classifier.weight.shape == (4, 768)
    assert model.config.id2label == dict(enumerate(cfg.FULL_LABELS))
    checkpoint_path = cached_file(
        cfg.MODEL_NAME, 'model.safetensors', revision=cfg.BASE_REVISION,
        cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    with safe_open(checkpoint_path, framework='pt', device='cpu') as checkpoint:
        assert torch.equal(model.pooler.dense.weight, checkpoint.get_tensor('pooler.dense.weight'))
        assert torch.equal(model.pooler.dense.bias, checkpoint.get_tensor('pooler.dense.bias'))
    encoded = dataset.encoded
    batch = {'input_ids': encoded['input_ids'][:2], 'attention_mask': encoded['attention_mask'][:2]}
    with torch.inference_mode():
        result = model(**batch).logits
    assert result.shape == (2, 4) and torch.isfinite(result).all()
    print('真实CPU前向通过：四类顺序F/B/E/N正确，原NLI pooler权重逐元素保留。')

    # 已有金标只用于检查评分接口，既不更新权重，也不构造训练样本。
    gold = torch.tensor(dev['label'].map(cfg.LABEL_TO_ID).tolist())
    perfect = torch.zeros(len(dev), 4)
    perfect[torch.arange(len(dev)), gold] = 8
    print('下面的1.0仅为人工金标分数对评分接口的单元检查，不是模型成绩。')
    summary = score_logits(perfect, dev, cfg.EXPERIMENT_DIR / 'outputs/setup_check', cfg.GOLD_PATH)
    assert summary['weighted_f1'].eq(1).all() and summary['soft_cons'].eq(1).all()
    pair = next(group.index.tolist() for _, group in dev.groupby('pair_id') if len(group) == 2)
    singleton = next(group.index[0] for _, group in dev.groupby('pair_id') if len(group) == 1)
    modified = perfect.clone()
    modified[pair] = torch.tensor([0., 0, 0, 9.])
    modified[singleton] = torch.tensor([9., 0, 0, 0])
    assert decode_twins(modified, dev, True)[pair].eq(3).all()
    assert decode_twins(modified, dev, False)[pair].ne(3).all()
    assert decode_twins(modified, dev, True)[singleton] == 0
    assert decode_twins(modified, dev, False)[singleton] == 0
    print('评分和解码通过：NN候选与三组合区别明确，单例不会被强制判N。')


if __name__ == '__main__':
    torch.set_num_threads(4)
    check_losses()
    check_tiny_model()
    check_real_setup()
    print('全部检查通过。未执行optimizer.step，未启动训练，未写入模型目录。')
