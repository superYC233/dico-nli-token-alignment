"""检查无增强设置、旧循环梯度、本地分词和可选真实GPU前后向；不训练。"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_base_lexical import config as cfg
from experiments.deberta_base_lexical import train_deberta_base_lexical as current
from experiments.deberta_lexical import train_deberta_lexical as old
from experiments.deberta_lexical.data import build_feature_tensor, DiCoDataset, PairDataset
from experiments.deberta_lexical.lexical_features import FEATURE_NAMES


class TinyLexicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(8, 4)
        self.lexical_output = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, ids, mask, features):
        return self.output(self.dropout(self.embedding(ids).mean(1))) + self.lexical_output(features)


class GradientRecorder:
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self):
        # 只记录梯度，不更新任何参数。
        self.gradients.append({name: p.grad.detach().clone() for name, p in self.model.named_parameters()})


def check_training_loops():
    generator = torch.Generator().manual_seed(93)
    ids = torch.randint(0, 64, (4, 7), generator=generator)
    masks = torch.ones_like(ids)
    features = torch.randn(4, 20, generator=generator)
    labels = torch.tensor([0, 1, 2, 3])
    rows = [(ids, masks, features, labels)]
    pairs = [(ids[:2], masks[:2], features[:2], ids[2:], masks[2:], features[2:], torch.tensor([0, 2]))]
    row_ce, pair_ce = current.build_criteria(torch.device('cpu'))
    model = TinyLexicalModel()
    reference = copy.deepcopy(model)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    before, after = GradientRecorder(reference), GradientRecorder(model)
    torch.manual_seed(51)
    original_losses = [old.train_rows(reference, rows, before, row_ce, torch.device('cpu')),
                       old.train_pairs(reference, pairs, before, pair_ce, torch.device('cpu'))]
    original_rng = torch.get_rng_state()
    torch.manual_seed(51)
    new_losses = [current.train_rows(model, rows, after, row_ce, torch.device('cpu')),
                  current.train_pairs(model, pairs, after, pair_ce, torch.device('cpu'))]
    assert original_losses == new_losses and torch.equal(original_rng, torch.get_rng_state())
    for first, second in zip(before.gradients, after.gradients):
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert all(torch.equal(p, initial[name]) for name, p in model.named_parameters())
    print('训练循环通过：与0.9306版Row/Pair逐参数梯度和RNG完全一致；未更新参数。')


def check_data_and_tokenizer():
    reference = json.loads((cfg.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    fields = ['seed', 'epochs', 'row_batch_size', 'pair_batch_size', 'dev_batch_size',
              'max_length', 'learning_rate', 'lexical_learning_rate', 'weight_decay',
              'row_weights', 'pair_weights', 'pair_weight', 'pair_label_smoothing', 'lexical_hidden_size']
    for field in fields:
        assert getattr(cfg, field.upper()) == reference[field], field
    assert current.file_hash(cfg.TRAIN_PATH) == reference['train_sha256']
    assert current.file_hash(cfg.DEV_PATH) == reference['dev_sha256']
    assert current.file_hash(ROOT / 'experiments/deberta_lexical/lexical_features.py') == reference['feature_source_sha256']
    train, dev = current.read_data()
    assert len(train) == 3042 and len(dev) == 660
    features = build_feature_tensor(train)
    assert features.shape == (3042, 20)
    tokenizer = current.load_tokenizer()
    # 只对比分词，不把NLI模型参数用于新实验。
    repository = Path(get_model_cache_dir()) / 'models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli'
    revision = (repository / 'refs/main').read_text(encoding='utf-8').strip()
    nli_path = repository / 'snapshots' / revision
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_path, local_files_only=True)
    for name, frame in [('Train', train), ('Dev', dev)]:
        kwargs = {'truncation': True, 'padding': 'max_length', 'max_length': 128, 'return_tensors': 'pt'}
        raw = tokenizer(frame.text1.tolist(), frame.text2.tolist(), **kwargs)
        nli = nli_tokenizer(frame.text1.tolist(), frame.text2.tolist(), **kwargs)
        assert torch.equal(raw['input_ids'], nli['input_ids']), name
        assert torch.equal(raw['attention_mask'], nli['attention_mask']), name
    pairs = PairDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    assert len(pairs) == 1282
    assert train.label.value_counts().to_dict() == {
        'FORWARD_ENTAILMENT': 881, 'BACKWARD_ENTAILMENT': 881,
        'EQUIVALENCE': 802, 'NEGATIVE_OTHER': 478,
    }
    print('数据通过：官方3042/660、1282组Pair、20维原特征；191步Row + 81步Pair，无增强。')
    print('分词通过：原始base与旧NLI版在全部Train/Dev上的input_ids和attention_mask一致。')
    return tokenizer, train, features


def check_gpu(tokenizer, train, features):
    device = torch.device('cuda')
    current.set_seed(cfg.SEED)
    model = current.build_model(tokenizer, device)
    assert all(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.lexical_output.parameters()) == 804
    optimizer = current.build_optimizer(model)
    assert optimizer.param_groups[0]['lr'] == 2e-5 and optimizer.param_groups[1]['lr'] == 1e-3
    assert sum(p.numel() for p in optimizer.param_groups[1]['params']) == 804
    assert not optimizer.state
    del optimizer  # 不进行Adam更新；下面只检查真实前向、反向与梯度。
    row_ce, pair_ce = current.build_criteria(device)
    row_loader = DataLoader(DiCoDataset(train, tokenizer, features, 128), batch_size=16)
    pair_loader = DataLoader(PairDataset(train, tokenizer, features, 128), batch_size=16)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    ids, masks, values, labels = next(iter(row_loader))
    logits = model(ids.to(device), masks.to(device), values.to(device))
    loss = row_ce(logits, labels.to(device))
    loss.backward()
    assert torch.isfinite(loss) and logits.shape == (16, 4)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    ids0, masks0, values0, ids1, masks1, values1, targets = next(iter(pair_loader))
    first = model(ids0.to(device), masks0.to(device), values0.to(device))
    second = model(ids1.to(device), masks1.to(device), values1.to(device))
    loss = cfg.PAIR_WEIGHT * pair_ce(current.get_pair_logits(first, second), targets.to(device))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('GPU检查通过：本地原始bin权重可加载，全部参数可训练，804参数词汇分支及双学习率正确。')
    print('真实Row16行、Pair16组前后向均通过；峰值GiB：', round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print('峰值不含正式训练的Adam状态；没有执行optimizer.step。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    check_training_loops()
    tokenizer, train, features = check_data_and_tokenizer()
    if args.gpu:
        check_gpu(tokenizer, train, features)
    print('检查完成，没有启动训练。')
