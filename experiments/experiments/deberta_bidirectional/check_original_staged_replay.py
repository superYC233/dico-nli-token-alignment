"""CPU检查：原Row梯度、同批两项目标、真实Train曝光和完整权重重载；不训练。"""
import argparse
import copy
import json
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

from experiments.deberta_bidirectional import train_original_staged_replay as cfg
from experiments.deberta_stage_average import train_deberta_stage_average as legacy


class ToyRows(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        mapping = {'FORWARD_ENTAILMENT': 0, 'BACKWARD_ENTAILMENT': 1,
                   'EQUIVALENCE': 2, 'NEGATIVE_OTHER': 3}
        self.labels = torch.tensor(self.frame.label.map(mapping).tolist())
        self.encoded = {'input_ids': torch.tensor([
            [1, 10 + i % 30, 2, 45 + i % 30, 2, 0, 0, 0] for i in range(len(frame))
        ])}
        self.encoded['attention_mask'] = self.encoded['input_ids'].ne(0).long()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {'input_ids': self.encoded['input_ids'][index],
                'attention_mask': self.encoded['attention_mask'][index],
                'label': self.labels[index]}


class GradientRecorder:
    # step只记录梯度，不执行任何参数更新。
    def __init__(self, model):
        self.model = model
        self.states = []
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)
        self.states.append(torch.get_rng_state())

    def step(self):
        self.gradients.append({name: parameter.grad.detach().clone()
                               for name, parameter in self.model.named_parameters()
                               if parameter.grad is not None})


def toy_model():
    backbone = DebertaV2Model(DebertaV2Config(
        vocab_size=96, hidden_size=24, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=3, max_position_embeddings=128, pad_token_id=0,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        relative_attention=False,
    ))
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    return cfg.DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)


def criteria():
    row = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS),
                               label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    return row, pair


def check_coverage(train):
    rows = ToyRows(train)
    pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(pairs.pairs, batch_size=cfg.PAIR_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(123))
    batches = list(cfg.replay_batches(rows, loader, torch.Generator().manual_seed(456)))
    source = torch.cat([batch['source_rows'] for batch in batches])
    assert len(batches) == 87 and len(pairs) == 1380
    assert torch.equal(source.sort().values, torch.arange(len(rows)))
    assert sum(batch['n_count'] for batch in batches) == 530
    assert sum(batch['pair_count'] for batch in batches) == 1380
    assert max(len(batch['labels']) for batch in batches) == 39
    assert sum(len(batch['labels']) for batch in batches) == 3290
    targets = torch.tensor([[0, 1], [1, 0], [2, 2]])
    for batch in batches:
        count = batch['pair_count']
        expected = targets[batch['pair_targets']]
        assert torch.equal(batch['labels'][:count], expected[:, 0])
        assert torch.equal(batch['labels'][count:2 * count], expected[:, 1])
        assert batch['labels'][2 * count:].eq(3).all()
        if count == cfg.PAIR_BATCH_SIZE:
            assert batch['n_count'] in [6, 7]
    print('真实Train曝光检查通过：87批，1380个Pair，530条N，3290行恰好各一次，最多39行/批。')


def check_losses(train, folder):
    all_rows = ToyRows(train)
    all_pairs = cfg.RealPairDataset(all_rows)
    subset = [index for pair in all_pairs.pairs[:2] for index in pair[:2]]
    subset += torch.where(all_rows.labels.eq(3))[0][:3].tolist()
    rows = ToyRows(train.iloc[subset])
    pairs = cfg.RealPairDataset(rows)
    row_criterion, pair_criterion = criteria()
    pair_loader = DataLoader(pairs.pairs, batch_size=1, shuffle=False,
                             generator=torch.Generator().manual_seed(4))
    device = torch.device('cpu')

    # 新Row阶段必须与原Row CE逐参数梯度相同。
    first = toy_model()
    second = copy.deepcopy(first)
    old_recorder, new_recorder = GradientRecorder(first), GradientRecorder(second)
    row_batches = list(DataLoader(rows, batch_size=3, shuffle=False, collate_fn=cfg.collate_rows))
    torch.manual_seed(88)
    legacy.train_rows(first, row_batches, old_recorder, row_criterion, device)
    torch.manual_seed(88)
    record = cfg.train_rows(second, row_batches, new_recorder, row_criterion, device)
    assert record['row_ce_rows'] == len(rows)
    for old, new in zip(old_recorder.gradients, new_recorder.gradients):
        assert old.keys() == new.keys()
        for name in old:
            torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)

    # Pair本身对N列无梯度；四分类项同时压低配对行的N误报、监督N正例。
    targets = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 3, 3])
    logits = torch.zeros(len(labels), 4, requires_grad=True)
    loss, row_loss, pair_loss = cfg.compute_replay_loss(
        logits, labels, targets, row_criterion, pair_criterion,
    )
    pair_gradient = torch.autograd.grad(pair_loss, logits, retain_graph=True)[0]
    row_gradient = torch.autograd.grad(row_loss, logits, retain_graph=True)[0]
    assert pair_gradient[:, 3].eq(0).all()
    assert row_gradient[:6, 3].gt(0).all() and row_gradient[6:, 3].lt(0).all()
    manual = row_criterion(logits, labels) + .5 * pair_criterion(torch.stack([
        logits[:3, 0] + logits[3:6, 1], logits[:3, 1] + logits[3:6, 0],
        logits[:3, 2] + logits[3:6, 2],
    ], dim=1), targets)
    torch.testing.assert_close(loss, manual, rtol=0, atol=0)

    # 真正跑训练函数的前向/反向，但recorder.step不更新参数。
    model = toy_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    recorder = GradientRecorder(model)
    result, exposure = cfg.train_pair_replay(
        model, rows, pair_loader, torch.Generator().manual_seed(5), recorder,
        row_criterion, pair_criterion, device,
    )
    assert result['pair_stage_ce_rows'] == 7 and result['replayed_n_rows'] == 3
    assert result['pair_ce_groups'] == 2 and len(recorder.gradients) == 2
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    source = torch.tensor([int(value) for value in exposure[0]['source_rows'].split('|')])
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[0])
    manual_batch = cfg.trim_padding({'input_ids': rows.encoded['input_ids'][source],
                                     'attention_mask': rows.encoded['attention_mask'][source]})
    output = model(manual_batch['input_ids'], manual_batch['attention_mask'], None)
    expected = row_criterion(output, rows.labels[source]) + .5 * pair_criterion(
        torch.stack([output[:1, 0] + output[1:2, 1], output[:1, 1] + output[1:2, 0],
                     output[:1, 2] + output[1:2, 2]], dim=1), torch.tensor([pairs.pairs[0][2]]),
    )
    expected.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, recorder.gradients[0][name], rtol=1e-5, atol=1e-6)

    model.eval()
    trimmed = cfg.trim_padding({'input_ids': rows.encoded['input_ids'],
                                'attention_mask': rows.encoded['attention_mask']})
    with torch.inference_mode():
        expected = model(rows.encoded['input_ids'], rows.encoded['attention_mask'], None)
        actual_trimmed = model(trimmed['input_ids'], trimmed['attention_mask'], None)
    torch.testing.assert_close(expected, actual_trimmed, rtol=1e-5, atol=1e-6)
    assert trimmed['input_ids'].shape[1] == 5
    checkpointed = copy.deepcopy(model)
    checkpointed.deberta_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    model.train()
    checkpointed.train()
    model.zero_grad(set_to_none=True)
    checkpointed.zero_grad(set_to_none=True)
    for current in [model, checkpointed]:
        torch.manual_seed(789)
        z = current(trimmed['input_ids'], trimmed['attention_mask'], None)
        row_criterion(z, rows.labels).backward()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)
    model.eval()
    cfg.save_weights(model, folder / 'toy.pth')
    restored = toy_model().eval()
    cfg.load_weights(restored, folder / 'toy.pth')
    with torch.inference_mode():
        actual = restored(rows.encoded['input_ids'], rows.encoded['attention_mask'], None)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    scores = pd.DataFrame([{'method': 'independent', 'weighted_f1': .86},
                           {'method': 'pair_aware', 'weighted_f1': .93}])
    selected = cfg.selection_record(3, scores)
    assert abs(selected['selection_score'] - min(.86/.87, .93/.94)) < 1e-12
    assert not selected['targets_met'] and selected['epoch'] == 3
    assert all((ROOT / path).is_file() for path in cfg.source_hashes())
    print('CPU梯度/保存检查通过：原Row梯度一致，完整四分类+0.5Pair同一次更新，N有直接监督，完整权重可重载。')


def check_real(train, folder):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=cfg.get_model_cache_dir(), revision=cfg.BASE_REVISION,
        local_files_only=True,
    )
    rows = cfg.RowDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.RealPairDataset(rows)
    loader = DataLoader(pairs.pairs, batch_size=4, shuffle=False)
    batch = next(cfg.replay_batches(rows, loader, torch.Generator().manual_seed(1)))
    assert batch['pair_count'] == 4 and batch['n_count'] == 1
    model = cfg.build_model(tokenizer)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 184001800
    assert all(p.requires_grad for p in model.parameters())
    row_criterion, pair_criterion = criteria()
    model.train()
    logits = model(batch['input_ids'], batch['attention_mask'], None)
    loss, _, _ = cfg.compute_replay_loss(
        logits, batch['labels'], batch['pair_targets'], row_criterion, pair_criterion,
    )
    loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad[3].abs().sum() > 0
    assert model.deberta_model.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.deberta_model.embeddings.word_embeddings.weight[1, 0].add_(.001)
    model.eval()
    with torch.inference_mode():
        expected = model(batch['input_ids'], batch['attention_mask'], None)
    model_dir = folder / 'real'
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    (model_dir / 'run_config.json').write_text(json.dumps({
        'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
    }), encoding='utf-8')
    cfg.save_weights(model, model_dir / 'best_model.pth')
    del logits, loss, model
    restored, _, _ = cfg.load_saved_model(model_dir, torch.device('cpu'))
    with torch.inference_mode():
        actual = restored(batch['input_ids'], batch['attention_mask'], None)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print('真实NLI检查通过：184001800参数、4组Pair+1条N的两项梯度、完整encoder重载；没有优化器参数更新。')


def main(real=False):
    torch.set_num_threads(4)
    train, _, _ = cfg.read_data()
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'original_staged_replay_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_coverage(train)
        check_losses(train, folder)
        if real:
            check_real(train, folder)
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('original_staged_replay_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
