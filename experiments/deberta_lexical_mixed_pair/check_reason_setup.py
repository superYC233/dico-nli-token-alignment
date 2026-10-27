"""检查标注、token映射、梯度和保存读取；不创建优化器、不训练。"""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.deberta_lexical_mixed_pair import config_reason as cfg
from experiments.deberta_lexical_mixed_pair.reason_data import (
    load_reason_train, ReasonGroupDataset, collate_reason_groups, encode_reason_targets,
)
from experiments.deberta_lexical_mixed_pair.reason_model import ReasonDebertaModel, evidence_loss
from experiments.deberta_lexical_mixed_pair.train_deberta_reason import load_tokenizer
from experiments.deberta_lexical_stage_probe.data import build_feature_tensor
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(32, 16)
        self.projection = nn.Linear(16, 16)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        hidden = self.projection(self.embedding(input_ids)).tanh()
        return SimpleNamespace(last_hidden_state=hidden)


def check_model():
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    torch.manual_seed(19)
    original = DebertaDiCoModel(TinyBackbone(), tokenizer, 20)
    original_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(19)
    model = ReasonDebertaModel(TinyBackbone(), tokenizer, 20)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for name, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    ids = torch.tensor([[1, 10, 11, 2, 12, 2, 0, 0], [1, 12, 2, 10, 11, 2, 0, 0]])
    mask = ids.ne(0).long()
    features = torch.zeros(2, 20)
    state = torch.random.get_rng_state().clone()
    expected = original(ids, mask, features)
    torch.random.set_rng_state(state)
    logits, evidence = model(ids, mask, features, return_evidence=True)
    assert torch.equal(logits, expected)
    assert model.deberta_model.forward_calls == 1
    targets = torch.tensor([[-100, 0, 1, -100, 0, -100, -100, -100],
                            [-100, 0, -100, 0, 1, -100, -100, -100]])
    cue_loss = evidence_loss(evidence, targets)
    gradients = torch.autograd.grad(cue_loss, [model.evidence_head.weight, model.deberta_model.embedding.weight])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    empty = evidence_loss(evidence, torch.full_like(targets, -100))
    assert empty.item() == 0 and torch.isfinite(empty)
    mixed_logits = torch.cat([evidence.detach(), torch.randn_like(evidence)])
    mixed_targets = torch.cat([targets, torch.full_like(targets, -100)])
    assert torch.equal(evidence_loss(mixed_logits, mixed_targets), evidence_loss(evidence.detach(), targets))
    # 权重0时，原有分类梯度与原模型相同。
    original.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    original.eval()
    model.eval()
    labels = torch.tensor([0, 1])
    nn.functional.cross_entropy(original(ids, mask, features), labels).backward()
    nn.functional.cross_entropy(model(ids, mask, features), labels).backward()
    for name, parameter in original.named_parameters():
        other = dict(model.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name
    assert model.evidence_head.weight.grad is None
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = ReasonDebertaModel(TinyBackbone(), tokenizer, 20)
    restored.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
    restored.eval()
    assert torch.equal(model(ids, mask, features), restored(ids, mask, features))
    return {'single_encoder_forward': True, 'original_initialization_unchanged': True,
            'auxiliary_gradient_reaches_encoder': True, 'baseline_classification_gradient_unchanged': True,
            'empty_annotation_loss_zero': True, 'inference_without_reason': True,
            'checkpoint_roundtrip': True, 'test_backbone': 'tiny CPU module; not a real-model accuracy test'}


def main():
    train = load_reason_train()
    original = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    assert train[original.columns].equals(original)
    tokenizer = load_tokenizer({'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION})
    features = build_feature_tensor(train)
    dataset = ReasonGroupDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    targets = dataset.evidence_targets
    expected_evidence = int(train.has_evidence.sum())
    assert len(train) == 3042 and train.has_reason.all()
    assert targets[~torch.tensor(train.has_evidence.tolist())].eq(-100).all()
    special = ((dataset.encoded['input_ids'] == tokenizer.cls_token_id) |
               (dataset.encoded['input_ids'] == tokenizer.sep_token_id) |
               dataset.encoded['attention_mask'].eq(0))
    assert targets[special].eq(-100).all()
    assert targets[torch.tensor(train.has_evidence.tolist())].eq(1).any(dim=1).all()
    changed = train.copy()
    changed['reason'] = 'Changed annotation text must not enter the input.'
    encoded, changed_targets = encode_reason_targets(changed, tokenizer, cfg.MAX_LENGTH)
    assert torch.equal(encoded['input_ids'], dataset.encoded['input_ids'])
    assert torch.equal(changed_targets, targets)
    loader = DataLoader(dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
                        collate_fn=collate_reason_groups, generator=torch.Generator().manual_seed(cfg.SEED))
    rows = []
    pairs = 0
    annotated_seen = 0
    for batch in loader:
        rows.extend(batch['source_rows'].tolist())
        pairs += len(batch['pair_targets'])
        annotated_seen += int(batch['evidence_targets'].ne(-100).any(dim=1).sum())
    assert sorted(rows) == list(range(3042))
    assert pairs == 1282 and len(loader) == 220 and annotated_seen == expected_evidence
    result = {'rows': len(train), 'rows_with_reason': int(train.has_reason.sum()),
              'evidence_supervised_rows': annotated_seen, 'steps_per_epoch': len(loader),
              'pairs_per_epoch': pairs, 'original_text_and_labels_unchanged': True,
              'reason_text_absent_from_model_inputs': True, 'token_spans_valid': True,
              'optimizer_created': False, 'training_performed': False, **check_model()}
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.PREPARED_DIR / 'setup_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
