"""CPU随机微型ModernBERT验证；不下载权重、不运行正式训练。"""
import ast
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from experiments.modernbert_nli_base import config as cfg
from experiments.modernbert_nli_base.data import read_data, build_feature_tensor, DiCoDataset, PairDataset
from experiments.modernbert_nli_base.evaluation import get_pair_logits
from experiments.modernbert_nli_base.lexical_features import FEATURE_NAMES
from experiments.modernbert_nli_base.model import ModernBertDiCoModel


def main():
    torch.set_num_threads(2)
    train, dev = read_data()
    assert len(train) == 3042 and len(dev) == 660
    assert train.groupby('pair_id').size().eq(2).sum() == 1282
    # 本机已有large版tokenizer时，可先核对ModernBERT分隔格式，不加载large主干。
    tokenizer_path = cfg.MODEL_PATH if cfg.MODEL_PATH.exists() else Path('D:/BaiduNetdiskDownload/PretrainedModel/models/tasksource--ModernBERT-large-nli')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    features = build_feature_tensor(train)
    rows = DiCoDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    pairs = PairDataset(train, tokenizer, features, cfg.MAX_LENGTH)
    assert len(pairs) == 1282
    encoded = tokenizer(train.text1.tolist(), train.text2.tolist(), padding='max_length', truncation=True,
                        max_length=cfg.MAX_LENGTH, return_tensors='pt')
    ids, attention = encoded['input_ids'], encoded['attention_mask']
    is_sep = ids.eq(tokenizer.sep_token_id)
    count = is_sep.long().cumsum(1)
    valid = attention.bool() & ~is_sep & ids.ne(tokenizer.cls_token_id)
    # 与原版Token Alignment相同的CLS A SEP B SEP格式。
    for index in range(len(train)):
        assert is_sep[index].sum() == 2
        expected = encoded.sequence_ids(index)
        assert (valid[index] & count[index].eq(0)).tolist() == [value == 0 for value in expected]
        assert (valid[index] & count[index].eq(1)).tolist() == [value == 1 for value in expected]

    small = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=256,
                            pad_token_id=tokenizer.pad_token_id, cls_token_id=tokenizer.cls_token_id,
                            sep_token_id=tokenizer.sep_token_id, local_attention=64, global_attn_every_n_layers=2)
    small._attn_implementation = 'sdpa'
    backbone = ModernBertModel(small)
    torch.manual_seed(42)
    state = torch.get_rng_state()
    tree = ast.parse((cfg.PROJECT_ROOT / 'experiments/deberta_original/train.py').read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'debertaDiCoModel')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'my_model': copy.deepcopy(backbone), 'my_tokenizer': tokenizer}
    exec(compile(module, 'original_model_only', 'exec'), namespace)
    original = namespace['debertaDiCoModel']().eval()
    torch.set_rng_state(state)
    model = ModernBertDiCoModel(backbone, tokenizer, len(FEATURE_NAMES), use_lexical_features=False).eval()
    assert model.state_dict().keys() == original.state_dict().keys()
    for key in original.state_dict():
        torch.testing.assert_close(original.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
    sample = tokenizer(train.text1[:2].tolist(), train.text2[:2].tolist(), padding=True, return_tensors='pt')
    sample = {key: value for key, value in sample.items() if key in ['input_ids', 'attention_mask']}
    logits = model(**sample, lexical_features=features[:2])
    torch.testing.assert_close(logits, original(**sample), rtol=0, atol=0)
    pair_logits = get_pair_logits(logits[:1], logits[1:])
    loss = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))(logits, torch.tensor([0, 1]))
    loss = loss + cfg.PAIR_WEIGHT * nn.CrossEntropyLoss(label_smoothing=cfg.PAIR_LABEL_SMOOTHING)(pair_logits, torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
    report = {'train_rows': len(train), 'dev_rows': len(dev), 'real_pairs': len(pairs),
              'lexical_default': cfg.USE_LEXICAL_FEATURES, 'original_model_formula_identical': True,
              'all_train_token_masks_match_sequence_ids': True, 'tiny_modernbert_sdpa_forward_backward': True,
              'tokenizer_used_for_check': str(tokenizer_path), 'real_backbone_loaded': False,
              'optimizer_created': False, 'training_performed': False}
    folder = cfg.EXPERIMENT_DIR / 'outputs' / 'setup_check'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
