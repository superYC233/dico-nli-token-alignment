"""仅检查分组和配置；不加载预训练权重、不前向、不反传、不执行 step。"""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from experiments.deberta_lexical_mixed_pair import config_slow_backbone as cfg
from experiments.deberta_lexical_mixed_pair.train_deberta_slow_backbone import build_optimizer
from experiments.deberta_lexical_stage_probe.model import DebertaDiCoModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embeddings = nn.Embedding(32, 16)
        self.encoder = nn.Linear(16, 16)


def check_optimizer_groups():
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    torch.manual_seed(42)
    model = DebertaDiCoModel(TinyBackbone(), tokenizer, feature_size=20, lexical_hidden=32)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = build_optimizer(model)
    rates = {group['name']: group['lr'] for group in optimizer.param_groups}
    assert rates == {'backbone': 5e-6, 'heads': 2e-5, 'lexical': 1e-3}
    groups = {group['name']: {id(p) for p in group['params']} for group in optimizer.param_groups}
    all_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        expected = 'backbone' if name.startswith('deberta_model.') else (
            'lexical' if name.startswith('lexical_output.') else 'heads'
        )
        assert id(parameter) in groups[expected], name
        assert parameter.requires_grad, name
        assert torch.equal(parameter, before[name]), name
    for prefix in ['output.', 'align_projection.', 'compare_layer.', 'match_output.']:
        selected = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        assert selected and all(id(p) in groups['heads'] for p in selected), prefix
    assert all(group['weight_decay'] == cfg.WEIGHT_DECAY for group in optimizer.param_groups)
    assert len(optimizer.state) == 0
    print('三个参数组无遗漏、无重复、全部可训练：', rates)
    print('模型参数未发生变化；没有 optimizer.step 或 backward。')


def check_protocol():
    reference = json.loads((cfg.EXPERIMENT_DIR / 'outputs/run01_mixed_pair/run_config.json').read_text(encoding='utf-8'))
    for key, value in [
        ('model_name', cfg.MODEL_NAME), ('seed', cfg.SEED), ('epochs', cfg.EPOCHS),
        ('use_lexical_features', cfg.USE_LEXICAL_FEATURES),
        ('lexical_hidden_size', cfg.LEXICAL_HIDDEN_SIZE),
        ('learning_rate', cfg.LEARNING_RATE), ('lexical_learning_rate', cfg.LEXICAL_LEARNING_RATE),
        ('weight_decay', cfg.WEIGHT_DECAY), ('row_weights', cfg.ROW_WEIGHTS),
        ('pair_weights', cfg.PAIR_WEIGHTS), ('pair_label_smoothing', cfg.PAIR_LABEL_SMOOTHING),
        ('pair_weight', cfg.PAIR_WEIGHT), ('group_batch_size', cfg.GROUP_BATCH_SIZE),
        ('dev_batch_size', cfg.DEV_BATCH_SIZE), ('max_length', cfg.MAX_LENGTH),
    ]:
        assert reference[key] == value, key
    assert cfg.SELECT_BY == 'independent'
    assert cfg.RUN_NAME == 'run04_slow_backbone'
    old = ast.parse((cfg.EXPERIMENT_DIR / 'train_deberta_mixed_pair.py').read_text(encoding='utf-8-sig'))
    new = ast.parse((cfg.EXPERIMENT_DIR / 'train_deberta_slow_backbone.py').read_text(encoding='utf-8-sig'))
    for name in ['compute_joint_loss', 'train_epoch', 'make_dev_loader']:
        a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), name
    print('损失、混合训练和 Dev 批次逻辑不变；其他超参数匹配 run01。')
    print('选模已明确改为 independent，使用旧 run01 的独立峰值作对照。')


if __name__ == '__main__':
    check_optimizer_groups()
    check_protocol()
    print('检查通过。未启动训练。')
