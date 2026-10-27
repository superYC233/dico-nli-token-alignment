"""检查本实验只改变 Pair CE，且实际梯度等于纯四分类 CE。"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from experiments.deberta_joint_pair_frozen import config as reference
from experiments.deberta_row_ce_frozen import config as cfg
from experiments.deberta_row_ce_frozen.train_deberta_row_ce import compute_joint_loss


class UnusedPairLoss:
    def __call__(self, *args):
        raise AssertionError('Pair CE 不应被调用。')


def function_source(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    # 进度条文字从“联合训练”改成“四分类训练”，不属于计算逻辑变化。
    for item in ast.walk(node):
        if isinstance(item, ast.keyword) and item.arg == 'desc':
            item.value = ast.Constant('progress label')
    return ast.dump(node, include_attributes=False)


def check_augmentation_data():
    """只检查数据和 batch；不构建模型、不训练、不做反向传播。"""
    import json
    import math
    import random
    import pandas as pd
    from torch.utils.data import DataLoader
    from experiments.deberta_joint_pair.data import GroupDataset, collate_groups
    from experiments.deberta_row_ce_frozen.scope_augmentation import build_train_variant, FAMILIES
    from experiments.deberta_minimal_contrasts.config import FAMILIES as DIAGNOSTIC_FAMILIES

    class ShapeTokenizer:
        def __call__(self, first, second, truncation, padding, max_length, return_tensors):
            shape = (len(first), max_length)
            return {'input_ids': torch.ones(shape, dtype=torch.long),
                    'attention_mask': torch.ones(shape, dtype=torch.long)}

    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    original = train.copy(deep=True)
    random_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    scope, scope_extra, _ = build_train_variant(train, dev, 'scope', cfg.SEED)
    copy, copy_extra, copy_audit = build_train_variant(train, dev, 'copy', cfg.SEED)
    none, _, _ = build_train_variant(train, dev, 'none', cfg.SEED)
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(none, original)
    assert random.getstate() == random_state
    assert len(scope) == len(copy) == 3242
    assert len(scope_extra) == len(copy_extra) == 200
    assert scope_extra.label.value_counts().to_dict() == copy_extra.label.value_counts().to_dict()
    assert scope_extra.label.tolist() == copy_extra.label.tolist()
    assert scope.instance_id.is_unique and copy.instance_id.is_unique

    training_nouns = {word for _, noun, foil in FAMILIES for word in [noun, foil]}
    diagnostic_nouns = {word.lower() for family in DIAGNOSTIC_FAMILIES for word in [family[2], family[4]]}
    assert not training_nouns & diagnostic_nouns
    lookup = train.set_index('instance_id')
    for index, row in copy_extra.iterrows():
        source = lookup.loc[copy_audit.iloc[index].control_source_instance_id]
        assert (row.text1, row.text2, row.label) == (source.text1, source.text2, source.label)

    loaders = []
    for frame in [scope, copy]:
        dataset = GroupDataset(frame, ShapeTokenizer(), cfg.MAX_LENGTH)
        assert len(dataset) == 1880 and dataset.pair_count == 1362
        loaders.append(DataLoader(dataset, batch_size=cfg.GROUP_BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_groups,
                                  generator=torch.Generator().manual_seed(cfg.SEED)))
    assert len(loaders[0]) == len(loaders[1]) == 235
    for first, second in zip(*loaders):
        for name in ['labels', 'pair_indices', 'pair_targets']:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        assert first['input_ids'].shape == second['input_ids'].shape
    assert torch.equal(torch_state, torch.get_rng_state())
    report = {'rows_per_arm': len(scope), 'extra_rows_per_arm': len(scope_extra),
              'groups_per_arm': 1880, 'steps_per_epoch': 235,
              'total_optimizer_steps_per_arm': 235 * cfg.EPOCHS,
              'extra_class_counts': scope_extra.label.value_counts().to_dict(),
              'matching_batch_labels_and_shapes': True,
              'training_executed': False}
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / 'budget_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: scope/copy rows, labels, groups, shuffled batch labels and shapes match; no training executed')


def main():
    for name in [
        'SEED', 'MODEL_NAME', 'EPOCHS', 'MAX_LENGTH', 'LEARNING_RATE', 'WEIGHT_DECAY',
        'FREEZE_EMBEDDINGS', 'FREEZE_FIRST_N_LAYERS', 'ROW_WEIGHTS', 'PAIR_WEIGHTS',
        'PAIR_LABEL_SMOOTHING', 'GROUP_BATCH_SIZE', 'DEV_BATCH_SIZE', 'SELECT_BY',
        'TRAIN_PATH', 'DEV_PATH', 'GOLD_PATH',
    ]:
        assert getattr(cfg, name) == getattr(reference, name), name
    assert cfg.PAIR_WEIGHT == 0 and reference.PAIR_WEIGHT == 0.5
    assert cfg.MODEL_DIR != reference.MODEL_DIR and cfg.RESULTS_DIR != reference.RESULTS_DIR

    old_script = reference.EXPERIMENT_DIR / 'train_deberta_frozen.py'
    new_script = cfg.EXPERIMENT_DIR / 'train_deberta_row_ce.py'
    for name in ['freeze_backbone', 'build_model', 'train_epoch', 'evaluate', 'make_dev_loader']:
        assert function_source(old_script, name) == function_source(new_script, name), name

    torch.manual_seed(42)
    # 一个 FB pair + 一个 N 单例；确认 N 和配对行都有四分类梯度。
    logits = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 3])
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    total, row_loss, pair_loss = compute_joint_loss(
        logits, labels, torch.tensor([[0, 1]]), torch.tensor([0]), row_ce, UnusedPairLoss(),
    )
    expected = row_ce(logits, labels)
    actual_grad = torch.autograd.grad(total, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(total, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert pair_loss.item() == 0 and total is row_loss
    assert (actual_grad.abs().sum(dim=1) > 0).all()
    print('PASS: same frozen recipe, Pair CE skipped, gradients exactly equal four-class CE')


if __name__ == '__main__':
    main()
