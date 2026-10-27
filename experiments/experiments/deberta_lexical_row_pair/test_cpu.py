"""只做数据、静态代码与 CPU 损失检查，不加载模型、不反传、不训练。"""
import ast
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.deberta_lexical_row_pair import config as cfg
from experiments.deberta_lexical_row_pair.data import read_data, build_feature_tensor, PairDataset
from experiments.deberta_lexical_row_pair.train_deberta_row_pair import compute_pair_stage_loss
from experiments.deberta_lexical_stage_probe.data import PairDataset as OriginalPairDataset


class TinyTokenizer:
    def __call__(self, first, second, **kwargs):
        a = 3 + sum(map(ord, first)) % 25
        b = 3 + sum(map(ord, second)) % 25
        return {
            'input_ids': torch.tensor([[1, a, 2, b, 2, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0]]),
        }


def function_tree(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node)


def check_unchanged_parts():
    source_dir = PROJECT_ROOT / 'experiments/deberta_lexical_stage_probe'
    hashes = json.loads((cfg.EXPERIMENT_DIR / 'source_hashes.json').read_text(encoding='utf-8-sig'))
    for record in hashes:
        actual = hashlib.sha256((source_dir / record['Name']).read_bytes()).hexdigest()
        assert actual == record['SHA256'].lower(), record['Name']

    for name in ['model.py', 'lexical_features.py', 'evaluation.py']:
        old = (source_dir / name).read_text(encoding='utf-8-sig')
        new = (cfg.EXPERIMENT_DIR / name).read_text(encoding='utf-8-sig')
        new = new.replace('experiments.deberta_lexical_row_pair', 'experiments.deberta_lexical_stage_probe')
        assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(new)), name

    old_train = source_dir / 'train_deberta_stage_probe.py'
    new_train = cfg.EXPERIMENT_DIR / 'train_deberta_row_pair.py'
    for name in ['set_seed', 'build_optimizer', 'train_rows', 'evaluate_without_changing_rng']:
        assert function_tree(old_train, name) == function_tree(new_train, name), name

    reference = json.loads((cfg.REFERENCE_RESULTS_DIR / 'run_config.json').read_text(encoding='utf-8'))
    for key, value in [
        ('model_name', cfg.MODEL_NAME), ('seed', cfg.SEED),
        ('lexical_hidden_size', cfg.LEXICAL_HIDDEN_SIZE),
        ('lexical_learning_rate', cfg.LEXICAL_LEARNING_RATE),
        ('learning_rate', cfg.LEARNING_RATE), ('weight_decay', cfg.WEIGHT_DECAY),
        ('row_weights', cfg.ROW_WEIGHTS), ('pair_weights', cfg.PAIR_WEIGHTS),
        ('pair_weight', cfg.PAIR_WEIGHT), ('pair_label_smoothing', cfg.PAIR_LABEL_SMOOTHING),
        ('epochs', cfg.EPOCHS), ('row_batch_size', cfg.ROW_BATCH_SIZE),
        ('pair_batch_size', cfg.PAIR_BATCH_SIZE), ('dev_batch_size', cfg.DEV_BATCH_SIZE),
        ('max_length', cfg.MAX_LENGTH),
    ]:
        assert reference[key] == value, key
    print('旧文件未修改；模型、特征、优化器、第一阶段、推理与主要参数保持一致。')


def check_pair_labels():
    train, dev = read_data()
    features = build_feature_tensor(train)
    tokenizer = TinyTokenizer()
    pairs = PairDataset(train, tokenizer, features)
    original = OriginalPairDataset(train, tokenizer, features)
    assert pairs.pairs == original.pairs
    assert len(train) == 3042 and len(dev) == 660 and len(pairs) == 1282
    combinations = [(0, 1), (1, 0), (2, 2)]
    for index in range(len(pairs)):
        new_item = pairs[index]
        old_item = original[index]
        assert all(torch.equal(new_item[i], old_item[i]) for i in range(7))
        target, label0, label1 = new_item[6:]
        assert combinations[target.item()] == (label0.item(), label1.item())
    batch = next(iter(DataLoader(pairs, batch_size=16, shuffle=False)))
    assert len(batch) == 9
    assert batch[6].shape == batch[7].shape == batch[8].shape == (16,)
    assert batch[7].dtype == batch[8].dtype == torch.long
    print('1282 个真实配对的方向、特征、三组合标签及两行金标均一致；批量形状通过。')


@torch.inference_mode()
def check_losses():
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS, dtype=torch.float64))
    pair_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.PAIR_WEIGHTS, dtype=torch.float64),
        label_smoothing=cfg.PAIR_LABEL_SMOOTHING,
    )
    labels0, labels1, targets = [torch.tensor([value]) for value in (0, 1, 0)]
    first = torch.tensor([[8.0, 1.0, 0.0, -5.0]], dtype=torch.float64)
    second = torch.tensor([[1.0, 8.0, 0.0, -5.0]], dtype=torch.float64)

    def loss_for(a, b):
        return compute_pair_stage_loss(
            a, b, labels0, labels1, targets, row_criterion, pair_criterion,
        )

    base_loss, base_row, base_pair = loss_for(first, second)
    assert torch.allclose(base_loss, base_row + cfg.PAIR_WEIGHT * base_pair)
    assert torch.allclose(base_row, (row_criterion(first, labels0) + row_criterion(second, labels1)) / 2)

    # 旧 Pair CE 看不到 N 分数上涨；新增 Row CE 必须提高损失。
    n_shift = torch.tensor([[0.0, 0.0, 0.0, 105.0]], dtype=torch.float64)
    n_loss, n_row, n_pair = loss_for(first + n_shift, second + n_shift)
    assert torch.allclose(n_pair, base_pair)
    assert n_row > base_row and n_loss > base_loss

    # 旧 Pair CE 看不到所有行 F 降、B 升的抵消；新增 Row CE 必须提高损失。
    fb_shift = torch.tensor([[-10.0, 10.0, 0.0, 0.0]], dtype=torch.float64)
    fb_loss, fb_row, fb_pair = loss_for(first + fb_shift, second + fb_shift)
    assert torch.allclose(fb_pair, base_pair)
    assert fb_row > base_row and fb_loss > base_loss

    # 用差分检查损失对 N 分数的方向，不调用 autograd 或 backward。
    n_first = first + n_shift
    step = torch.tensor([[0.0, 0.0, 0.0, 1e-4]], dtype=torch.float64)
    assert loss_for(n_first + step, second)[0] > loss_for(n_first - step, second)[0]

    # FB、BF、EE 混合批次，验证两行 CE 的带权归一化及两方向标签。
    a = torch.tensor([[8., 1., 0., -5.], [1., 8., 0., -5.], [0., 1., 8., -5.]], dtype=torch.float64)
    b = torch.tensor([[1., 8., 0., -5.], [8., 1., 0., -5.], [1., 0., 8., -5.]], dtype=torch.float64)
    gold_a, gold_b = torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2])
    pair_targets = torch.tensor([0, 1, 2])
    total, row, paired = compute_pair_stage_loss(
        a, b, gold_a, gold_b, pair_targets, row_criterion, pair_criterion,
    )
    reference_row = row_criterion(torch.cat([a, b]), torch.cat([gold_a, gold_b]))
    assert torch.allclose(row, reference_row)
    assert torch.allclose(total, reference_row + cfg.PAIR_WEIGHT * paired)
    swapped = compute_pair_stage_loss(
        b, a, gold_b, gold_a, torch.tensor([1, 0, 2]), row_criterion, pair_criterion,
    )
    assert all(torch.allclose(x, y) for x, y in zip((total, row, paired), swapped))
    print('损失公式通过；新增四分类 CE 能约束 N 过高和 F/B 整体偏移这两个旧盲区。')


if __name__ == '__main__':
    check_unchanged_parts()
    check_pair_labels()
    check_losses()
    print('全部通过。没有加载预训练模型，没有反向传播，没有执行训练。')
