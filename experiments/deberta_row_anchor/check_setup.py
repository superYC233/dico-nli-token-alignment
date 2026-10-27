"""CPU检查：缓存行对应、四类KL梯度、零权重对照；不加载或训练DeBERTa。"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import GroupDataset
from experiments.deberta_manual_aug.train_relation_manual import train_staged
from experiments.deberta_row_anchor import train_deberta_row_anchor as cfg


HERE = Path(__file__).resolve().parent
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']


def check_data():
    train, dev, reference = cfg.read_data()
    for name, value in [('model_name', cfg.MODEL_NAME), ('seed', cfg.SEED), ('epochs', cfg.EPOCHS),
                        ('learning_rate', cfg.LEARNING_RATE), ('weight_decay', cfg.WEIGHT_DECAY),
                        ('row_batch_size', cfg.ROW_BATCH_SIZE), ('pair_batch_size', cfg.PAIR_BATCH_SIZE),
                        ('dev_batch_size', cfg.DEV_BATCH_SIZE), ('max_length', cfg.MAX_LENGTH),
                        ('row_weights', cfg.ROW_WEIGHTS), ('pair_weights', cfg.PAIR_WEIGHTS),
                        ('pair_weight', cfg.PAIR_WEIGHT), ('pair_label_smoothing', cfg.PAIR_LABEL_SMOOTHING)]:
        assert reference[name] == value, name
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    groups = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.AnchoredPairDataset(groups.rows)
    assert len(train) == 3290 and len(dev) == 660 and len(pairs) == 1380
    loader = DataLoader(pairs, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(92))
    visited = []
    for ids0, mask0, ids1, mask1, targets, indices in loader:
        torch.testing.assert_close(ids0, groups.rows.encoded['input_ids'][indices[:, 0]])
        torch.testing.assert_close(ids1, groups.rows.encoded['input_ids'][indices[:, 1]])
        assert groups.rows.labels[indices].ne(3).all(), 'N单例不能混进Pair保持。'
        for first, second in indices.tolist():
            assert train.iloc[first].text1 == train.iloc[second].text2
            assert train.iloc[first].text2 == train.iloc[second].text1
        visited.extend(indices.flatten().tolist())
    assert len(visited) == len(set(visited)) == 2760
    saved = pd.read_csv(cfg.REFERENCE_DIR / 'error_analysis/train_readout.csv', keep_default_na=False)
    # 用已经保存的Row阶段分数检查过滤覆盖率，不读取Dev作为缓存。
    row_path = ROOT / 'experiments/deberta_stage_average/outputs/run01_late_average/epoch_2/after_row/train/predictions_with_scores.csv'
    row = pd.read_csv(row_path, keep_default_na=False)
    assert row.instance_id.tolist() == train.instance_id.tolist() == saved.instance_id.tolist()
    logits = torch.tensor(row[[f'logit_{x}' for x in cfg.LABELS]].values, dtype=torch.float32)
    correct = logits.argmax(dim=1).eq(groups.rows.labels)
    assert int(correct.sum()) == 3214
    assert int((correct & groups.rows.labels.ne(3)).sum()) == 2735
    assert int((correct & groups.rows.labels.eq(3)).sum()) == 479
    assert (cfg.REFERENCE_DIR / 'epoch_2/summary.csv').is_file()
    print('数据/参数一致，打乱后的1380组缓存索引正确，N单例未进入Pair。')


def check_keep_loss():
    teacher = torch.tensor([[8., 0., 0., -6.], [0., 8., 0., -6.], [0., 0., 8., -6.]], requires_grad=True)
    student = teacher.detach().clone()
    student[:, 3] += 20
    student.requires_grad_()
    correct = torch.tensor([True, True, False])
    loss = cfg.compute_keep_loss(student, teacher, correct, cfg.TEMPERATURE)
    teacher_p = F.softmax(teacher.detach()[correct] / cfg.TEMPERATURE, dim=-1)
    student_log = F.log_softmax(student[correct] / cfg.TEMPERATURE, dim=-1)
    manual = (teacher_p * (teacher_p.log() - student_log)).sum(dim=1).mean() * cfg.TEMPERATURE ** 2
    torch.testing.assert_close(loss, manual)
    student_grad, teacher_grad = torch.autograd.grad(loss, (student, teacher), allow_unused=True)
    assert teacher_grad is None, '缓存目标不能接收梯度。'
    assert student_grad[:2, 3].gt(0).all(), 'N过高应受到直接惩罚。'
    assert student_grad[2].eq(0).all(), 'Row答错行不应被保持约束锁住。'
    expected = cfg.TEMPERATURE * (F.softmax(student.detach()[correct] / cfg.TEMPERATURE, dim=-1) - teacher_p) / 2
    torch.testing.assert_close(student_grad[correct], expected)

    empty = cfg.compute_keep_loss(student, teacher, torch.zeros(3, dtype=torch.bool), cfg.TEMPERATURE)
    assert empty.item() == 0
    assert torch.autograd.grad(empty, student)[0].eq(0).all()
    identical = cfg.compute_keep_loss(teacher.detach(), teacher, torch.ones(3, dtype=torch.bool), cfg.TEMPERATURE)
    assert abs(identical.item()) < 1e-5

    # F加a/B减a是三组合Pair CE看不到的偏移，四类保持必须看得到。
    shifted = teacher.detach() + torch.tensor([2., -2., 0., 0.])
    old_pair = cfg.get_pair_logits(teacher[:1], teacher[1:2])
    new_pair = cfg.get_pair_logits(shifted[:1], shifted[1:2])
    torch.testing.assert_close(old_pair, new_pair)
    assert cfg.compute_keep_loss(shifted, teacher, torch.ones(3, dtype=torch.bool), cfg.TEMPERATURE) > 0
    print('KL方向/T²/正确行过滤/缓存无梯度均正确；能够惩罚N漂移与F/B整体偏移。')


def check_batch_order():
    first = torch.tensor([[3., 1., 0., -1.], [1., 3., 0., -1.]], requires_grad=True)
    second = torch.tensor([[1., 3., 0., -1.], [3., 1., 0., -1.]], requires_grad=True)
    indices = torch.tensor([[7, 2], [3, 6]])
    cache = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 8
    cache = cache + torch.arange(8, dtype=torch.float32).unsqueeze(1) * torch.tensor([1., -1., .5, -.5])
    correct = torch.tensor([False, True, True, True, False, False, True, True])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    targets = torch.tensor([0, 1])
    total, pair_loss, keep_loss = cfg.compute_pair_stage_loss(
        first, second, targets, indices, cache, correct, criterion, cfg.KEEP_WEIGHT, cfg.TEMPERATURE)
    expected_cache = torch.stack([cache[7], cache[3], cache[2], cache[6]])
    expected_keep = cfg.compute_keep_loss(torch.cat([first, second]), expected_cache,
                                         torch.ones(4, dtype=torch.bool), cfg.TEMPERATURE)
    torch.testing.assert_close(keep_loss, expected_keep)
    torch.testing.assert_close(total, cfg.PAIR_WEIGHT * pair_loss + cfg.KEEP_WEIGHT * expected_keep)
    zero, original_pair, _ = cfg.compute_pair_stage_loss(
        first, second, targets, indices, cache, correct, criterion, 0, cfg.TEMPERATURE)
    torch.testing.assert_close(zero, cfg.PAIR_WEIGHT * original_pair)
    print('批内第一行/第二行拼接顺序与缓存一致；总损失权重正确。')


class ToyRows(Dataset):
    def __init__(self):
        self.frame = pd.DataFrame({'instance_id': [f'check_{i}' for i in range(5)],
                                  'pair_id': ['FB', 'FB', 'EE', 'EE', 'N'],
                                  'text1': ['a', 'b', 'c', 'd', 'e'],
                                  'text2': ['b', 'a', 'd', 'c', 'f'],
                                  'label': [FULL_LABELS[i] for i in [0, 1, 2, 2, 3]]})
        self.labels = torch.tensor([0, 1, 2, 2, 3])
        self.ids = torch.arange(20).reshape(5, 4)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'input_ids': self.ids[index], 'attention_mask': torch.ones(4, dtype=torch.long),
                'label': self.labels[index]}


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(8, 4)

    def forward(self, input_ids, attention_mask, features):
        return self.output(self.dropout(self.embedding(input_ids).mean(dim=1)))


def toy_run(anchor):
    cfg.set_seed(cfg.SEED)
    rows = ToyRows()
    model = ToyModel()
    row_loader = DataLoader(rows, batch_size=2, shuffle=True)
    pairs = cfg.AnchoredPairDataset(rows) if anchor else cfg.RealPairDataset(rows)
    pair_loader = DataLoader(pairs, batch_size=2, shuffle=True)
    cache_loader = DataLoader(rows, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    for _ in range(2):
        if anchor:
            cfg.train_rows(model, row_loader, optimizer, row_ce, torch.device('cpu'))
            before = torch.get_rng_state().clone()
            cache, correct = cfg.cache_train_logits(model, cache_loader, rows.labels, torch.device('cpu'))
            assert torch.equal(before, torch.get_rng_state()) and model.training
            assert not cache.requires_grad and not torch.is_inference(cache)
            assert torch.equal(correct, cache.argmax(dim=1).eq(rows.labels))
            cfg.train_pairs(model, pair_loader, optimizer, pair_ce, cache, correct,
                            torch.device('cpu'), keep_weight=0)
        else:
            train_staged(model, row_loader, pair_loader, optimizer, row_ce, pair_ce, torch.device('cpu'))
    return model.state_dict(), optimizer.state_dict(), torch.get_rng_state()


def check_zero_weight():
    old_weights, old_optim, old_rng = toy_run(anchor=False)
    new_weights, new_optim, new_rng = toy_run(anchor=True)
    for name in old_weights:
        torch.testing.assert_close(new_weights[name], old_weights[name], rtol=0, atol=0)
    for index, state in old_optim['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(new_optim['state'][index][name], value, rtol=0, atol=0)
    assert torch.equal(old_rng, new_rng)
    print('KEEP_WEIGHT=0：合成网络参数/AdamW状态/随机数与原两阶段流程逐项一致。')


def main():
    ast.parse((HERE / 'train_deberta_row_anchor.py').read_text(encoding='utf-8'))
    check_data()
    check_keep_loss()
    check_batch_order()
    check_zero_weight()
    output_dir = HERE / 'outputs/checks'
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {'data_and_hyperparameters_match': True, 'pair_cache_indices_match': True,
              'four_label_kl_and_gradients_correct': True, 'zero_weight_toy_path_identical': True,
              'deberta_loaded': False, 'deberta_training_performed': False}
    (output_dir / 'check_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\n全部CPU检查通过；只训练合成小网络作对照，没有加载或训练DeBERTa。')


if __name__ == '__main__':
    main()
