"""CPU检查数据、评分、平均公式和新增评估对训练路径的影响；不训练DeBERTa。"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_joint_pair.data import GroupDataset
from experiments.deberta_manual_aug.train_relation_manual import train_staged
from experiments.deberta_stage_average import train_deberta_stage_average as cfg


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / 'outputs/checks'


def check_data():
    train, dev, reference = cfg.read_data()
    for name, value in [('model_name', cfg.MODEL_NAME), ('seed', cfg.SEED), ('epochs', cfg.EPOCHS),
                        ('row_batch_size', cfg.ROW_BATCH_SIZE), ('pair_batch_size', cfg.PAIR_BATCH_SIZE),
                        ('dev_batch_size', cfg.DEV_BATCH_SIZE), ('max_length', cfg.MAX_LENGTH),
                        ('learning_rate', cfg.LEARNING_RATE), ('weight_decay', cfg.WEIGHT_DECAY),
                        ('row_weights', cfg.ROW_WEIGHTS), ('pair_weights', cfg.PAIR_WEIGHTS),
                        ('pair_weight', cfg.PAIR_WEIGHT), ('pair_label_smoothing', cfg.PAIR_LABEL_SMOOTHING)]:
        assert reference[name] == value, f'{name}和0.9381配方不同。'
    assert 1 <= cfg.AVERAGE_START <= cfg.EPOCHS and cfg.TRAIN_AUDIT_EPOCH == 2
    for name in ['training_data.csv', 'extra_train_rows.csv', 'relation_examples_source.csv',
                 'manual_label_notes.csv', 'train_word_sources.csv', 'overlap_report.json', 'data_config.json']:
        assert (cfg.REFERENCE_DIR / name).is_file()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True)
    groups = GroupDataset(train, tokenizer, cfg.MAX_LENGTH)
    pairs = cfg.RealPairDataset(groups.rows)
    assert len(groups) == 1910 and len(pairs) == 1380
    assert torch.bincount(groups.rows.labels, minlength=4).tolist() == [933, 933, 894, 530]
    assert (len(train) + cfg.ROW_BATCH_SIZE - 1) // cfg.ROW_BATCH_SIZE == 206
    assert (len(pairs) + cfg.PAIR_BATCH_SIZE - 1) // cfg.PAIR_BATCH_SIZE == 87
    cfg.write_train_reference(train, OUTPUT_DIR / 'train_reference.csv')
    print('数据/超参数：与0.9381一致；Row206步，Pair87步。')
    return train, dev


def check_averaging():
    model = nn.Linear(1, 1, bias=False)
    before = torch.get_rng_state().clone()
    average = AveragedModel(model, device='cpu')
    assert torch.equal(before, torch.get_rng_state())
    assert int(average.n_averaged) == 0
    for epoch in range(1, 11):
        with torch.no_grad():
            model.weight.fill_(epoch)
        if epoch >= cfg.AVERAGE_START:
            average.update_parameters(model)
    assert int(average.n_averaged) == cfg.EPOCHS - cfg.AVERAGE_START + 1
    expected = sum(range(cfg.AVERAGE_START, cfg.EPOCHS + 1)) / int(average.n_averaged)
    torch.testing.assert_close(average.module.weight, torch.tensor([[expected]]))
    model.load_state_dict(average.module.state_dict(), strict=True)
    torch.testing.assert_close(model.weight, torch.tensor([[expected]]))
    assert model.weight.device.type == 'cpu' and average.module.weight.grad is None
    print('平均公式：只计入第6—10轮，测试参数平均值为8；保存时无module前缀。')


class ToyRows(Dataset):
    def __init__(self):
        self.frame = pd.DataFrame({
            'instance_id': [f'check_{i}' for i in range(5)],
            'pair_id': ['check_FB', 'check_FB', 'check_EE', 'check_EE', 'check_N'],
            'text1': ['a', 'b', 'c', 'd', 'e'], 'text2': ['b', 'a', 'd', 'c', 'f'],
            'label': [cfg.FULL_LABELS[i] for i in [0, 1, 2, 2, 3]],
        })
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
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(8, 4)

    def forward(self, input_ids, attention_mask, features):
        return self.output(self.dropout(self.embedding(input_ids).mean(dim=1)))


def toy_training(extra_evaluation):
    cfg.set_seed(cfg.SEED)
    model = ToyModel()
    if extra_evaluation:
        # 模拟正式脚本：在CPU构造平均副本不应改变在线模型的随机数。
        average = AveragedModel(model, device='cpu')
    rows = ToyRows()
    row_loader = DataLoader(rows, batch_size=2, shuffle=True)
    pair_loader = DataLoader(cfg.RealPairDataset(rows), batch_size=2, shuffle=True)
    # 故意不指定独立generator，检验fork_rng确实能保护全局随机数。
    eval_loader = DataLoader(rows, batch_size=2, shuffle=False)
    gold_path = OUTPUT_DIR / 'toy_reference.csv'
    cfg.write_train_reference(rows.frame, gold_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    row_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    for epoch in range(1, 3):
        if extra_evaluation:
            cfg.train_rows(model, row_loader, optimizer, row_ce, torch.device('cpu'))
            before = torch.get_rng_state().clone()
            cfg.evaluate_phase(model, eval_loader, rows.frame, torch.device('cpu'),
                               OUTPUT_DIR / f'toy_epoch_{epoch}/after_row', gold_path)
            assert torch.equal(before, torch.get_rng_state()) and model.training
            cfg.train_pairs(model, pair_loader, optimizer, pair_ce, torch.device('cpu'))
            cfg.evaluate_phase(model, eval_loader, rows.frame, torch.device('cpu'),
                               OUTPUT_DIR / f'toy_epoch_{epoch}/after_pair', gold_path)
            average.update_parameters(model)
        else:
            train_staged(model, row_loader, pair_loader, optimizer, row_ce, pair_ce, torch.device('cpu'))
    return model.state_dict(), optimizer.state_dict(), torch.get_rng_state()


def check_training_path():
    old_weights, old_optimizer, old_rng = toy_training(extra_evaluation=False)
    new_weights, new_optimizer, new_rng = toy_training(extra_evaluation=True)
    for name in old_weights:
        torch.testing.assert_close(new_weights[name], old_weights[name], rtol=0, atol=0)
    for index, state in old_optimizer['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(new_optimizer['state'][index][name], value, rtol=0, atol=0)
    assert torch.equal(old_rng, new_rng)
    print('合成小网络：新增阶段评估和平均后，在线参数/AdamW状态/随机数与原两阶段流程逐项一致。')


def check_scoring(train, dev):
    for split, frame, source, gold_path in [
        ('dev', dev, cfg.REFERENCE_DIR / 'epoch_2/predictions_with_scores.csv', cfg.GOLD_PATH),
        ('train', train, cfg.REFERENCE_DIR / 'error_analysis/train_readout.csv', OUTPUT_DIR / 'train_reference.csv'),
    ]:
        saved = pd.read_csv(source, keep_default_na=False)
        assert saved.instance_id.tolist() == frame.instance_id.tolist()
        logits = torch.tensor(saved[[f'logit_{label}' for label in cfg.LABELS]].values, dtype=torch.float32)
        output_dir = OUTPUT_DIR / f'historical_{split}'
        summary = cfg.score_predictions(logits, frame, output_dir, gold_path)
        current = pd.read_csv(output_dir / 'predictions_with_scores.csv', keep_default_na=False)
        for column in ['prediction', 'pair_prediction']:
            assert current[column].tolist() == saved[column].tolist()
        paired = summary[summary.method.eq('pair_aware')].iloc[0]
        raw = summary[summary.method.eq('independent')].iloc[0]
        assert int(raw.nonN_to_N) == (297 if split == 'dev' else 1139)
        if split == 'dev':
            assert abs(paired.weighted_f1 - 0.9381451572973858) < 1e-12
        else:
            assert abs(paired.accuracy * len(train) - 3280) < 1e-9
        print(f'{split}评分复核：两种预测与历史逐条一致。')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ast.parse((HERE / 'train_deberta_stage_average.py').read_text(encoding='utf-8'))
    train, dev = check_data()
    check_averaging()
    check_training_path()
    check_scoring(train, dev)
    result = {'data_and_hyperparameters_match_reference': True, 'average_formula_correct': True,
              'toy_training_path_identical': True, 'historical_train_dev_scoring_matches': True,
              'deberta_loaded': False, 'deberta_training_performed': False}
    (OUTPUT_DIR / 'check_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\n全部CPU检查通过。只训练了合成小网络用于对照，没有加载或训练DeBERTa。')


if __name__ == '__main__':
    main()
