"""CPU 验证统计和配置切换；不加载真实大模型，不写 D 盘。"""
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from experiments.deberta_lexical_reliability import config as cfg
from experiments.deberta_lexical_reliability.protocol import digest, protocol_definition
from experiments.deberta_lexical_reliability.run_one import bind_original_code
from experiments.deberta_lexical_reliability.analyze import (
    weighted_f1, group_confusion, paired_group_bootstrap, load_prediction,
)
from experiments.deberta_lexical.model import DebertaDiCoModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(32, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def main():
    torch.set_num_threads(2)
    output = cfg.RESULTS_ROOT / f'_cpu_{time.time_ns()}'
    output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for labels_present in [[0, 1, 2, 3], [0], [1, 3]]:
        y = rng.choice(labels_present, size=100)
        p = rng.integers(0, 4, size=100)
        cm = np.zeros((4, 4), dtype=np.int64)
        np.add.at(cm, (y, p), 1)
        expected = f1_score(y, p, labels=[0, 1, 2, 3], average='weighted', zero_division=0)
        assert abs(float(weighted_f1(cm)) - expected) < 1e-12
    y = np.array([0, 1, 2, 2, 3])
    p = np.array([0, 1, 0, 1, 3])
    groups = np.array([0, 0, 1, 1, 2])
    matrices = group_confusion(y, p, groups)
    perfect = group_confusion(y, y, groups)
    samples = paired_group_bootstrap([matrices, matrices, perfect], 500, 21)
    assert (samples[:, 0] == samples[:, 1]).all()  # 同抽样，相同模型差值严格为 0。
    assert (samples[:, 2] == 1).all()
    assert np.array_equal(samples, paired_group_bootstrap([matrices, matrices, perfect], 500, 21))
    assert matrices[0].sum() == 2 and matrices[2].sum() == 1
    print('Weighted F1 与 sklearn 一致；组抽样、共同抽样、固定随机数和缺失类别检查通过。')

    dev = pd.DataFrame({'instance_id': ['a', 'b', 'c', 'd', 'e'],
                        'pair_id': ['p', 'p', 'q', 'q', 'n'],
                        'label': [cfg.FULL_LABELS[i] for i in y]})
    shuffled = dev.sample(frac=1, random_state=7)
    shuffled[['instance_id', 'label']].to_csv(output / 'historical.csv', index=False)
    assert np.array_equal(load_prediction(output / 'historical.csv', dev), y)
    frame = dev.rename(columns={'label': 'gold_label'}).copy()
    frame['pair_prediction'] = [cfg.LABELS[i] for i in p]
    frame.to_csv(output / 'current.csv', index=False)
    assert np.array_equal(load_prediction(output / 'current.csv', dev), p)
    frame.loc[0, 'gold_label'] = cfg.FULL_LABELS[3]
    frame.to_csv(output / 'bad.csv', index=False)
    try:
        load_prediction(output / 'bad.csv', dev)
    except ValueError:
        pass
    else:
        raise AssertionError('金标不一致必须报错。')

    old_config_path = cfg.PROJECT_ROOT / 'experiments/deberta_lexical/config.py'
    old_digest = digest(old_config_path)
    saved_seed = cfg.SEED
    cfg.SEED = 7
    first_protocol = protocol_definition()
    cfg.SEED = 21
    assert protocol_definition() == first_protocol
    cfg.SEED = 7
    baseline_dir, _ = cfg.get_output_dirs(False)
    lexical_dir, _ = cfg.get_output_dirs(True)
    assert baseline_dir != lexical_dir
    assert str(baseline_dir).replace('\\', '/').startswith('D:/model/SEAMEVAL/deberta_lexical_reliability/')
    cfg.SEED = saved_seed
    training = bind_original_code()
    assert training.cfg is cfg
    assert digest(old_config_path) == old_digest
    from experiments.deberta_lexical import data, evaluation
    assert data.cfg is cfg and evaluation.cfg is cfg
    train, development = data.read_data()
    assert len(train) == 3042 and len(development) == 660

    torch.manual_seed(7)
    backbone = TinyBackbone()
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2)
    before = torch.get_rng_state().clone()
    baseline = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20, use_lexical_features=False)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    lexical = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20, use_lexical_features=True)
    assert torch.equal(after, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, lexical.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 3, 4, 2, 5, 2, 0, 0]])
    mask = ids.ne(0).long()
    features = torch.rand(1, 20)
    baseline.eval(); lexical.eval()
    with torch.no_grad():
        torch.testing.assert_close(baseline(ids, mask, features), lexical(ids, mask, features), rtol=0, atol=0)
    opt = training.build_optimizer(lexical)
    assert [group['lr'] for group in opt.param_groups] == [2e-5, 1e-3]
    assert [group['lr'] for group in training.build_optimizer(baseline).param_groups] == [2e-5]
    (output / 'protocol_preview.json').write_text(json.dumps(first_protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print('原源码未修改、D 盘路径、冻结协议、共同参数/RNG/初始输出及优化器分组检查通过。')
    print('CPU 检查全部通过；没有加载真实模型权重，没有启动 GPU 训练，没有写 D 盘。')
    print('检查记录：', output)


if __name__ == '__main__':
    main()
