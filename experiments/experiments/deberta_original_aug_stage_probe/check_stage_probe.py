"""CPU 合成分数自检；不加载真实主干，不训练，不执行 backward 或 optimizer.step。"""

import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.deberta_original_aug_stage_probe import train_stage_probe as probe
from experiments.deberta_original_aug_stage_probe.analyze_stages import (
    analyze_saved, compare_stages, summarize_run,
)


class ScoreModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer('scores', scores.clone())

    def forward(self, input_ids, attention_mask, features):
        # 故意消耗三种随机数，检查评估后是否真正恢复。
        random.random()
        np.random.rand()
        torch.rand(1)
        return self.scores[input_ids[:, 0]]


def main():
    frame = pd.read_csv(probe.base.DEV_PATH, keep_default_na=False)
    labels = {'FORWARD_ENTAILMENT': 0, 'BACKWARD_ENTAILMENT': 1,
              'EQUIVALENCE': 2, 'NEGATIVE_OTHER': 3}
    gold = torch.tensor(frame.label.map(labels).tolist())
    scores = torch.zeros(len(frame), 4)
    scores[torch.arange(len(frame)), gold] = 8
    batches = [{'input_ids': torch.tensor([i, 0]), 'attention_mask': torch.ones(2, dtype=torch.long),
                'label': gold[i]} for i in range(len(frame))]
    loader = DataLoader(batches, batch_size=16, shuffle=False,
                        generator=torch.Generator().manual_seed(43))
    model = ScoreModel(scores)
    model.train()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, loader_state = torch.get_rng_state().clone(), loader.generator.get_state().clone()

    # 所有合成结果都放在本实验目录，退出后自动清理。
    parent = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='_cpu_check_', dir=parent) as directory:
        work = Path(directory).resolve()
        assert work.is_relative_to(parent) and work.name.startswith('_cpu_check_')
        row_dir, pair_dir = work / 'epoch_1/after_row', work / 'epoch_1/after_pair'
        before = probe.evaluate_stage(model, loader, frame, torch.device('cpu'), row_dir)
        assert model.training and random.getstate() == python_state
        ns = np.random.get_state()
        assert ns[0] == numpy_state[0] and np.array_equal(ns[1], numpy_state[1]) and ns[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert torch.equal(loader.generator.get_state(), loader_state)
        assert torch.equal(model.scores, scores)
        assert (before.weighted_f1 == 1.0).all()

        # 直接设置测试分数，模拟“所有配对行的 N 分数过高”；没有训练更新。
        model.scores[:, 3] = 12
        after = probe.evaluate_stage(model, loader, frame, torch.device('cpu'), pair_dir)
        effects = compare_stages(
            row_dir / 'predictions_with_scores.csv', pair_dir / 'predictions_with_scores.csv',
            before, after, work / 'epoch_1/pair_changes', 1, 'pair_ce',
        )
        independent, paired = effects
        assert independent['broken'] == 554 and independent['fixed'] == 0
        assert independent['nonN_to_N_before'] == 0 and independent['nonN_to_N_after'] == 554
        assert independent['N_correct_before'] == independent['N_correct_after'] == 106
        assert paired['broken'] == paired['fixed'] == 0
        assert after.set_index('method').loc['pair_aware', 'weighted_f1'] == 1.0

        history = []
        for stage, summary in [('after_row', before), ('after_pair', after)]:
            for row in summary.to_dict('records'):
                row.update(epoch=1, stage=stage, row_loss=0.2, pair_loss=0.3)
                history.append(row)
        pd.DataFrame(history).to_csv(work / 'stage_summary.csv', index=False)
        pd.DataFrame(effects).to_csv(work / 'phase_effects.csv', index=False)
        old_config = json.loads((probe.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
        (work / 'run_config.json').write_text(json.dumps(old_config), encoding='utf-8')
        table = summarize_run(work, probe.REFERENCE_DIR)
        assert len(table) == 1 and table.iloc[0].pair_nonN_to_N == 554
        assert table.iloc[0].pair_ind_delta_pp < -70 and (work / 'REPORT.md').exists()
        original_history = (work / 'stage_summary.csv').read_bytes()
        rebuilt = analyze_saved(work, probe.REFERENCE_DIR)
        pd.testing.assert_frame_equal(table, rebuilt)
        assert (work / 'stage_summary.csv').read_bytes() == original_history
        report = json.loads((work / 'reference_protocol_check.json').read_text(encoding='utf-8'))
        assert report['same_recorded_protocol'] and report['all_compared_predictions_match'] is False

    print('\n自检通过：随机数/模型状态恢复，554条非N→N变化，修好/破坏统计，报告及重建分析。')
    print('未加载真实主干，未执行 backward、optimizer.step 或训练。')


if __name__ == '__main__':
    main()
