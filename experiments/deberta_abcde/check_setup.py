"""只检查五组数据，不加载预训练权重、不训练、不占 GPU 显存。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.deberta_abcde import config as cfg
from experiments.deberta_abcde.data_variants import build_train_dataset, PairDataset, validate_groups


def main():
    original = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    validate_groups(dev)
    frames = {}
    summary = []
    for mode in 'ABCDE':
        frame, n_audit, extra_audit = build_train_dataset(original, mode)
        frames[mode] = frame
        real_pairs = len(PairDataset(frame, tokenizer=None))
        summary.append({
            'mode': mode, 'rows': len(frame),
            'N_rows': int((frame['label'] == 'NEGATIVE_OTHER').sum()),
            'real_pairs': real_pairs,
            'extra_copy': int((extra_audit['operation'] == 'copy').sum()),
            'extra_swap': int((extra_audit['operation'] == 'swap').sum()),
        })
    pd.testing.assert_frame_equal(original.reset_index(drop=True), frames['A'])
    n = len(original)
    b_extra = frames['B'].iloc[n:].reset_index(drop=True)
    c_extra = frames['C'].iloc[n:].reset_index(drop=True)
    assert b_extra['instance_id'].equals(c_extra['instance_id'])
    assert b_extra['label'].equals(c_extra['label'])
    assert b_extra['text1'].equals(c_extra['text2'])
    assert b_extra['text2'].equals(c_extra['text1'])
    assert len(frames['D']) == len(frames['E']) == 2 * n
    assert frames['D']['label'].value_counts().equals(frames['E']['label'].value_counts())
    assert len(PairDataset(frames['D'], None)) == 2 * len(PairDataset(original, None))
    assert len(PairDataset(frames['E'], None)) == 2 * len(PairDataset(original, None))
    print(pd.DataFrame(summary).to_string(index=False))
    print('Dev 保持原样：', len(dev))
    print('N 筛选：', n_audit['selection'].value_counts().to_dict())
    print('检查通过。正式训练请运行 run_all.py。')


if __name__ == '__main__':
    main()
