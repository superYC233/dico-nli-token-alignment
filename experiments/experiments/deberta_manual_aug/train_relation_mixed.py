"""右键运行：旧scope200 + 新48行；四类混合，同一步Row CE + 0.5×Pair CE。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.deberta_manual_aug.train_relation_manual import main


RUN_NAME = 'run08_relation48_mixed_g8'
AUGMENT_MODE = 'manual'  # manual=旧3242+新48；copy=旧3242+复制48；baseline=旧3242。


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(RUN_NAME, AUGMENT_MODE, training_mode='mixed', prepare_only=args.prepare_only)
