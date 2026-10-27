"""右键运行：旧scope200 + 新48行；先完整Row CE，再完整Pair CE。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.deberta_manual_aug.train_relation_manual import main


RUN_NAME = 'run06_relation48_staged'
AUGMENT_MODE = 'manual'  # manual=旧3242+新48；copy=旧3242+复制48；baseline=旧3242。


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(RUN_NAME, AUGMENT_MODE, training_mode='staged', prepare_only=args.prepare_only)
