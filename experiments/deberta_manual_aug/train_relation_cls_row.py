"""右键运行A：DeBERTa+LoRA+CLS，只用四分类Row CE。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.deberta_manual_aug.train_relation_row import main


RUN_NAME = 'run22_cls_row_seed7'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, head_type='cls', prepare_only=args.prepare_only)
