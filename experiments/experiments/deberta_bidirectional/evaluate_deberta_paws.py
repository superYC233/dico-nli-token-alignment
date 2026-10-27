"""可选：重载PAWS辅助训练的最佳模型，模型结构和run02完全相同。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.deberta_bidirectional.evaluate_deberta_alignment_joint import main
from experiments.deberta_bidirectional.train_deberta_paws import RUN_NAME, MODEL_ROOT


MODEL_DIR = MODEL_ROOT / RUN_NAME


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.model_dir)
