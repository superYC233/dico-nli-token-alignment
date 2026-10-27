"""右键运行：复核原0.9248版已保存的Dev预测，不加载模型。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argparse import ArgumentParser

from evaluation_functions.scorer import score_files
from project_paths import PROJECT_ROOT


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold", type=str,
        default=str(PROJECT_ROOT / "data/dev/dico_nli_dev_track1_reference.csv"),
    )
    parser.add_argument(
        "--predictions", type=str,
        default=str(PROJECT_ROOT / "artifacts/final/dev_predictions_pair_aware.csv"),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "results/final_baseline"),
    )
    args = parser.parse_args()
    report = score_files(args.gold, args.predictions, output_dir=args.output_dir).to_dict()
    print(f"Weighted F1：{report['weighted_f1']:.6f}")
    print(f"SoftCons：{report['soft_cons']:.6f}")
    print(f"HardCons：{report['hard_cons']:.6f}")
    print(f"评分结果：{args.output_dir}")


if __name__ == "__main__":
    main()
