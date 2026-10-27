"""右键运行：融合历史0.938/0.9355/0.9306/0.9248的已有预测，不训练。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.deberta_lexical_mixed_pair import config_high_fusion
from experiments.deberta_lexical_mixed_pair.analyze_fusion import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-qwen', action='store_true', help='补算高分版本与原Qwen C+r16的等权融合。')
    args = parser.parse_args()
    if args.with_qwen:
        from experiments.deberta_lexical_mixed_pair import config_high_qwen_fusion
        main(config_high_qwen_fusion)
    else:
        main(config_high_fusion)
