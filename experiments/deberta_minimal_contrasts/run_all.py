"""右键运行：准备并评估诊断候选；不会训练或修改原模型。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.deberta_minimal_contrasts.evaluate_diagnostics import main


if __name__ == '__main__':
    main()
