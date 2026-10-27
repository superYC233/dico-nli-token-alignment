"""右键运行：先做Train三折评分，再训练最终一个模型。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deberta_lexical_mixed_pair.score_train_oof import main as score_train
from experiments.deberta_lexical_mixed_pair.train_deberta_oof_hard import main as train_final


if __name__ == '__main__':
    score_train()
    train_final()
