"""右键运行：四次训练串行执行，分别保存最佳模型，最后汇总。"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.deberta_lexical_reliability import config as cfg
from experiments.deberta_lexical_reliability.protocol import ensure_protocol


def main():
    ensure_protocol()
    print('新模型统一保存到：', cfg.MODEL_ROOT, flush=True)
    print('配方冻结；seed 7、21 各跑 baseline 和 lexical。', flush=True)
    if shutil.disk_usage(cfg.MODEL_ROOT.anchor).free < 4 * 1024 ** 3:
        raise OSError('四次训练建议至少预留 4 GB 可用空间。')
    script = cfg.EXPERIMENT_DIR / 'run_one.py'
    for seed in cfg.SEEDS:
        for arm in ['baseline', 'lexical']:
            subprocess.run([sys.executable, '-u', str(script), '--seed', str(seed), '--arm', arm],
                           cwd=cfg.PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, '-u', str(cfg.EXPERIMENT_DIR / 'analyze.py')],
                   cwd=cfg.PROJECT_ROOT, check=True)


if __name__ == '__main__':
    main()
