"""右键运行：依次训练 A、B、C、D、E。每组使用一个独立进程。"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.deberta_abcde import config as cfg


# 中断后可改成尚未完成的组，例如 ['C', 'D', 'E']。
MODES = ['A', 'B', 'C', 'D', 'E']


def main():
    for mode in MODES:
        model_dir, results_dir = cfg.get_output_dirs(mode)
        if model_dir.exists() or results_dir.exists():
            raise FileExistsError(f'{mode} 的输出目录已经存在。请更换 RUN_NAME 或修改 MODES。')

    entry = Path(__file__).resolve().parent / 'train_deberta_abcde.py'
    output_path = cfg.EXPERIMENT_DIR / 'outputs' / cfg.RUN_NAME / 'summary.csv'
    # 只补跑剩余组时，保留已经完成的组的汇总记录。
    summary_rows = pd.read_csv(output_path).to_dict('records') if output_path.exists() else []
    for mode in MODES:
        print(f'\n========== 开始训练 DeBERTa {mode} ==========', flush=True)
        subprocess.run(
            [sys.executable, '-u', str(entry), '--mode', mode],
            cwd=str(PROJECT_ROOT), check=True,
        )
        # 每组结束，子进程退出并释放显存，再启动下一组。
        model_dir, results_dir = cfg.get_output_dirs(mode)
        best = json.loads((model_dir / 'best_metrics.json').read_text(encoding='utf-8'))
        summary = pd.read_csv(results_dir / 'best_summary.csv')
        for record in summary.to_dict('records'):
            record.update(model=mode, best_epoch=best['best_epoch'])
            summary_rows.append(record)

        combined = pd.DataFrame(summary_rows)
        combined.to_csv(output_path, index=False, encoding='utf-8-sig')
        print('\n===== 已完成各组：各自最佳 pair-aware F1 轮次 =====')
        print(combined.to_string(index=False))
        print('汇总文件：', output_path)


if __name__ == '__main__':
    main()
