"""右键运行：下载经过MNLI、ANLI等任务微调的ModernBERT-base。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import HfApi, snapshot_download
from experiments.modernbert_nli_base import config as cfg


def main():
    print('下载模型：', cfg.MODEL_NAME)
    print('保存目录：', cfg.MODEL_PATH)
    revision = HfApi().model_info(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION).sha
    snapshot_download(repo_id=cfg.MODEL_NAME, revision=revision, local_dir=str(cfg.MODEL_PATH))
    info = {'repo_id': cfg.MODEL_NAME, 'revision': revision, 'local_dir': str(cfg.MODEL_PATH)}
    (cfg.MODEL_PATH / 'download_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('下载完成：', cfg.MODEL_PATH)


if __name__ == '__main__':
    main()
