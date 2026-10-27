"""先运行一次，下载约 3.4GB 的 BF16 模型权重及 tokenizer。"""
import json

from huggingface_hub import HfApi, snapshot_download

import smol_config as cfg


def main():
    # 固定本次下载的具体版本，重跑时 Hugging Face 会复用已下载文件。
    revision = HfApi().model_info(cfg.MODEL_ID).sha
    print('模型：', cfg.MODEL_ID)
    print('保存位置：', cfg.MODEL_PATH)
    snapshot_download(
        repo_id=cfg.MODEL_ID,
        revision=revision,
        local_dir=str(cfg.MODEL_PATH),
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', 'README.md'],
    )
    (cfg.MODEL_PATH / 'download_info.json').write_text(
        json.dumps({'model_id': cfg.MODEL_ID, 'revision': revision}, indent=2),
        encoding='utf-8',
    )
    print('下载完成，下一步运行 train_lora.py。')


if __name__ == '__main__':
    main()
