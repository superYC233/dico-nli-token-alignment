"""先运行这个文件：只下载预训练模型，不训练。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

from experiments.deberta_nli_large import config as cfg


def main():
    # 只下载一份 safetensors 权重和分词文件，不下载重复的 bin / ONNX 权重。
    model_dir = snapshot_download(
        repo_id=cfg.MODEL_NAME,
        revision=cfg.MODEL_REVISION,
        cache_dir=str(cfg.MODEL_CACHE_DIR),
        allow_patterns=[
            'config.json', 'model.safetensors', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json',
            'added_tokens.json', 'spm.model', 'README.md',
        ],
    )

    print('模型下载完成：')
    print(model_dir)
    print('接下来运行 train_deberta_large.py')


if __name__ == '__main__':
    main()
