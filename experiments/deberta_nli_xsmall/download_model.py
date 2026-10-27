"""先运行：只下载xsmall预训练模型，不训练。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from experiments.deberta_nli_xsmall import config as cfg


def main():
    # 只下载一份safetensors权重和分词文件，不下载重复的bin / ONNX。
    model_dir = snapshot_download(
        repo_id=cfg.MODEL_NAME,
        local_dir=str(cfg.MODEL_PATH),
        allow_patterns=[
            'config.json', 'model.safetensors', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json',
            'added_tokens.json', 'spm.model', 'README.md',
        ],
    )
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    assert (Path(model_dir) / 'model.safetensors').is_file(), '下载目录缺少model.safetensors。'
    assert config.model_type == 'deberta-v2' and config.hidden_size == 384
    assert tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None
    print('模型下载完成：', model_dir)
    print('隐藏维度 / 层数：', config.hidden_size, '/', config.num_hidden_layers)
    print('接下来运行 train_deberta_xsmall.py')


if __name__ == '__main__':
    main()
