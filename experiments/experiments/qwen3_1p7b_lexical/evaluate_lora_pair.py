import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.data_utils import (
    load_data, load_tokenizer, get_label_ids, get_datasets, pair_aware_decode, score_methods,
)
from experiments.qwen3_1p7b_lexical.lexical_features import FEATURE_NAMES, FEATURE_VERSION
from experiments.qwen3_1p7b_lexical.model import QwenLexicalModel
from experiments.qwen3_1p7b_lexical.prediction import predict_with_scores


def main():
    saved = json.loads((cfg.BEST_DIR / 'run_config.json').read_text(encoding='utf-8'))
    if saved['model_id'] != cfg.MODEL_ID or saved['labels'] != cfg.LABELS:
        raise ValueError('模型或标签配置不一致。')
    feature_file = Path(__file__).resolve().parent / 'lexical_features.py'
    if saved['feature_names'] != FEATURE_NAMES or saved['feature_version'] != FEATURE_VERSION:
        raise ValueError('词汇特征定义与训练时不同。')
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != saved['feature_source_sha256']:
        raise ValueError('词汇特征代码已改变，请恢复训练时版本。')
    if not torch.cuda.is_available():
        raise RuntimeError('正式预测需要 CUDA。')
    tokenizer = load_tokenizer(cfg.BEST_DIR)
    if get_label_ids(tokenizer) != saved['label_ids']:
        raise ValueError('标签 token ID 与训练时不同。')
    base = AutoModelForCausalLM.from_pretrained(
        saved['model_path'], dtype='auto', local_files_only=True,
    )
    language_model = PeftModel.from_pretrained(base, str(cfg.BEST_DIR))
    model = QwenLexicalModel(
        language_model, saved['label_ids'], tokenizer.eos_token_id,
        feature_size=len(FEATURE_NAMES), lexical_hidden=saved['lexical_hidden_size'],
        chunk_size=saved['loss_chunk_size'], use_lexical_features=saved['use_lexical_features'],
    )
    state = torch.load(cfg.BEST_DIR / 'lexical_head.pt', map_location='cpu', weights_only=True)
    model.lexical_head.load_state_dict(state, strict=True)
    model.to('cuda').eval()
    formatted = get_datasets(load_data(), saved['prompt_template'])
    frame = predict_with_scores(model, tokenizer, formatted['dev'], saved['prediction_batch_size'])
    frame = pair_aware_decode(frame)
    score_methods(frame, cfg.RESULTS_DIR)
    print('保存样本数：', len(frame))
    print('结果目录：', cfg.RESULTS_DIR)


if __name__ == '__main__':
    main()
