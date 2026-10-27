"""第三步：加载第二阶段最佳 Adapter，输出独立预测和 pair-aware 成绩。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiments.qwen_mnli_transfer import config as cfg
from experiments.qwen_mnli_transfer.common import file_hash, load_tokenizer
from experiments.generative_nli import (
    load_data, get_datasets, predict_with_scores, pair_aware_decode, score_methods,
)


def main():
    manifest = json.loads((cfg.BEST_ADAPTER_DIR / 'transfer_config.json').read_text(encoding='utf-8'))
    if manifest['stage'] != 'dico':
        raise ValueError('请加载第二阶段 DiCo 的 best_adapter，不能直接拿 MNLI Adapter 做四分类。')
    if file_hash(cfg.BEST_ADAPTER_DIR / 'adapter_model.safetensors') != manifest['adapter_sha256']:
        raise ValueError('DiCo Adapter 权重与清单不一致。')
    # 使用训练时保存的路径、tokenizer 和 Prompt，避免后来改配置导致预测不匹配。
    tokenizer = load_tokenizer(cfg.BEST_ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model_path'], dtype='auto', local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(cfg.BEST_ADAPTER_DIR), local_files_only=True).to('cuda')
    model.eval()
    dataset = load_data(cfg.PROJECT_ROOT, cfg.CACHE_DIR)
    dataset = get_datasets(dataset, manifest['prompt_template'])
    frame = predict_with_scores(model, tokenizer, dataset['dev'], manifest['prediction_batch_size'])
    frame = pair_aware_decode(frame)
    score_methods(frame, cfg.RESULTS_DIR, cfg.GOLD_PATH)
    print('结果目录：', cfg.RESULTS_DIR)


if __name__ == '__main__':
    main()
