import json

from peft import PeftModel
from transformers import AutoModelForCausalLM

import smol_config as cfg
from smol_utils import (
    load_data, load_tokenizer, get_datasets, predict_with_scores,
    pair_aware_decode, score_methods,
)


def main():
    config_path = cfg.BEST_DIR / 'run_config.json'
    if not config_path.exists():
        raise FileNotFoundError('还没有最佳 Adapter，请先运行 train_lora.py。')
    run_config = json.loads(config_path.read_text(encoding='utf-8'))
    if run_config['model_id'] != cfg.MODEL_ID:
        raise ValueError('Adapter 不是本次 SmolLM2 实验的模型。')
    tokenizer = load_tokenizer(cfg.BEST_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        run_config['model_path'], dtype='auto', local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(cfg.BEST_DIR)).to('cuda')
    model.eval()
    dataset = get_datasets(load_data(), prompt_template=run_config['prompt_template'])
    frame = predict_with_scores(
        model, tokenizer, dataset['dev'], batch_size=run_config['prediction_batch_size'],
    )
    frame = pair_aware_decode(frame)
    score_methods(frame, cfg.RESULTS_DIR)
    print('保存样本数：', len(frame))
    print('结果目录：', cfg.RESULTS_DIR)


if __name__ == '__main__':
    main()
