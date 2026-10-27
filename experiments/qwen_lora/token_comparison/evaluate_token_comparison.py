import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import load_data
from pair_data import PhraseDataset, PhraseCollator
from pair_model import load_model
from pair_metrics import make_prediction_frame, score_prediction_frame


HEAD_TYPE = 'alignment_global'
MODEL_DIR = Path(
    f'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_token_comparison/run01/{HEAD_TYPE}/best_model'
)
OUTPUT_DIR = QWEN_DIR / f'outputs/token_comparison/run01/{HEAD_TYPE}/best_evaluation'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(MODEL_DIR, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dev_frame = load_data()['dev'].to_pandas()
    dataset = PhraseDataset(dev_frame, tokenizer, config['prompt_template'], config['max_length'])
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=PhraseCollator(tokenizer.pad_token_id)
    )

    scores = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{HEAD_TYPE} 预测 Dev'):
            batch.pop('labels')
            inputs = {name: tensor.to(device) for name, tensor in batch.items()}
            # 与训练时的 Dev 评估采用一致的混合精度设置。
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == 'cuda' and config['bf16']
            ):
                logits = model(**inputs)['logits']
            scores.append(logits.float().cpu())
    predictions = make_prediction_frame(dev_frame, torch.cat(scores).numpy())
    score_prediction_frame(predictions, OUTPUT_DIR)
    print('结果目录：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
