"""检查 Woman 例子的冠词/大小写敏感性；仅作诊断，不生成训练数据。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from transformers import AutoTokenizer
from experiments.deberta_row_ce_frozen import config as cfg
from experiments.deberta_row_ce_frozen.train_deberta_row_ce import build_model
from export_predictions import OUT, BASELINE_MODEL_DIR, predict, sha256


def main():
    manifest = json.loads((OUT / 'inference_manifest.json').read_text(encoding='utf-8'))
    checkpoint = BASELINE_MODEL_DIR / 'best_model.pth'
    if sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('权重已变化，不能与本次审计混用。')
    rows = []
    for first in ['Woman', 'woman', 'A woman', 'a woman']:
        for second in ['an adult female', 'An adult female']:
            for direction in [0, 1]:
                a, b = (first, second) if direction == 0 else (second, first)
                rows.append({'instance_id': f'probe_{len(rows)}', 'pair_id': f'probe_pair_{len(rows)//2}',
                             'text1': a, 'text2': b, 'label': 'EQUIVALENCE'})
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_DIR / 'tokenizer', local_files_only=True)
    model = build_model(tokenizer)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    result = predict(pd.DataFrame(rows), tokenizer, model, device)
    # Train / Dev 已有 E 标签提供两个端点；中间人工变体没有额外金标裁决。
    result['status'] = '人工诊断变体，E 为待审核的保义假设；不计入正式 Dev 成绩'
    result.to_csv(OUT / 'woman_surface_probes.csv', index=False, encoding='utf-8-sig')
    print(result[['text1', 'text2', 'prediction', 'prob_E', 'pair_prediction']].to_string(index=False))


if __name__ == '__main__':
    main()
