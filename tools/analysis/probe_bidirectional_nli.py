
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT = (PROJECT_ROOT)
MODEL_DIR = Path(
    r"D:\BaiduNetdiskDownload\PretrainedModel\models"
    r"\models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli"
    r"\snapshots\6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
)


# todo 1. 计算一批短语对的“蕴含”概率
@torch.inference_mode()
def entailment_probs(model, tokenizer, left, right, device, entail_id):
    probabilities = []

    for start in range(0, len(left), 16):
        inputs = tokenizer(
            left[start:start + 16],
            right[start:start + 16],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}

        logits = model(**inputs).logits
        batch_probs = logits.softmax(dim=-1)[:, entail_id]
        probabilities.extend(batch_probs.cpu().tolist())

    return probabilities


# todo 2. 检查双向蕴含能否区分 E 和 F/B
def probe_bidirectional_nli():
    data = pd.read_csv(PROJECT / "data/internal_split/valid.csv")

    # 这里只评估 F/B/E；用 pair_id 去掉同一组的重复方向。
    # 这两步只用于诊断统计，不用于预测 N。
    pairs = data[data["label"] != "NEGATIVE_OTHER"]
    pairs = pairs.drop_duplicates("pair_id").copy()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    entail_id = next(
        number
        for number, name in model.config.id2label.items()
        if name.lower() == "entailment"
    )

    text1 = pairs["text1"].astype(str).tolist()
    text2 = pairs["text2"].astype(str).tolist()

    pairs["p12"] = entailment_probs(
        model, tokenizer, text1, text2, device, entail_id
    )
    pairs["p21"] = entailment_probs(
        model, tokenizer, text2, text1, device, entail_id
    )

    # 两个方向都要高，才像“等价”
    pairs["both_entail"] = pairs[["p12", "p21"]].min(axis=1)

    is_equivalence = pairs["label"].eq("EQUIVALENCE")
    print("全部配对 AUC：", roc_auc_score(is_equivalence, pairs["both_entail"]))

    old_file = PROJECT / (
        "model/deberta_nli_base_token_alignment_internal_seed_42"
        "/dev_predictions_pair_aware.csv"
    )
    old_preds = pd.read_csv(old_file).rename(
        columns={"label": "old_prediction"}
    )
    pairs = pairs.merge(old_preds, on="instance_id", validate="one_to_one")

    errors = pairs[pairs["label"] != pairs["old_prediction"]]
    print("原模型判错的配对数：", len(errors))
    print(
        "这些错误上的 AUC：",
        roc_auc_score(
            errors["label"].eq("EQUIVALENCE"),
            errors["both_entail"],
        ),
    )


if __name__ == "__main__":
    probe_bidirectional_nli()