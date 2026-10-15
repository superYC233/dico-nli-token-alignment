
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

from pathlib import Path
import os

import pandas as pd
import torch


# 当前 py 文件所在的项目目录
PROJECT_DIR = PROJECT_ROOT

# 旧模型文件用 ./data 读取数据，因此导入前设定工作目录
os.chdir(PROJECT_DIR)

# 复用旧文件中的模型结构、tokenizer 和 pair-aware 函数
from experiments.alignment import deberta_nli_base_null_alignment_M as model_code


INPUT_FILE = (
    PROJECT_DIR
    / "data/trial/dico_nli_trial_track1_participant_unlabeled.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "model/deberta_nli_base_null_alignment_M_seed_42/epoch_3.pth"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "predictions/trial_track1_predictions.csv"
)

ID_TO_LABEL = {
    0: "FORWARD_ENTAILMENT",
    1: "BACKWARD_ENTAILMENT",
    2: "EQUIVALENCE",
    3: "NEGATIVE_OTHER",
}


# TODO 1：读取无标签数据
def load_unlabeled_data():
    data = pd.read_csv(INPUT_FILE)

    required = {"instance_id", "pair_id", "text1", "text2"}
    assert required <= set(data.columns)
    assert "label" not in data.columns
    assert len(data) > 0
    assert not data["instance_id"].duplicated().any()
    assert data[["text1", "text2"]].notna().all().all()

    # pair_aware 用行号寻找同一个 pair_id 的两条数据
    data = data.reset_index(drop=True)

    pair_sizes = data.groupby("pair_id").size()
    assert pair_sizes.isin([1, 2]).all()

    print("无标签样本数：", len(data))
    return data


# TODO 2：加载已训练好的模型
def load_model():
    model = model_code.debertaDiCoModel().to(model_code.device)

    weights = torch.load(
        MODEL_FILE,
        map_location=model_code.device,
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    print("已加载模型：", MODEL_FILE)
    print("设备：", model_code.device)
    return model


# TODO 3：分批得到每条样本的四分类分数
def predict_logits(model, data, batch_size=16):
    all_logits = []

    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start:start + batch_size]

            encoded = model_code.my_tokenizer(
                batch["text1"].tolist(),
                batch["text2"].tolist(),
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )

            logits = model(
                input_ids=encoded["input_ids"].to(model_code.device),
                attention_mask=encoded["attention_mask"].to(
                    model_code.device
                ),
            )

            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)


# TODO 4：按 pair_id 调整预测，并保存两列 CSV
def save_predictions(data, logits):
    pred_ids = model_code.pair_aware(logits, data)

    predictions = pd.DataFrame({
        "instance_id": data["instance_id"].tolist(),
        "label": [ID_TO_LABEL[int(i)] for i in pred_ids.tolist()],
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("预测条数：", len(predictions))
    print("输出文件：", OUTPUT_FILE)
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    data = load_unlabeled_data()
    model = load_model()
    logits = predict_logits(model, data)
    save_predictions(data, logits)