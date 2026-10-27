# 数据放置说明

从 DiCo-NLI 比赛官方渠道取得 Track 1 数据后，将文件放在以下位置：

```text
data/train/dico_nli_train_track1_participant_labeled.csv
data/dev/dico_nli_dev_track1_participant_labeled.csv
data/dev/dico_nli_dev_track1_reference.csv
data/dev/dico_nli_dev_track1_submission_template.csv
```

官方数据下载入口：[DiCo-NLI final_data](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/final_data)。保持官方文件名，不把Dev追加到Train中。

原0.9248版使用官方Train 3042行；后续增强版也使用scope200、relation48等人工样本。3290行表示官方Train 3042＋scope200＋relation48。多模型方案分别使用各自表中列出的训练数据，具体见根目录README第8、14阶段。

本目录的官方原始CSV、内部划分和下载缓存不上传。为复现实验，仓库会保留必要的人工样本、理由标注，以及融合脚本直接读取的少量训练快照；这些完整快照包含官方Train行和新增行，来源仍须按官方任务要求说明。它们不能代替上述官方Dev文件。

原0.9248版预测保存在 `artifacts/final/`；本次S14预测保存在 `experiments/ensemble_selected/outputs/S14/run01_retrain/`。S14开发阶段上传包是 `submissions/S14_track1_dev_20261027/submission_S14_track1_dev.zip`。上传的是预测结果，不需要上传模型权重。

运行[原版evaluate.py](../experiments/deberta_original/evaluate.py)不需要模型权重，但需要上述Dev reference文件。原0.9248版的训练与评估均在[独立目录](../experiments/deberta_original/README.md)中；新训练还需要下载各脚本配置的预训练模型，并按本机情况修改模型路径。

历史数据处理工具在 `tools/data/`。新增样本与Dev的检查结果见 [文本重合核对报告](../docs/DATA_AUGMENTATION_DEV_OVERLAP_AUDIT.md)。
