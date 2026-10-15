# 数据放置说明

请从比赛官方渠道取得 Track 1 数据，然后按以下路径放置：

```text
data/train/dico_nli_train_track1_participant_labeled.csv
data/dev/dico_nli_dev_track1_participant_labeled.csv
data/dev/dico_nli_dev_track1_reference.csv
```

官方训练集为 3042 行，官方 DEV 为 660 行。数据没有放入这个上传副本。
最终 DEV 预测与评分保存在 `artifacts/final/`，不需要 checkpoint 即可复核评分，但需要对应的 DEV reference 文件。

历史的内部划分和数据增强工具见 `tools/data/`；最终方案不使用增强数据。
