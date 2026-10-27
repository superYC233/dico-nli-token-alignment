# 词汇特征版 + 范围增强

右键运行本目录 `train_deberta_lexical_aug.py`。训练由用户启动。

当前配置在训练文件顶部：

```python
RUN_NAME = 'run01_lexical_scope'
AUGMENT_MODE = 'scope'
```

复用原 0.930569 词汇版的模型、特征和训练流程，仅加入与 0.935493 原版增强相同的 200 条样本。仍从同一 NLI 预训练模型初始化，不加载已选好的词汇版 checkpoint 继续训练。

- 全参数 DeBERTa + Token Alignment + 20维词汇特征分支，新增分支804参数。
- 主模型 lr=2e-5，词汇分支 lr=1e-3，hidden=32；float32、batch16、10轮、seed42。
- 原 Train 3042 + 200，得到3242行、1362个配对组；每轮203个Row步骤和86个Pair步骤。
- 每轮先完整Row CE，再完整0.5×Pair CE；类别权重和label smoothing与旧词汇版相同。
- 按Dev pair-aware W-F1只保存一份最佳权重，同时报告独立预测。
- 最佳模型选定后，再评估同一批保留诊断，独立和结构解码分别报告。

词汇特征只从当前text1/text2提取，不读取金标、pair_id或Dev统计。正反两行分别生成自己的特征。增强样本仍是未独立审核的人工模板标签；Dev错误分析曾用于确定增强方向，结果是开发集结果。

训练结束自动与旧词汇版0.930569、原版增强0.935493比较，列出修好与新增错误。是否有效需要看实际结果，不预设两种方法组合一定涨分。

## 保存位置

- 模型：`D:/model/SEAMEVAL/deberta_lexical_aug/run01_lexical_scope/best_model.pth`。
- 结果：本目录`outputs/run01_lexical_scope/`。
- 数据快照：`training_data.csv`、`extra_train_rows.csv`、`augmentation_audit.csv`。
- 最佳分数：`best_summary.csv`；逐轮分数：`epoch_summary.csv`。
- 参照比较：`reference_comparison.csv`和`comparisons/`。

原词汇版、原版增强和冻结版的配置、权重与结果均保留。更换实验时改本文件顶部RUN_NAME与AUGMENT_MODE；只准备数据可加`--prepare-only`，不会训练。
