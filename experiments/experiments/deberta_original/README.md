# 原0.9248版

本目录保存原0.9248配方的复现入口：官方Train 3042行，DeBERTa-v3-base的NLI初始化，Token Alignment，先Row CE再Real Pair CE，预测使用pair-aware。默认不加人工增强，也没有Lexical分支。

| 文件 | 用途 |
|---|---|
| [train.py](train.py) | 训练并评估这一个版本 |
| [evaluate.py](evaluate.py) | 读取已保存的预测CSV，计算官方成绩；不重新加载模型 |

在项目根目录运行：

```powershell
python experiments/deberta_original/train.py
python experiments/deberta_original/evaluate.py
```

也可在IDE中直接运行对应文件。训练参数、增强开关和本机模型路径仍在原代码中设置，数据位置见[data说明](../../data/README.md)。移动入口只调整了项目根目录定位，没有改训练算法和参数。

原版历史Dev pair-aware F1为 **0.924794**。历史预测和配置保留在[artifacts/final](../../artifacts/final)，独立预测错误报告在[原版错误分析](../../artifacts/original_error_analysis)，分析工具在[tools/analysis](../../tools/analysis/analyze_deberta_original.py)。

[alignment](../alignment)里保留更早的对齐、NULL等实验，与本目录的完整官方Train复现入口有区别。[S14](../ensemble_selected/S14.py)等后期融合方案有自己的目录，不从这里运行。
