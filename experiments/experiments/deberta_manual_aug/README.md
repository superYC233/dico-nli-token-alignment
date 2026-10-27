# 回到0.935配方，增加与Dev词汇隔离的手工设计样本

默认运行 `train_original_manual.py`。数据准备和审计会自动完成，训练由你启动。
也可以先运行 `prepare_manual_data.py`，它不加载模型、不训练。

## 保留哪些东西

- 复用 `deberta_original_aug/outputs/run01_original_scope/training_data.csv` 的精确3242行快照：官方Train3042行 + 旧scope增强200行。
- 仍从 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` 初始化；不是在Dev选出的0.935权重上接着训练。
- 原Token Alignment、全参数、无Lexical、float32、lr2e-5、batch16、seed42、10轮。
- 每轮完整Row CE后再完整 `0.5 * Real Pair CE`，不切到混合联合训练。
- 继续按pair-aware W-F1选最佳，每轮同时报告独立预测；只保存一份最佳模型。
- 两阶段训练导致的独立预测退化可能继续存在；这次目标是检验新增样本能否改善最终系统。

## 新样本

`manual_examples.csv` 是助手逐条设计的50组草稿，每组附标签理由。
F/B/E补足反向视图，N保留单例。全部通过检查时扩展成80行，F/B/E/N各20行。
不把Dev错误样本或其原词汇直接搬进训练。

主题包括：额外修饰限制、严格上下位、等价表达/拼写变体、独立属性、核心对象替换、事件状态和模态。
词汇隔离限制了常用介词、冠词等的使用，因此多数是短名词短语；部分等价练习是拼写变体，不能声称覆盖所有句法等价问题。
这些是**助手手工设计的训练草稿，不是新增官方金标，也未经独立人工标注者复核**。

## Dev隔离到底检查什么

仅针对**本次新增人工行**：

1. Unicode NFKC、大小写归一后，完整字母/数字token零交集；不排除停用词。
2. 额外检查Porter词干零交集，减少单复数等词形变体绕过的情况。
3. 整短语和正反向文本对零精确重合。
4. 自动排除已有训练文本对和人工源表内部重复。

旧版200条scope与原始Train有Dev共用词，会另表报告；不能把整个训练集说成词汇零交集。
严格词汇零交集不等于语义零交集，也不等于模型subword零交集。
Porter词干不是完整词形还原算法，审计结论限定于报告所列规则。

Dev错误分析用于选择增强方向，Dev文本用作过滤黑名单。这仍是Dev指导的开发实验。
隐藏Test没有文本时不能声称已证明与Test无交集。Dev反复用于选模的影响不能由去重消除。

## 查看什么文件

- `reference_analysis/REPORT.md`：0.935版错误计数与标签/语境疑点。
- `reference_analysis/error_groups_for_review.csv`：25个错误组及逐组分析。
- `prepared/manual_rows.csv`：实际新增的人工训练行。
- `prepared/manual_label_notes.csv`：逐条标签理由与来源。
- `prepared/rejected_examples.csv`：被词汇、词干或重复检查排除的候选。
- `prepared/overlap_report.json`：全部重合计数、适用范围、文件SHA256。
- `prepared/training_data_manual.csv`：训练数据，预计3322行。

`analyze_reference.py`、`prepare_manual_data.py`只处理现有数据，不训练。

## 训练与对照

配置仍放在训练文件顶部，沿用原写法：

```python
RUN_NAME = 'run01_original_manual'
AUGMENT_MODE = 'manual'
```

默认仅运行这一次训练。模型目录：`D:/model/SEAMEVAL/deberta_manual_aug/<RUN_NAME>/`。
结果目录：本目录 `outputs/<RUN_NAME>/`，包含每轮预测、独立/结构指标和训练数据快照。
训练结束自动与0.935参考逐条比较，导出 `comparison_with_0p935/fixed.csv` 和 `broken.csv`。

如需匹配增加的训练预算，改为新RUN_NAME及 `AUGMENT_MODE='copy'`。
复制对照使用已有Train的相同类别和组数，不创造新文本；它不是词汇隔离的人工集，词汇可与Dev相交。
默认人工模式预计3322行、1392个真实配对组；每轮208个Row步骤、87个Pair步骤。
`AUGMENT_MODE='baseline'` 则只复现原3242行。

先看修好/破坏及类别变化，再决定是否保留。单种子在反复使用的Dev上涨分不能证明稳定泛化。
