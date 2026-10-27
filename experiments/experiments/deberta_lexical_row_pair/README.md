# DeBERTa 词汇版：第二阶段同时监督单行与配对

以 `deberta_lexical_stage_probe/run01_rr/lexical_seed_42` 为对照，保留模型、
20 维词汇特征、数据、学习率、随机种子、批次和两阶段顺序。
不加载之前的最佳模型继续训练，而是从同一个 NLI 初始化重新开始。

本次唯一的训练干预：第二阶段在原有 Pair CE 上加回两行的四分类 CE。

```python
# 第一阶段：完整 Train，包含 N 单例，沿用原版。
loss = row_criterion(logits, labels)

# 第二阶段：真实配对组，同一模型分别预测正反两个方向。
row_logits = torch.cat([first, second], dim=0)  # [2 * batch, 4]
row_labels = torch.cat([labels0, labels1], dim=0)
row_loss = row_criterion(row_logits, row_labels)
pair_loss = pair_criterion(get_pair_logits(first, second), targets)
loss = row_loss + 0.5 * pair_loss
```

`row_loss` 对两个方向的全部行求一次带权平均，不是两个平均损失直接相加。
四分类分母包含 F/B/E/N，因此真实 F/B/E 行的 N 分数过高也会受到惩罚。
第二阶段没有 N 单例，仍可能通过共享参数影响 N 召回；这次改动不保证涨分。
新增 CE 复用原有两次前向，不新增模型前向，也不新增样本。

这与 `deberta_lexical_joint` 的旧实验不同：旧实验在 Pair 阶段另外抽完整 Train
的辅助批次、使用 0.1 倍辅助 Row CE。本实验直接监督配对的两行，Row CE 系数为 1。

## 运行

在 PyCharm 直接运行本目录的 `train_deberta_row_pair.py`。
也可以从项目根目录运行：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_row_pair/train_deberta_row_pair.py'
```

配置在 `config.py`：seed=42、10 轮、batch=16、float32、原模型学习率 2e-5、
词汇分支学习率 1e-3。Row 权重 `[1, 1, 1, 1.5]`、Pair 权重 `[1, 1, 1.25]`、
Pair label smoothing=0.05，均沿用对照。再次运行改 `RUN_NAME` 即可。

权重目录默认：

```text
D:/model/SEAMEVAL/deberta_lexical_row_pair/run01/lexical_seed_42/best_model.pth
```

结果目录默认：

```text
experiments/deberta_lexical_row_pair/outputs/run01/lexical_seed_42/
```

仍按每轮第二阶段后的 Dev pair-aware W-F1 选模，仅在刷新最佳时覆盖
`best_model.pth`。这样避免同时改变选模标准。`after_row` 只做诊断，不保存权重。
每个阶段都报告独立与 pair-aware 结果，最佳权重对应的两个分数一起写入
`best_metrics.json` / `best_summary.csv`，不能把不同轮的最高分拼成一个模型。

## 看哪些结果

- `stage_summary.csv`：每轮两阶段的独立 / pair-aware 分数，以及 F/B/E/N 正确数。
- `phase_effects.csv`：第二阶段修好和破坏多少条，非 N→N 与 F↔B 是否减少。
  第二阶段标记为 `training_phase=row_plus_pair_ce`。
- `epoch_summary.csv`：第二阶段后各轮结果，含三个分开的损失分量。
  `row_loss` 是第一阶段均值；`pair_stage_row_loss` 是第二阶段 Row CE；
  `pair_loss` 是乘 0.5 前的 Pair CE；`pair_stage_total_loss` 是联合损失。
  这些损失统计为各批次均值的平均。
- `epoch_*/after_row/`、`epoch_*/after_pair/`：预测、四类分数、混淆矩阵。
- `reference_prediction_comparison.csv`：与旧纯 Pair 版同轮预测的差异；不是复现检查。

重点比较第二阶段前后的独立 W-F1、F 正确数、非 N→N 数量，也观察 pair-aware
W-F1 和 N 正确数有没有损失。Dev 已反复使用，这仍属于开发实验。

训练后运行 `evaluate_best.py` 加载最佳权重重新评估。
`analyze_stages.py` 只读已保存的 CSV 重建诊断表。

## 不训练的检查

`check_setup.py` 只检查真实数据、配对及词汇特征。
`test_cpu.py` 检查标签传递、损失公式、旧 Pair CE 的盲区是否被新增 Row CE 约束，
以及没有顺带修改模型、优化器、第一阶段或评估逻辑。
不加载预训练权重，不调用 backward 或 optimizer.step。
`source_hashes.json` 记录创建新目录时旧目录文件的哈希，用于核对旧文件未被修改。
