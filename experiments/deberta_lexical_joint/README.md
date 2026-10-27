# 词汇分支：Pair 阶段加入少量四分类 CE

以 `experiments/deberta_lexical_stage_probe` 为基础，只改第二阶段的训练目标。
原代码、权重和 0.930569 模型均保留。本目录不使用 WordNet。

## 运行

在 PyCharm 中直接运行本目录的 `train_deberta_joint.py`，或从项目根目录运行：

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments/deberta_lexical_joint/train_deberta_joint.py
```

仍从同一个 NLI 预训练主干重新初始化，不从已训练的最佳模型接着训练。

每轮顺序保持：

1. 完整 Train 的 Row CE 更新，随后评估 Dev。
2. 真实 pair 的训练；每个 pair 批次额外抽一个完整 Train 的四分类批次：

```python
loss = 0.5 * pair_ce + 0.1 * auxiliary_row_ce
```

3. 再评估 Dev，仍按此时的 pair-aware W-F1 保存最佳模型。

两项梯度在同一次 optimizer.step 前累加。为了减少显存占用，先反传 Pair 损失，再计算并反传辅助四分类损失；中间不清空梯度，也不更新参数。

## 哪些设置保持

模型结构、20 维词汇特征、3042 条 Train、660 条 Dev、学习率、batch、seed、10 轮、float32、原有 CE 类别权重、Pair 标签平滑、pair-aware 解码和选模规则都沿用原版。

- 原模型学习率：2e-5。
- 词汇分支学习率：1e-3。
- Row 权重：[1, 1, 1, 1.5]。
- Pair 权重：[1, 1, 1.25]，label smoothing=0.05。
- Row batch=16，Pair batch=16 组，辅助 Row batch=16，Dev batch=16。

## 新增参数

在 `config.py` 中：

```python
PAIR_ROW_WEIGHT = 0.1
PAIR_ROW_BATCH_SIZE = 16
PAIR_ROW_SEED = SEED + 1000
```

辅助 loader 从完整官方 Train 中随机抽样，因此数据源包含 N；不保证每个批次都有 N，不按 Dev 错误筛选，也不做类别平衡采样。
使用独立的抽样 generator。每轮重新开始一次辅助遍历；遍历不足时重新打乱继续取。
当前真实 Pair 共 81 个批次，因此每轮第二阶段额外看到 1296 行已有 Train 样本。optimizer.step 数保持原样，计算量和样本曝光次数增加，不能称为等计算预算。
两次前向中的 dropout 仍按正常训练运行，因此非零权重实验不要求复现原训练随机轨迹。

将 `PAIR_ROW_WEIGHT=0.0` 并设置新的 `RUN_NAME`，可运行同代码的零权重对照。此时不创建/迭代辅助 loader，不执行额外前向。

## 保存和结果

- 权重：`model/deberta_lexical_joint/run01/lexical_seed_42/best_model.pth`。
- 结果：`experiments/deberta_lexical_joint/outputs/run01/lexical_seed_42/`。
- 只有 after_pair 的 F1 刷新最佳时才保存，始终保留一个最佳权重文件。
- 写入时临时多占一份权重空间；写完原子替换，避免磁盘满导致旧最佳损坏。训练前至少预留 2 GiB。
- 训练结束的内存模型仍是最后一轮，重新加载最佳权重请运行本目录的 `evaluate_best.py`。

重点看：

- `epoch_summary.csv`：分开记录第一阶段 Row CE、第二阶段 Pair CE、辅助 Row CE、第二阶段加权总损失，以及辅助样本 F/B/E/N 数量。
- `stage_summary.csv`：两个阶段的 independent / pair-aware 分数、E/N 正确数等。
- `phase_effects.csv` 和每轮的 `pair_changes/`：训练前后修好和破坏的样本。
- `comparison_with_lexical/summary.csv`：最佳新模型相对原 0.930569 最佳模型的改动；fixed/broken 文件分两种解码保存。
- `reference_protocol_check.json`：本实验应显示与旧版存在训练目标差异，不能当作原版逐条复现。

`after_pair` 在本实验表示“Pair CE + 辅助 Row CE”结束，沿用目录名便于和旧版对照。
主比较仍是最佳 pair-aware W-F1；独立 F1 上升而最终分数下降，只能说明单行输出改善，不能算整体提分。
这是一次训练对照，不保证超过 0.930569，结果仍是已经反复用于开发的 Dev 上的结果。

`check_setup.py` 只检查数据和特征。`test_cpu.py` 用微型随机模型检查梯度累加、零权重退化、完整 Train 抽样、保存失败保护和历史评分；不会加载真实预训练权重，不跑 GPU。
