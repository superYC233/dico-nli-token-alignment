# 官方3042 + Lexical + Row参数锚定

直接运行新增的 `train_deberta_lexical_anchor.py`。原 `train_deberta_lexical.py` 和 `config.py` 不修改。

```python
RUN_NAME = 'run06_row_parameter_anchor'
ANCHOR_STRENGTH = 1000.0
```

## 基准保持什么

- 只读官方Train3042行和Dev660行：Train F881、B881、E802、N478，真实Pair1282组。
- 不加载人工增强、反向N、SAM、BLOOP或旧最佳模型续训。
- 原NLI DeBERTa-base + CLS + Token Alignment + lexical_v1的20→32→4分支，新增分支804参数。
- 原模型学习率2e-5，词汇分支1e-3；AdamW、weight decay0.01、float32、seed42、10轮。
- Row权重[1,1,1,1.5]；Pair权重[1,1,1.25]、系数0.5、label smoothing0.05。
- 固定padding128，不启用动态padding、SAM或梯度检查点。两方向分别使用自己的词汇特征。

数据及特征源码哈希与旧0.930569运行核对，特征不读取标签、pair_id或组大小。新入口的配置写在文件顶部，不依赖旧config.py中的RUN_NAME和可调训练超参数。

## 一轮流程

1. 完整Row CE，直接复用原Lexical的Row函数，191次AdamW更新。
2. 立即复制当前所有可训练参数为本轮锚点，并评估Dev。
3. 原三组合Pair CE，81次AdamW更新；每次更新后向固定锚点做一次近端回拉。
4. 评估Pair后的Dev。下一轮完整Row结束后重新创建锚点，不把上一轮锚点带入Row训练。

共享的AdamW动量保留原流程；锚定不加入它的m/v，不清空优化器。锚定覆盖主干、CLS、alignment、词汇分支，包括当前Pair批次没有梯度的可训练参数。

## 回拉公式和默认强度

概念上的惩罚为 `mu/2 * sum((theta - theta_Row)^2)`，使用全局平方和，**不是除以1.84亿参数的平均值**。

令u为原0.5×Pair CE的AdamW候选参数，lr为该参数组学习率：

```python
theta = theta_Row + (u - theta_Row) / (1 + lr * mu)
```

这是候选参数的平方距离近端步。代码先做原Pair反向和AdamW，再无梯度回拉；不把概念惩罚与Pair loss相加后再次反向，也不额外更新一次AdamW。

首轮固定mu=1000，未经Dev选择：

| 参数组 | lr | 每步保留候选偏移 | 朝锚点拉回 |
|---|---:|---:|---:|
| 主干/CLS/alignment | 2e-5 | 0.980392 | 约1.96% |
| Lexical分支 | 1e-3 | 0.5 | 50% |

不同回拉比例来自原来不同的学习率与同一个mu。仍允许Pair学习，参数不冻结；这不是硬约束，也不保证单行输出或Dev F1不退化。

`ANCHOR_STRENGTH=0`关闭回拉，提供同代码原Pair路径对照。只运行一个配置，不自动扫强度或运行额外训练。

## 与RecAdam的关系

[RecAdam（EMNLP2020）](https://aclanthology.org/2020.emnlp-main.634/)及[作者代码](https://github.com/Sanyuan-Chen/RecAdam/blob/master/RecAdam.py)提供参数保持的思路。原论文使用预训练参数和目标权重退火；本任务改为每轮Row后参数、固定强度、AdamW候选后的近端步。因此这是**任务适配的参数锚定实验，不是RecAdam原论文复现**。

与之前的四类KL保持不同，本次惩罚参数偏移。它不会补充新的词义或语境，也没有已验证的0.95保证。

## 输出与检查

模型：`D:/model/SEAMEVAL/deberta_lexical/run06_row_parameter_anchor/best_model.pth`。

结果：本目录 `outputs/run06_row_parameter_anchor/`。仍按Pair后Dev pair-aware W-F1保存当前最佳，覆盖更新一份完整权重；同时记录同一模型的independent，不拼接不同轮指标。保存时临时文件原子替换；训练结束重载并核对最佳Dev逐条预测，再评估Train。

- `stage_summary.csv`：两阶段指标。
- `stage_changes.csv`：Pair修好/破坏、非N→N、各类变化。
- `anchor_batches.csv`：每批回拉前后参数距离，分别记录原模型和词汇分支；日志中的惩罚仅是诊断值，不是传入Adam反向的合并loss。
- `train_features.csv`、`dev_features.csv`、数据及源码哈希：复核数据与特征。

不增加编码器前向/反向：每轮仍5606行编码、272次AdamW更新；新增约0.74GB参数备份和参数距离计算。只保存模型，不保存优化器/本轮锚点的完整续训状态。

`evaluate_lexical_anchor.py`单独加载最佳模型，预测不需要锚点，也不强制单例为N。

`check_lexical_anchor.py`检查公式、参数组、数据/特征匹配、原Row复用、强度0与原Pair梯度一致和参数回拉时序；`--gpu`检查真实模型前后向，不运行训练优化器更新。正式训练由你运行。

本次语法、prepare-only、CPU公式/梯度检查和真实GPU检查均已通过。GPU使用184002604参数模型，Row及两个Pair方向各16×128，包含完整参数锚点及两份等量Adam矩缓冲，峰值allocated约9.44GiB。缓冲用于模拟显存占用，没有执行AdamW更新，不能将其视为完整训练全程峰值测量。CPU中的非零预设位移已验证回拉公式；GPU无更新检查前后模型参数完全一致。
