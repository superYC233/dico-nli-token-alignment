# 原0.938配方 + BLOOP

直接运行同目录的 `train_relation_bloop.py`。正式训练由你运行；检查脚本不更新参数。

## 基于哪一版

- 数据固定读取 `outputs/run06_relation48_staged/training_data.csv`：官方3042 + scope200 + relation48 = **3290行**。没有新增64行或外部数据。
- 模型仍为 **DeBERTa NLI-base + CLS四分类 + Token Alignment残差**，关闭词汇分支；全参数微调，不用LoRA、联合分类头或N门控。
- 从相同NLI预训练权重重新开始，不从按Dev选出的0.938权重续训。
- 保留原学习率2e-5、10轮、Row权重[1,1,1,1.5]、Pair权重[1,1,1.25]、Pair系数0.5及平滑0.05。

## 一轮做什么

1. **第一阶段：完整Row CE**。3290行、batch16、206次更新。照常反向传播，同时维护Row梯度的EMA。
2. **第二阶段：同批Row为主、Pair为辅**。1380个真实反向组各出现一次，530条原N单例均匀放入这87批；共3290行各一次，每批最多39行。
3. 同一次前向计算完整四类Row CE和三组合Pair CE，分别求梯度，BLOOP合成后交给AdamW。N单例只参与Row CE，未人为构造NN。
4. 两个阶段后都评估Dev；只按第二阶段后pair-aware W-F1保存当前最佳模型，同时记录同一模型的independent指标与非N→N数。

**第二阶段不再是原来的Pair-only。** 历史0.938每轮编码6050行，本版6580行；第二阶段同批拼接和动态padding也会改变计算/随机序列。因此和历史0.938的差异不能全归因于投影。代码提供相同批次、相同梯度计算、相同选模规则的 `MODE='mixed'` 对照，默认只跑 `bloop`，不会自动启动多次训练。

## BLOOP到底改什么

记当前Row梯度为 `gr`，**已乘0.5**的Pair梯度为 `gp`：

```python
# 第一次m=gr，后续每次Row/第二阶段更新都维护m，不在换轮时清空。
m = 0.9 * m + 0.1 * gr
c = dot(m, gp) / (dot(m, m) + 1e-7)
g = gr + gp - c * m
```

它从辅助Pair梯度中去掉沿主梯度EMA的分量。正、负内积都做投影；不同于旧PCGrad版只处理负内积，也不能把最后一项写成当前 `gr`。EMA平均的是**梯度**，不是SWA平均模型权重。初始化、`rescale=False`、`use_ratio_only=False` 对应作者基础设置；此处为独立PyTorch实现和本任务的两阶段适配。

这是交给AdamW之前的梯度处理，不能保证实际有限步更新后Row损失必降，也不能保证Dev F1提升。多维护约0.74GB的float32 EMA，第二阶段有两次反向；开启非reentrant梯度检查点和动态padding以控制12GB显存。GPU检查报告不含Adam状态，不能作为完整训练峰值。

## 文件与输出

- `bloop_gradients.py`：EMA和全模型投影。
- `train_relation_bloop.py`：训练入口；顶部改 `RUN_NAME`，`MODE` 默认 `bloop`。
- `evaluate_relation_bloop.py`：单独重载最佳模型预测。推理不需要EMA，仍是四类argmax + 原FB/BF/EE解码；单例由模型预测。
- `check_relation_bloop.py`：公式、真实数据覆盖、无参数更新的前后向检查；可带 `--gpu`。

默认模型：`D:/model/SEAMEVAL/deberta_manual_aug/run25_row_pair_ema_bloop/best_model.pth`。

默认结果：本目录 `outputs/run25_row_pair_ema_bloop/`。

`stage_summary.csv`比较Row/第二阶段；`stage_changes.csv`记录修好/破坏和各类变化；`gradient_batches.csv`记录梯度内积、EMA夹角和投影系数。只覆盖保存当前最佳完整权重，不逐轮保存大模型。配置及最佳指标同目录保存；模型检查点不含优化器/EMA，不能用于精确恢复中断训练。

本次已通过公式/覆盖/CPU梯度检查、`--prepare-only`启动检查，以及真实184001800参数DeBERTa的GPU双梯度反向检查。GPU检查选取固定批次方案里最长的一批（38行、26 tokens），峰值allocated约3.45GiB，未分配Adam状态；全程未执行真实优化器更新，也未启动正式训练。

## 来源

论文：[Careful with that Scalpel: Improving Gradient Surgery with an EMA，ICML 2024](https://proceedings.mlr.press/v235/hsieh24a.html)。

作者实现：[optim.py / bloop_direction](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/optim.py)、[utils.py / tree_proj和update_ema](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/utils.py)。
