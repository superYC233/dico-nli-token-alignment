# DeBERTa + Lexical + KIM + 上下文门控

在 `deberta_kim` 版本上只增加门控，新目录独立运行，旧代码和权重不改动。
从同一 NLI 预训练主干重新训练，不加载 KIM 的最佳 checkpoint 继续训练。

## 直接运行

右键运行 **train_deberta_kim_gate.py**。默认不需要传参数。

使用现有 nlpbase 环境、缓存模型和项目 `.nltk_data` 中的 WordNet。
如果提示没有 WordNet，先运行 `prepare_wordnet.py`。
训练后重新预测最佳模型，运行 `evaluate_best.py`。

## 门控怎么工作

对每个有词典关系的词对，使用当前两个词的上下文表示和四维关系，分别计算四个 0~1 权重：

```text
g[i,j] = sigmoid(Ws * context[i] + Wt * context[j] + Wr * R[i,j] + b)
有效关系 = g[i,j] * R[i,j]
局部知识 = sum_j alignment[i,j] * 有效关系[i,j]
```

上下文使用原有 128 维投影的归一化 token 表示，无新增主干前向。
正反方向共享门控参数，但分别输入各自有向关系，交换上下位通道。
门控零权重、零偏置初始化，初始权重为 0.5，新增 1044 参数；旧知识投影仍为 512 参数且零初始化。
初始化隔离 RNG，不改变旧参数及后续 shuffle/dropout 的初始随机序列。
不存在的关系乘任何门值仍为 0；padding 和特殊 token 不注入知识。
该门控是本项目的 KIM 式扩展，不是原 KIM 论文的完整复现。

门控学习的是“任务中如何使用关系”，不等同于词义消歧。
即使门值高，也不能据此宣称词典知识正确；门值低也不证明词典错误。

## 原设置保持一致

Train 3042，不增强；20 维原词汇分支；seed42；float32；10 epochs。
Row batch16、Pair batch16组、Dev batch16。
每轮完整 Row CE → 完整 Real Pair CE，原权重、平滑和 optimizer.step 顺序不变。
原词汇分支 lr=1e-3；主干、原分类/对齐头和知识投影 lr=2e-5。
本次 `run02_gate_lr` 只把门控单独设为 lr=1e-3；上一轮 run01 的门控 lr=2e-5，旧结果保留。
没有新增损失、词性标注、词义消歧、新 WordNet 关系或人工标签。
每轮评估 independent 和 pair-aware，按后者 W-F1 保存唯一最佳完整权重。
原子更新最佳文件时磁盘需要额外一份模型的空间。

模型：`model/deberta_kim_gate/run02_gate_lr/kim_learned_seed_42/best_model.pth`。
结果：本目录 `outputs/run02_gate_lr/kim_learned_seed_42/`。

## 看哪些结果

- `epoch_summary.csv`：各轮 W-F1、N_correct、E_correct 和错误方向。
- `epoch_N/gate_statistics.csv`：只统计实际词典边，包含两个方向的各关系门控均值和边数；按子词边统计，不是词义可信度。
- `comparison_with_lexical`：相对 0.930569 原词汇版的修好/破坏样本。
- `comparison_with_kim`：相对 0.920039 无门控 KIM 的修好/破坏样本。

对照路径只用于训练完成后的分析，不进入训练或选模。

## 可选对照，不会自动启动

初始门控 0.5 也会减弱知识强度，所以只比较有无门控不足以证明“学会识别噪声”。
同一脚本提供：

```powershell
# 固定 0.5：检查收益是否只是整体减弱知识强度。
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_kim_gate/train_deberta_kim_gate.py --gate-mode fixed

# 恒为 1：原无门控 KIM 的同代码对照。
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_kim_gate/train_deberta_kim_gate.py --gate-mode open
```

三种模式保存到不同目录。重新评估时带相同 `--gate-mode`。
先运行默认 learned 一个实验即可；不要提前假定它会提升。

## CPU 验证

`test_cpu.py` 仅用随机微型 DeBERTa、缓存 tokenizer 和 WordNet。
验证门控响应上下文、两阶段损失的门控梯度、初始行为一致、
禁用知识模块等价于原 Lexical、全开门控等价于原 KIM，
以及完整 660 条官方评分、统计、RNG 恢复、保存/严格重载。
不加载真实 DeBERTa 大权重，不启动正式 GPU 训练。

## 第 8 轮模型的免训练消融

右键运行 `evaluate_gate_ablation.py`，默认加载 run02_gate_lr 的 best_model.pth（本次第 8 轮）。
只加载一次模型、准备一次数据，然后固定同一份 Dev 顺序和训练时的推理 batch：

1. learned：正常门控；必须先逐条复现该 checkpoint 的历史 independent / pair-aware 预测及最佳分数。
2. fixed_half：推理时把所有门值改为 0.5。
3. fixed_one：推理时把所有门值改为 1，知识全部通过。
4. no_knowledge：推理时跳过局部知识投影，保留原 20 维词汇分支。

不会更新参数、保存新模型或修改原模型文件。各分支结束后恢复原计算模式。
历史正常预测未复现时停止，不用不一致的基线解释消融。
四种结果保存到该运行结果目录下新的 `gate_ablation_年月日_时分秒/`。
其中 summary.csv 汇总指标，changes.csv 报告各消融相对 learned 的修好/破坏数量，
各子目录保存逐条预测、混淆矩阵和改动样本。

这些结果只回答“当前已训练权重在推理时是否依赖知识/动态门控”。
关闭知识不等于重新训练的原 Lexical；固定门控也不等于用固定门控重新训练。
不能把结果差直接叫做知识的完整训练贡献，也不自动把消融中分数最高的方式宣布为新方法。
若使用另一组已训练 learned 模型，可传 `--model-dir`，同时需要保留该次历史预测和配置。
