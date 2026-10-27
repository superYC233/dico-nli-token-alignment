# 原版 DeBERTa：同一步 Row CE + Pair CE

运行 `train_deberta_joint_pair.py`。训练完成后，运行 `evaluate_best.py` 重新加载最佳权重。

原版结构保持不变：NLI DeBERTa + CLS 分类头 + Token Alignment，没有词汇分支。
从 NLI 初始化重新训练，不加载已选中的 DiCo epoch_7。原训练文件和模型不修改。

## 核心变化

原版是整轮 Row CE，然后整轮只训练 Pair CE。

本版把每个配对组的两行绑定，N 单例也作为一个组，所有组一起打乱。
每批 8 组，最多 16 行，每个 batch 一次前向得到全部行的四类 logits。

```python
row_loss = RowCE(all_logits, all_gold_labels)
pair_loss = PairCE(FB_BF_EE_scores, pair_gold_labels)
loss = row_loss + 0.5 * pair_loss
loss.backward()
optimizer.step()
```

配对行同时接受四分类和组合监督；N 单例接受四分类监督。
某批全是单例则只算 Row CE。四分类归一化包含 N，Pair CE 仍是 FB/BF/EE 三组合。
完整训练集每轮遍历一次：3042 行、1282 个配对组、478 个单例，共 1760 组，220 步。
不增加数据、不额外采样、不使用 Dev 训练。

学习率、类别权重、Pair 权重、label smoothing、10 epochs、float32 沿用原版。
相比原版，训练顺序、行暴露次数和总更新步数发生变化，因此这是新训练配方，不能把差值只归因于一种梯度机制。

## 保存与评估

- 默认按 **独立 Dev Weighted F1** 选最佳，每轮同时报告 independent 和 pair-aware。
- 模型仅保留 `D:/model/SEAMEVAL/deberta_joint_pair/run01/best_model.pth`，最佳更新时替换。
- 小型预测 CSV、官方分数和曲线保存在本目录 `outputs/run01`。
- `best_summary.csv` 的两行来自同一个被选中的模型。
- `epoch_summary.csv` 保留所有轮次的两个分数。不能把不同轮次的最好成绩拼成一个模型的表现。
- 改参数或重跑时，更改 `config.py` 的 RUN_NAME，避免覆盖已有运行。
- 推理 batch 固定 16，max_length 128。官方评分和 pair-aware 复用现有模块。

这与旧 `deberta_lexical_joint` 不同：旧版仍先完整 Row CE，再在 Pair 阶段加 0.1 倍随机 Row CE；
本版从头至尾同一步优化两个目标，对配对的原始两行也施加强度为 1 的 Row CE，且不启用 lexical。

CPU 自检可运行 `test_cpu.py`，不下载模型、不加载大权重、不使用 GPU。
代码检查不能保证新配方超过 0.9248 或 0.9306，需要看实际两种解码成绩。
