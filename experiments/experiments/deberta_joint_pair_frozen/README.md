# 联合训练：冻结底部 6 层的对照

直接运行 `train_deberta_frozen.py`。从原 NLI 初始化重新训练，不读取旧 DiCo 最佳权重。

与 `experiments/deberta_joint_pair/run01` 的配方相比，只改变可训练参数范围：

- 冻结 embedding，包括相对位置 embedding 及其归一化。
- 冻结底部 6 层 Transformer，以及底部的卷积模块。
- 顶部 6 层、CLS 分类头、Token Alignment 正常训练。
- 冻结模块不进入优化器；不关闭它们的 dropout，避免同时改变训练模式。

数据、分组批次、随机种子、学习率、类别权重、10 epochs、float32 和损失保持不变：

```python
loss = row_loss + 0.5 * pair_loss
```

训练分组与官方评分直接复用现有实现；没有新增词汇分支，也没有扩充数据。
每轮 220 步，每步最多 16 行。Dev 推理固定 batch=16。

仍按独立 Dev Weighted F1 选最佳，同时报告同一轮的 pair-aware。
对照目标是当前联合版：独立 0.844739、同一轮结构后 0.888315。
原联合版结构最高 0.896237 来自第 7 轮，是另一个模型状态，不能拼入第 5 轮成绩。

模型仅保留一份最佳权重：

`D:/model/SEAMEVAL/deberta_joint_pair_frozen/run01_freeze6/best_model.pth`

小型预测和成绩在本目录 `outputs/run01_freeze6`。
重新预测最佳模型运行 `evaluate_best.py`。
重复实验先改 `config.py` 的 RUN_NAME，以保留之前的结果。

CPU 检查运行 `test_cpu.py`：检查初始预测不变、冻结参数不更新、顶部和分类头能更新。
冻结不保证涨分；若训练拟合下降但 Dev 不改善，就不支持该容量限制方案。
