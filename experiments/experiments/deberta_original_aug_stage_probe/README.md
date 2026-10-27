# 原版 + 增强：分阶段检查独立判断

右键运行本目录的 `train_stage_probe.py`。训练由用户执行。

本版以当前 Dev 最高的 0.935493 配方为参照：原版 DeBERTa + Token Alignment、
原始 3042 行 + 同样 200 行 scope 增强、全参数 float32、lr=2e-5、batch16、10轮。
Row 和 Pair 损失、类别权重、优化器及先 Row 后 Pair 的顺序均沿用原版。
不会加载 0.935493 的权重续训，诊断中间过程需要从相同初始化重新训练。

## 唯一新增的观测

每轮完整训练 Row CE 后，预测一次 Dev；然后训练原有 0.5 × Pair CE，再预测一次 Dev。
两个点都报告 independent 和 pair-aware，并保存逐条分数、混淆矩阵、修好/破坏清单。
还会比较上一轮 Pair 后与本轮 Row 后，观察四分类训练是否修复了上一阶段的退化。

额外评估前后恢复 Python、NumPy、PyTorch、CUDA 和 Dev DataLoader 随机数状态。
这尽量保持训练随机序列，并不保证 GPU 浮点结果逐位一致；结束后自动核对旧运行同轮预测。
模型仍只按 **after_pair 的 pair-aware F1** 保存一份最佳权重；after_row 只诊断，不改变选模协议。
不额外保存每轮模型，也不保存 after_row 的模型权重。

## 目录

- 顶部只需换 `RUN_NAME`；默认 `run01_scope`。本次保持 `AUGMENT_MODE='scope'`。
- 模型：`D:/model/SEAMEVAL/deberta_original_aug_stage_probe/run01_scope/best_model.pth`。
- 结果：本目录 `outputs/run01_scope`。
- 其余训练配置复用 `experiments/deberta_original_aug/train_original_aug.py`；本次不要同时修改原版参数。

## 跑完发什么

优先发送最后打印的“每轮 Row / Pair 前后对比”，或 `stage_comparison.csv`。
`REPORT.md` 会提取原协议最佳轮的独立 F1、非N→N和N正确数的前后变化。

若 Row 后较好、Pair 后下降，说明本次 Pair 更新阶段损害了独立四分类；
若 Row 后已经差，需要继续检查 Row 阶段和上一轮的遗留影响。
这些是阶段观测，不能单独断言过拟合或某个机制已经成立。

只准备数据、不训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_original_aug_stage_probe/train_stage_probe.py --prepare-only
```

重新分析已保存结果、不训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_original_aug_stage_probe/analyze_stages.py
```

Dev 仍用于开发；新增模板标签尚未独立审核。此运行不是新增种子，也不构成独立测试。
