# 词汇分支版：两个训练阶段的诊断

参考当前词汇分支最佳运行 `deberta_lexical/run02_branch_lr/lexical_seed_42`（Dev pair-aware W-F1 为 0.930569）。
新目录独立保存代码、权重和结果，沿用原模型、20 维特征、数据、损失、两个参数组学习率和训练顺序。
本实验用于观察训练过程，不承诺提高成绩。

## 运行

在 PyCharm 中直接运行 `train_deberta_stage_probe.py`。
从项目根目录也可以运行：

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments/deberta_lexical_stage_probe/train_deberta_stage_probe.py
```

每轮顺序：

1. 原版 Row CE 更新一遍 Train。
2. 新增：预测 Dev，分别算 independent / pair-aware 成绩，写入 `after_row`。
3. 原版 Real Pair CE 更新一遍真实配对组。
4. 原版：预测 Dev，写入 `after_pair`，按此时的 pair-aware W-F1 选择最佳权重。

从 NLI 预训练主干重新开始，不加载已经训练好的 0.930569 权重接着训练。
只保存 after_pair 阶段的最佳完整模型。after_row 只写预测和成绩，不保存模型、不参与选模。
新增评估保护 Python、NumPy、PyTorch CPU/CUDA 的随机状态，评估后恢复训练模式。

参数写在 `config.py`：主干学习率 2e-5，词汇分支 1e-3；10 轮，seed=42，训练和推理 batch=16，float32。
再次运行时更改 `RUN_NAME`，已有运行目录不会被覆盖。

## 重点看哪些文件

输出根目录：`experiments/deberta_lexical_stage_probe/outputs/run01/lexical_seed_42/`。

- `stage_summary.csv`：每轮两个阶段、每阶段两种解码的 W-F1 和各类正确数。
- `phase_effects.csv`：阶段前后的 W-F1、正确数变化、修好/破坏/改后仍错的数量，以及 E→F/B、F/B→E、N 错误的数量。
  - `training_phase=pair_ce`：本轮 after_row → 本轮 after_pair，共 10 轮。
  - `training_phase=row_ce`：上一轮 after_pair → 本轮 after_row，共第 2–10 轮。第 1 轮没有训练前评估，故没有该项。
- `epoch_*/after_row/`、`epoch_*/after_pair/`：预测分数、官方评分及混淆矩阵。
- `epoch_*/pair_changes/`：该轮 Pair CE 修好和破坏的具体样本；分别保存 independent / pair-aware 两种口径。
- `epoch_*/row_changes/`：第 2 轮起，该轮 Row CE 修好和破坏的样本。
- `epoch_summary.csv`：只含 after_pair，保留原版逐轮比较口径。
- `reference_reproduction.csv`：本次 after_pair 与旧词汇分支相同轮次的逐条预测差异及最大 logit 差异。
- `reference_protocol_check.json`：两次运行的数据和训练参数是否一致。

新的最佳完整权重：`model/deberta_lexical_stage_probe/run01/lexical_seed_42/best_model.pth`。
训练结束时内存中仍是最后一轮；重新加载最佳模型运行本目录的 `evaluate_best.py`。

## 怎么解释

先检查同轮 after_pair 是否复现旧词汇分支。若不同，需要先考虑软件、硬件、随机性或配置变化，不能直接归因于训练阶段。
然后看 Pair CE 是否反复降低 E 或 N 正确数，或者只是换了一批错误。一次阶段下降不足以证明整个训练策略无效。
本实验增加的 Dev 使用属于开发诊断；结果仍来自已经反复使用的 Dev，不是独立测试证据。

`analyze_stages.py` 可直接运行，使用已保存的 CSV 重建诊断表，不加载权重、不训练。
`check_setup.py` 仅检查真实数据与特征；`test_cpu.py` 用微型随机模型验证新增评估不会改变参数更新，不跑真实 GPU 训练。
