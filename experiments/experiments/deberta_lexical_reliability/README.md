# 确认 0.9306 的可信范围：原版 / Lexical 配对复验

## 新模型目录

```text
D:\model\SEAMEVAL\deberta_lexical_reliability\run01\
    baseline_seed_7\best_model.pth
    lexical_seed_7\best_model.pth
    baseline_seed_21\best_model.pth
    lexical_seed_21\best_model.pth
```

运行训练时自动创建目录。各组只保存自己的最佳完整权重；不覆盖原 0.9248 / 0.9306 模型。
四份权重约 3 GB。D 盘空间充足，已恢复为每组保存最佳权重，无需删除旧模型。
代码和小型 CSV 在项目这个新实验目录。

## 怎么运行

**只分析已有 seed42（不训练）：右键运行 `analyze.py`。**

**正式配对复验：右键运行 `run_all.py`。**
依次运行 baseline7、lexical7、baseline21、lexical21，最后自动分析。
使用当前 nlpbase Python，单进程串行训练，预计约为原训练耗时的四倍。
每次都是从相同 NLI 预训练主干开始；不是加载之前最佳权重继续训练。
已完成且协议一致的组会跳过；未完成目录不覆盖、不自动删除。

也可只运行一组：

```powershell
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_lexical_reliability/run_one.py --seed 7 --arm baseline
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_lexical_reliability/run_one.py --seed 7 --arm lexical
```

运行开始后不要改配方/源码。若有意改变实验，使用新 RUN_NAME 建立新协议。
本次默认只准备好脚本和 CPU 检查，不自动替用户启动四次 GPU 训练。

## 保证比较的是什么

复用 `deberta_lexical/train_deberta_lexical.py` 原有训练主函数、模型、数据和评分代码。
子进程只把这些模块的配置对象指向这里的冻结配置，不修改原源码。
baseline 关闭词汇分支；lexical 开启 20→32→4 分支，并用其原来的 1e-3 学习率。
其他设置保持 0.930569 运行不变：seed 以外只有词汇分支开关不同。
保存源码/数据 SHA256 和协议，避免中途改配置造成不可比。
同种子保证共同参数的初始化及随机序列起点一致；参数更新不同后训练轨迹仍会分开。

10 epochs；Row CE→Real Pair CE；batch 16行/16组；Dev batch16；float32；lr2e-5；
Row 权重 [1,1,1,1.5]；Pair 权重 [1,1,1.25]；Pair 系数0.5；平滑0.05。
两组都在十轮内按 Dev pair-aware W-F1 选最佳，平分取最早轮次。
同时保留最终轮次成绩，便于检查选模带来的影响。

## 结果怎么看

`outputs/run01/analysis/per_seed.csv`：每个种子的原版、Lexical 和配对差值。
`seed_summary.csv`：均值、样本标准差以及配对差值的均值/标准差。
`bootstrap.csv`：按 Dev 383 个 pair_id 整组同步重采样 20,000 次的固定模型差值区间。

重点看**两个新种子的配对差值**，不要从种子里挑最高分作为方法结果。
主比较预先设为两个新种子的平均差值；每个种子的区间是辅助描述。
这些百分位区间没有多重比较校正，且不包含训练随机性。
历史 seed42 已参与超参数选择，单列标明，不当作新的独立确认。
所有结果仍在反复使用过的 Dev 上；不能由此宣称在未知测试集一定有相同分数。

判断的是两个问题：Lexical 的分数是否对种子敏感；相对于同种子原版是否稳定改善。
两对新种子只提供初步证据，没有固定数量能保证结论成立。
