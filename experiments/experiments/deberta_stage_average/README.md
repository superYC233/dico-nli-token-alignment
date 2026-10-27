# 阶段诊断 + 第6—10轮等权平均

直接运行 `train_deberta_stage_average.py`。检查程序是 `check_setup.py`，不会训练DeBERTa。

## 本次做什么

- 从同一NLI预训练初始化重新训练，沿用0.938145版的模型、float32精度和超参数。
- 直接读取 `experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv`：官方3042 + 旧scope200 + 新48，共3290行，核对历史哈希；不重新生成增强，不加载旧最佳权重续训。
- 每轮仍先完整Row CE，再完整Pair CE；同一AdamW，Row206步、Pair87步。
- 每轮在Row结束后、Pair结束后分别预测Dev，记录独立预测和pair-aware。
- 第2轮两个阶段各额外预测完整Train，确认训练行的四分类读出是否被Pair阶段改变。
- 第6—10轮各取一次**完整Pair阶段结束后**的参数，固定等权平均。
- 平均副本一直放在CPU。最终只将平均参数装入现有GPU模型，不创建第二个GPU模型。

这是固定学习率下的后期权重平均对照，没有加入SWA学习率调度，也没有采用EMA。
平均副本首次update才计数，第1—5轮和初始化参数不参与平均。
模型使用LayerNorm而没有BatchNorm，无需额外重估BatchNorm统计。

## 运行

```powershell
G:\anaconda3\envs\nlpbase\python.exe G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_stage_average\train_deberta_stage_average.py
```

后续复验只改脚本顶部 `RUN_NAME`（换种子时再改 `SEED`）。
`--prepare-only` 只复制和核对数据，不加载模型。正式运行需要本机已有的DeBERTa缓存，与之前同一路径。

## 保存位置

模型：`D:/model/SEAMEVAL/deberta_stage_average/run01_late_average/`

- `best_model.pth`：按Dev **Pair阶段结束后的**pair-aware F1选择的最佳单轮，只在刷新纪录时覆盖。
- `averaged_model.pth`：固定第6—10轮平均的模型，最后保存一次。
- `tokenizer/`、`run_config.json`、`best_metrics.json`。

两份权重合计约1.47GB，覆盖最佳权重时另有临时文件，建议模型盘至少留约2.3GB空闲。
CPU平均副本的参数约736MB；这不包含在线模型、编码数据和优化器等其他内存。
原模型和原结果不覆盖。没有保存每轮的完整权重，也没有额外保存第10轮的权重。
两份pth均是普通模型state_dict，没有`module.`前缀；预测只运行一个模型。

结果：当前目录 `outputs/run01_late_average/`。

| 文件 | 看什么 |
|---|---|
| `final_comparison.csv` | 最佳单轮、第10轮、固定平均模型的独立/结构成绩、N和F/B/E正确数、非N→N数量 |
| `epoch_summary.csv` | 每轮Row后/Pair后Dev指标，第2轮还包含Train；用phase和split区分 |
| `phase_changes.csv` | Pair阶段相对Row阶段修好/破坏多少行，各类正确数和非N→N变化 |
| `trajectory_comparison.csv` | 每轮Pair后的预测相对历史0.9381那次有多少不同；用于检查额外评估是否改变在线轨迹 |
| `epoch_2/after_row/train/`、`epoch_2/after_pair/train/` | 同一轮两个阶段的完整Train预测、混淆矩阵 |
| `averaged_final/dev/` | 固定平均模型的预测、官方评分、混淆矩阵和解码改动统计 |

额外评估会恢复模型原来的train/eval状态及PyTorch的CPU/GPU随机数，不改变参数。
平均副本在CPU模型初始化后复制，不引入新的随机参数；在线模型继续原来的优化过程。
`trajectory_comparison.csv`仍应实际检查；相同seed不承诺跨硬件/环境逐位一致。

## 怎么判断

1. 先看第2轮Train及各轮Dev：若Row后独立预测较好，Pair后非N→N激增，支持Pair阶段导致读出漂移。
2. 若Row后已经失衡，就不能把问题全部归到Pair阶段。
3. 看平均模型的独立与结构成绩、N正确数、非N→N及F/B/E正确数，不能仅凭曲线平滑或N召回上升判有效。
4. 最佳单轮用Dev选过，第10轮和平均窗口提前固定；两种选择协议分别注明，不把它们的差值当作无偏泛化估计。
5. 平均窗口不按Dev重新挑选。该实验不消除此前方法开发反复使用Dev的影响，也不能替代跨种子验证。

本次不新增样本或标签。旧scope200与官方Train仍可能和Dev共享词汇；新48行的词形/词干/短语隔离沿用原审计，不声称全部3290行与Dev词汇无交集。
