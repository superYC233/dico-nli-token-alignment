# Lexical run02：Pair阶段只更新Alignment

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

直接运行 [train_deberta_pair_routing.py](train_deberta_pair_routing.py)。正式训练由你运行。

基线是**无增强的Lexical run02**，历史同一检查点 independent / pair-aware 为 **0.468191 / 0.930569**。本次从NLI预训练主干重新训练，使用官方Train 3042行、官方英文Dev 660行。

| 项目 | 本次设置 |
|---|---|
| 模型 | DeBERTa＋CLS分类头＋Token Alignment＋20维Lexical |
| 初始化 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli，需准备对应模型缓存 |
| Row阶段 | 全部参数更新；原加权四分类CE，权重F/B/E/N＝1/1/1/1.5 |
| Pair阶段 | 只更新align_projection、compare_layer、match_output |
| Pair冻结 | DeBERTa、CLS分类层output、Lexical分支lexical_output |
| Pair目标 | 原FB/BF/EE三组合CE，权重1/1/1.25，label smoothing 0.05，系数0.5 |
| 优化器 | 一个共享AdamW，状态跨阶段、跨轮保留 |
| 学习率 | 原模型2e-5；Lexical 1e-3；weight decay 0.01 |
| 训练 | 10轮，Row batch16，Pair batch16组，FP32，长度128 |
| 预测 | 原四分类和原pair-aware；单例仍由模型自己判断 |

一轮的流程：

```text
解冻全部参数
  → 全部3042行做Row CE（191步）
  → 评估Row后的independent和pair-aware，记录三分支logits
  → 清空梯度为None，冻结DeBERTa、CLS、Lexical
  → 全部1282个真实双行组做Pair CE（81步）
  → 再次评估，比较哪些预测改变、哪个分支改变
下一轮重新解冻全部参数
```

两个训练阶段都保留model.train()及原dropout。冻结参数的梯度设为None，AdamW跳过这些参数，不让旧动量或weight decay继续移动它们。Pair没有增加NN样本。

**先跑这一份**：

```powershell
python 'experiments/deberta_lexical_pair_routing/train_deberta_pair_routing.py'
```

默认模型保存在 `model/deberta_lexical_pair_routing/run01_alignment_only_official/seed42/`，结果保存在本目录 `outputs/run01_alignment_only_official/seed42/`。重复运行前在[config.py](config.py)修改RUN_NAME。

同代码的原run02共享优化器对照：

```powershell
python 'experiments/deberta_lexical_pair_routing/train_deberta_pair_routing.py' --pair-update all
```

对照保存到 `run01_all_official/seed42/`，不会覆盖alignment_only结果。当前建议先运行默认实验，不必一次把所有对照和种子都跑完。

调方法时可用 `--validation internal`：仅从官方Train按来源及重复整对文本划分，正反行不拆开。此模式不读取官方Dev；内部成绩与官方660行Dev成绩不可直接混比。

**模型怎么保存**：

- `best_pair_aware.pth`：仅比较每轮Pair后的结构F1，保留最高的一份。
- `best_independent.pth`：比较Row后和Pair后的独立F1，保留最高的一份。
- 只覆盖刷新纪录的最佳权重，不保存每一轮模型；两份最佳可能来自不同轮次或阶段。对应JSON分别列出该检查点的两种F1，不能把两份模型的最高分拼成一个成绩。
- 运行 [evaluate_best.py](evaluate_best.py) 可重新评估结构最佳；参数 `--kind independent` 评估独立最佳。完整模型权重、主干配置和tokenizer已保存在模型目录，重载无需重新下载主干权重。

**跑完发哪些结果**：

1. `epoch_summary.csv`：每轮两个阶段、两种预测方式的成绩。
2. `pair_changes.csv`：Pair修好/破坏多少行，非N判N和N正确数如何变化。
3. 每轮 `pair_branch_changes.csv`：CLS、Alignment、Lexical的分数变化。默认模式下CLS和Lexical的eval输出应不变。
4. 每阶段 `branch_logits.csv` 与 `predictions_with_scores.csv`：完整660行分支分数和最终预测；`stage_parameters.csv`列出实际更新范围。

本目录复用原Lexical的模型、数据集、词汇特征和官方评分函数；独立配置固定历史run02参数，原文件不修改。新增的是阶段参数路由与分支记录，没有文本增强或模型集成。

CPU检查入口：[check_setup.py](check_setup.py)。检查历史配置及数据哈希、all模式与原训练函数两轮逐参数一致、已有Adam动量下冻结参数/状态不变、Alignment更新、下一轮解冻、分支求和、评估随机数恢复及最佳模型保存重载。检查使用随机初始化的小DeBERTa，不启动正式训练，不加载真实预训练权重。

借鉴 [Gradient Routing（arXiv 2024）](https://arxiv.org/abs/2410.04332) 按监督来源限制反向更新的思想；这是针对本项目的模块冻结适配，并非该论文算法的完整复现。它能保护Pair阶段冻结的主路径，但Alignment仍能改变总预测，三组合CE的盲区仍在；实际收益由同轮结果判断。
