# run02词汇版：两阶段诊断与LLRD

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

单模型，无文本增强，无模型集成。从NLI初始化重新训练，不加载历史第3轮接着训练。
直接复用[原版模型](../deberta_lexical/model.py)、[20维词汇特征](../deberta_lexical/lexical_features.py)、[数据集类](../deberta_lexical/data.py)和[评分／解码](../deberta_lexical/evaluation.py)。

## 怎么运行

1. 可先运行`check_setup.py`：CPU检查历史参数、官方数据、内部划分、学习率分组和训练轨迹；不下载或加载真实权重。
2. **右键运行`run_comparison.py`**：先训练`decay=1.0`，结束后训练`decay=0.95`，分别保存最佳模型和验证成绩。训练由你运行。
3. 只跑一组时，修改`config.py`的`LAYERWISE_DECAY`，运行`train_deberta_llrd.py`。
4. 重新评估最佳模型：运行`evaluate_best.py`，使其`--decay`与对应实验一致。

已有本项目NLI DeBERTa缓存即可，不需要新下载。模型目录默认`model/deberta_lexical_llrd`，可在`config.py`修改。完成的组会被`run_comparison.py`跳过；未完成的旧目录不会覆盖，另改`RUN_NAME`后重跑。

```powershell
# 可直接右键run_comparison.py；下面是等价命令。
python experiments/deberta_lexical_llrd/run_comparison.py

# 仅检查官方数据，不训练。
python experiments/deberta_lexical_llrd/train_deberta_llrd.py --prepare-only

# 只用Train划内部验证，选参数时不读取官方Dev。
python experiments/deberta_lexical_llrd/run_comparison.py --validation internal

# 复查0.95组保存的最佳权重。
python experiments/deberta_lexical_llrd/evaluate_best.py --decay 0.95
```

## 恢复的run02配方

| 项目 | 固定设置 |
|---|---|
| 初始化 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| 架构 | CLS＋Token Alignment＋Lexical，20维词汇特征，经32维隐藏层输出4类 |
| 数据 | 官方Track1英文Train 3042行；不加scope、relation、反向N、复制或理由标注 |
| 训练 | float32，全参数，seed42，10轮，max_length128 |
| batch | Row16行，Pair16组，验证16行 |
| 原模型／词汇分支LR | 2e-5／1e-3 |
| AdamW | weight_decay0.01，所有参数保留原来的衰减处理 |
| Row CE | F/B/E/N权重[1,1,1,1.5] |
| Pair CE | FB/BF/EE权重[1,1,1.25]，smoothing0.05，系数0.5 |
| 顺序 | 每轮完整Row CE→评估→完整Pair CE→评估，同一个模型和优化器 |
| 选模 | 只按Pair后验证pair-aware Weighted F1选最佳，同分取最早轮 |

本目录不读取后来改过的原版config中的训练超参数。历史0.930569是参考成绩，不保证本次重跑逐位相同。当前源码和软件环境可能影响训练轨迹。

## LLRD只改哪里

最高编码器层保持2e-5，向下每层乘`decay`；第l层（0到11）的LR为`2e-5*decay**(11-l)`，embedding为`2e-5*decay**12`。
CLS、Alignment以及没有层号的共享相对位置参数仍为2e-5，Lexical仍为1e-3。实际分组写入`learning_rates.csv`。

`1.0`分支保留原run02的两个优化器参数组及组内顺序；`0.95`才拆分编码器层。没有新增warmup、scheduler、梯度裁剪、AMP、冻结、额外损失或数据增强。
这是参考[作者LLRD实现](https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py)做的DeBERTa适配，顶层LR约定与作者最高层先乘一次decay的实现有所不同。

## 每轮留下什么

```text
outputs/run01_llrd1_official/seed42/
  run_config.json
  learning_rates.csv
  training_data.csv
  validation_data.csv
  validation_reference.csv
  epoch_summary.csv
  best_metrics.json
  best_summary.csv
  epoch03/
    after_row/
      predictions_with_scores.csv
      summary.csv
      independent/             # 官方评分、混淆矩阵
      pair_aware/
    after_pair/
      predictions_with_scores.csv
      summary.csv
      independent/
      pair_aware/
    pair_changes.csv           # Pair阶段修好／破坏、非N→N的变化
```

0.95组在`run01_llrd0p95_official`。每份CSV都有instance_id、pair_id、金标、四类原始logits、独立预测和pair-aware预测。两阶段各报两种F1，但不把不同阶段的最好成绩拼成一组。

模型目录只有`best_model.pth`一份完整权重，附带tokenizer、backbone配置、run_config和best_metrics，不保存每轮权重或优化器续训状态。刷新最佳时会短暂保留新旧两份文件，需要留出写入空间。
额外评估会保存、恢复Python／NumPy／Torch／CUDA随机状态，评估后恢复训练模式，避免改变后续shuffle和dropout。

## 验证集和实验记录

- 默认`official`：3042行训练，官方Track1英文Dev660行验证。按Dev选最佳，成绩属于开发集选择结果。
- `internal`：仅从官方Train划约20%的验证组。同来源pair_id与NFKC／大小写／空白归一化后的相同无序文本对合并划分，保证正反行和这些重复文本对不跨集合。固定split_seed42，两臂及后续训练种子共用划分。不读取官方Dev；它的分数不能与660行官方Dev成绩比高低。
- 内部检查只能证明上述分组规则下的隔离，不保证所有近义文本、同一原新闻或语义重复都已隔离。
- 参数固定后，用完整3042行重新训练，再在官方660行Dev比较；官方Train本身与Dev已有的重合不在这里擅自删除。
- `epoch_summary.csv`每轮4条记录：阶段×预测方式。`run_comparison.py`最后输出两组各自Pair后最佳轮，以及该轮Row后成绩，不做融合。

本轮清单：1.0对照→0.95实验→比较同轮两阶段、非N→N和E/F/B混淆；若出现改善，再固定设置补seed7等种子。暂不叠加Mixout或其他方法。
