# Row阶段四类输出保持

运行 `train_deberta_row_anchor.py`。CPU检查脚本为 `check_setup.py`，不训练DeBERTa。

## 每轮流程

1. 沿用0.9381设置，完整训练Row CE。
2. 以eval模式、无梯度预测Train，缓存每行F/B/E/N四类logits。
3. 缓存argmax与Train金标一致的行才参与保持约束。缓存固定到本轮Pair结束，下轮Row结束后刷新。
4. Pair阶段仍有原来的两个方向前向，损失为 `0.5 * Pair CE + 1.0 * keep_loss`。
5. `keep_loss = T² × KL(Row缓存分布 || 当前分布)`，T=2，归一化包含全部四类。复用已有Pair前向，没有额外教师模型。
6. 分别评估Row后和Pair后的Dev；第2轮额外评估完整Train两个阶段的表现。

本次不做权重平均，不加新样本，不改网络。仍从同一个NLI预训练模型初始化，不读取旧最佳模型续训。
训练数据直接复用 `experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv` 并核对哈希。
这是官方3042 + 旧scope200 + 新48，共3290行。旧scope和新48的隔离范围沿用原审计。

## 修改点在哪

`AnchoredPairDataset`在原有五个返回值后加两个原训练行索引。
`cache_train_logits`缓存本轮Row结束时的Train分数。
`compute_keep_loss`只对缓存答对的行计算四类KL，teacher目标detach，只有当前模型接收梯度。
`compute_pair_stage_loss`把Pair CE和保持项合并，`train_pairs`完成原来的87次更新。

批次顺序是“本批所有第一行 + 本批所有第二行”，缓存也按相同顺序读取。
缓存/评估会恢复模型模式和PyTorch CPU/GPU随机数。Pair和Row的shuffle方式、同一AdamW和学习率不变。
`KEEP_WEIGHT=0`跳过保持项，提供同代码的原损失对照；若跑对照须改新的RUN_NAME。

## 默认设置

| 项 | 值 |
|---|---|
| 模型 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli + 原Token Alignment |
| 精度 | float32 |
| seed | 42 |
| epoch | 10 |
| Row batch / Pair batch | 16行 / 16组 |
| 每轮更新 | Row206步 + Pair87步 |
| 学习率 | 2e-5 |
| Row权重 | [1,1,1,1.5] |
| Pair权重 / smoothing | [1,1,1.25] / 0.05 |
| Pair系数 | 0.5 |
| 保持系数 / 温度 | 1.0 / 2.0 |

`keep_loss`在本批符合条件的行上取均值；记录的各损失是每批损失的均值。
`weighted_pair_loss`与`weighted_keep_loss`分别展示两项乘系数后的量级。

## 模型和结果

模型目录：`D:/model/SEAMEVAL/deberta_row_anchor/run01_keep1_t2/`。

- `best_row_model.pth`：Row后Dev **independent W-F1**最高的模型。
- `best_pair_model.pth`：Pair后Dev **pair-aware W-F1**最高的模型。
- 两者采用不同选择指标，分别报告，不能把两种选择当成同一口径。
- 只刷新对应最佳模型，不保存每轮完整权重。两份权重约1.47GB，替换时存在临时文件。

结果在本目录 `outputs/run01_keep1_t2/`。

- `final_comparison.csv`：历史0.9381、最佳Row、最佳Pair的独立/结构成绩。
- `epoch_summary.csv`：每轮两个阶段指标，以及Pair CE、keep_loss和加权损失。
- `phase_changes.csv`：Pair阶段修好/破坏行数、各类正确数变化、非N→N变化。
- `cache_summary.csv`：每轮每类缓存答对数、温度软化后的最高概率。
- `epoch_*/after_row/train_row_cache.pt`：Train样本ID、无梯度logits、正确行mask；四类分数本体约53KB，另有ID等元数据。
- 第2轮 `after_row/train/` 和 `after_pair/train/`：完整Train预测和混淆矩阵。

缓存不含Dev分数，Dev只用于评估和两个明确的选模规则。
直接参与Pair保持的只有成对F/B/E训练行；N单例只在Row阶段训练，没有额外N回放。
因此该约束直接阻止配对行的N分数漂移，但不保证单例N不受共享参数更新影响，必须继续观察N正确数。

先比较Pair阶段是否还把大量正确行改成N，再比较结构成绩能否保住；不预设一定超过0.9381。
本方法与Pair阶段加入四类Row CE有相似性，区别是保留本轮Row阶段的软分布，且只约束Row已答对行。

后续复验主要修改顶部 `RUN_NAME`，`--prepare-only`只复制核对数据，不加载模型或训练。
