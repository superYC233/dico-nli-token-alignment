原0.938模型 + Row阶段PAWS
=======================

运行`train_original_staged_paws.py`。
使用已有2000行PAWS训练池，不需要重新下载，也不改变旧模型与旧实验。

这次恢复的是原来的模型和训练安排
--------------------------------

- `DebertaDiCoModel`：CLS四类logits + Token Alignment四类残差，相加后分类。
- 没有双向联合头、LoRA或新的推理结构。
- 原3290行训练快照，seed42，全量微调，lr2e-5，10轮，float32。
- Row batch16、Pair batch16，固定max_length128，AdamW weight_decay0.01。
- Row类别权重[1,1,1,1.5]。
- 原Pair类别权重[1,1,1.25]、系数0.5、标签平滑0.05，全部保留。
- 从相同NLI底座重新初始化，不加载0.938权重继续训练。

一轮流程
--------

    206批Row CE：全部3290行各一次，包括530条N
      每第4批，把一批PAWS的0.1倍梯度合入同一次更新
                 ↓
    评估Dev：保存after_row的独立/结构分及逐行预测
                 ↓
    87批原Pair CE：1380个FB/BF/EE组，无PAWS、无N单例
                 ↓
    再评估Dev：保存after_pair结果，并记录Pair修好/破坏多少行
                 ↓
    只按after_pair的结构F1选择最佳模型

每轮仍是293次optimizer.step，其中Row阶段51次更新带PAWS，约408条PAWS样本曝光。
PAWS每条文本的AB、BA都由原模型各自前向，两个方向的二元CE取平均。
辅助目标仍是E versus logsumexp(F,B,N)，非等价不指定为N。
PAWS反向前先释放Row图，PAWS后才执行这次optimizer.step。

数据和随机性
------------

复用`outputs/paws_prepared_seed42/paws_train.csv`，保持相同2000行与来源审计。
Row/Pair shuffle与原0.938一样使用全局torch RNG，PAWS sampler使用独立generator。
PAWS前向会消费额外dropout随机数，因此PAWS_WEIGHT>0的后续轨迹不必与旧版相同。
权重0时不读取PAWS、不做PAWS前向；新增阶段评估使用fork_rng保护训练随机数。

注意：本实验只检验PAWS能否改善原模型的训练特征和最终结构成绩。
原Pair CE只约束三种组合，其单行校准风险没有被修改；请同时查看两个阶段的独立F1。
训练期的PAWS小权重不代表已经解决泛化或能保证超过0.938。

文件位置
--------

默认RUN_NAME：`run04_original_staged_paws_seed42`。

- 模型：`D:/model/SEAMEVAL/deberta_bidirectional/<RUN_NAME>/best_model.pth`
- 结果：当前目录`outputs/<RUN_NAME>/`
- `epoch_N/after_row/`、`epoch_N/after_pair/`：两阶段的Dev预测、官方分数和混淆矩阵。
- `stage_summary.csv`：每轮两个阶段、两种解码，共4行。
- `epoch_summary.csv`：只列Pair结束后成绩，便于和旧版比较。
- `stage_changes.csv`：Pair前后修好/破坏、N变化等。
- `paws_exposure.csv`：实际PAWS source_id、正负样本曝光。
- `best/dev/`、`best/train/`：最佳Pair后模型的最终Dev/Train评估。

只保存一份最佳Pair后完整权重，不保存每轮或Row阶段权重。
独立重载入口是`evaluate_original_staged_paws.py`，不要使用联合头版本的加载器。

检查入口`check_original_staged_paws.py`，加`--real`检查真实NLI模型。
训练入口加`--prepare-only`可以检查实际数据和配置，不训练。

2026-10-24检查结果：真实2000行PAWS复用与启动检查通过；全部可训练参数184001800，
与原0.938相同。小模型上逐参数核对了权重0时的Row梯度、原Pair梯度及Row+PAWS梯度合并；
真实NLI模型的PAWS前向/反向和完整权重重载通过。没有执行任何优化器参数更新，
也未创建run04正式训练模型目录。
