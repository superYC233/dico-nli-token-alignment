原0.938配方 + 双向联合分类头
==========================

运行入口：`train_deberta_alignment_joint.py`。
之前LoRA版的入口、代码、模型和结果都保留，不要用LoRA入口跑这版。

保留的部分
----------

- 相同NLI初始化、相同3290行训练快照（官方3042 + scope200 + relation48）。
- 原CLS特征和完整Token Alignment特征处理：128维对齐投影、温度0.1，
  [token, aligned, difference, product]比较，mean/max pooling和原LayerNorm/dropout。
- seed42，全量微调（embedding与全部encoder层都训练），不使用LoRA。
- lr2e-5，10轮，AdamW weight_decay0.01，float32，固定max_length128。
- 四类CE权重[1,1,1,1.5]，不增加新的增强样本。
- 按原版Dev pair-aware W-F1最高轮选模；同时报告该轮独立F1，平局保留较早轮。

替换的部分
----------

    原来：
      CLS → 四类logits ─────┐
      Alignment → 四类logits┴→ 相加
      每轮先Row CE，再单独Pair CE

    现在：
      同一DeBERTa分别读取AB、BA
      每个方向：CLS768 + Alignment512 → 128维投影
      两方向的表征 → 联合头 → F/B/E/N
      原始每行都做四类CE，不再有单独Pair阶段

因此，原来的两个四分类输出层已经被联合头替换；不是把Pair损失换成另一项损失。
两个方向共享同一个编码器和Alignment模块，没有两个独立模型，也没有额外二元概率乘积。
F/B随交换互换，E/N分数不变；这保证交换一致，不保证金标判断正确。

每批8个无序文本对、16个编码器输入，复用已存在的反向文本；所有N也自行读取反向。
保留全部3290行的监督，N不额外加倍。由于训练对象改为联合视图，批次和更新顺序
不可能与原Row16→Pair16两阶段完全相同，这版不是严格的单变量消融。

独立预测不需要pair_id或另一条记录；模型输入始终只有正反文本。
评分阶段仍单独报告原三组合pair-aware，配对行排除N的收益必须与独立能力分开看。

结果位置
--------

默认RUN_NAME：`run02_original_alignment_joint_seed42`。

- 完整权重：`D:/model/SEAMEVAL/deberta_bidirectional/<RUN_NAME>/best_model.pth`
- 每轮CSV：当前目录`outputs/<RUN_NAME>/epoch_summary.csv`
- 最佳同轮结果：`best_summary.csv`、`best_metrics.json`
- 最佳模型Dev/Train预测：`best/dev/`、`best/train/`
- 单独重载评估：`evaluate_deberta_alignment_joint.py`

只在刷新最佳时覆盖同一份完整权重，不保留每轮模型或优化器。
这是全量微调，权重约0.74GB，不再是几MB的LoRA权重。
从NLI底座重新训练，不加载0.938的best_model继续训练。

检查
----

- `train_deberta_alignment_joint.py --prepare-only`：检查源码、数据及分词，不训练。
- `check_alignment_joint.py`：小模型CPU检查，不执行优化器更新。
- `check_alignment_joint.py --real`：加查本机真实NLI模型前向、反向与完整权重重载。

所有需要改的训练参数都在新训练入口顶部；后续同代码实验改RUN_NAME即可。

本机检查记录（2026-10-24）
-------------------------

已通过`--prepare-only`及`check_alignment_joint.py --real`，没有执行优化器参数更新。
已逐值验证新模型的CLS/Alignment特征与原版特征计算相同，并验证embedding、底层、
顶层、Alignment和联合头都能获得梯度。完整模型保存及专用加载器重载后输出一致。
实际全部可训练参数为184230147，其中encoder183831552、Alignment165120、联合头233475。
测试临时权重已经清理；未创建本次正式训练模型目录。
