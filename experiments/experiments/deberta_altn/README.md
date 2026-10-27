# DeBERTa Lexical + ALTN 式差异位置特征

从当前 Lexical `run02_branch_lr`（历史 Dev pair-aware W-F1=0.930569）复制改写，原目录和权重均保留。
本次从同一个 NLI 预训练主干重新训练，不加载已经选过 Dev 的最佳 checkpoint 继续训练。

## 直接运行

1. 可选：运行 `check_setup.py`，检查原始 Train 3042 / Dev 660 / 真实 Pair 1282 组和全部 28 维特征。
2. 正式训练：运行 `train_deberta_altn.py`。
3. 训练后可运行 `evaluate_best.py`，重新加载最佳完整模型评分。

默认只跑 ALTN 一个版本，不自动跑额外对照。配置均在 `config.py`。
要单独做同代码 Lexical 对照，将 `USE_POSITION_FEATURES=False` 后运行训练，或给训练命令加 `--lexical-only`。对照保留原 20 维词汇分支，只关闭新增 8 维；不是关掉全部 Lexical。
重跑时修改 `RUN_NAME`；已有输出目录会被保护，不会覆盖。运行正式训练前结束其他占用 GPU 的训练进程。

## 仅新增的内容

原词汇分支仍是 20 -> 32 -> 4（804 个参数）。增加零初始化的 8 -> 32 无偏置投影（256 个参数），在 ReLU 前相加：

```text
原20维 -> Linear(20,32) --+
                         + -> ReLU -> 原Linear(32,4) -> lexical_logits
新增8维 -> Linear(8,32) --+

final_logits = 原CLS + 原Token Alignment + lexical_logits
```

原分支形状和初始化顺序保持不变；新增初始化保存/恢复 RNG，不额外添加 dropout。
新增位置投影与原词汇分支同用 lr=1e-3，其余参数仍是 lr=2e-5。

两侧分别增加四个数，总共八维：

- 未匹配连续片段数量 / 当前侧词数；
- 最长未匹配片段长度 / 当前侧词数；
- 首个未匹配词的 1-based 位置 / 当前侧词数；
- 最后一个未匹配词的 1-based 位置 / 当前侧词数。

全匹配或当前侧为空时，该侧四个数均为 0。位置从 1 开始，使 0 专门表示没有差异。
第一版保留冠词、介词、否定和数字词，不用新规则删除它们；原 DeBERTa 输入不作任何修改。

## 对齐定义和适用范围

参考 [ALTN: Word Alignment Features for Cross-lingual Textual Entailment](https://aclanthology.org/S13-2023/) 中的“未对齐片段长度和位置”思路。
**不是原论文完整复现**：原论文使用跨语言统计词对齐，本版为英文短语采用固定的、相同词之间的最长公共子序列（LCS）一对一对齐，不训练对齐器，也不改变原神经 Token Alignment。

重复词只能一对一匹配，不能把对侧同一个词反复使用。LCS 平局按规范词序确定，保证交换 A/B 会精确交换两侧特征，不把 Text1 当作固定优先侧。

这里“未匹配”仅指该词没有进入所选词面 LCS，不是语义上没有对应词。它不能识别同义词，词序调整也可能导致未匹配，模型必须通过监督学习判断是否该采信；没有任何“差异长就判 F/B”的硬规则。
这是待验证的小改动，不保证同时减少 F/B->E 和 E->F/B。

## 保持的协议

- MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli，float32，全参数训练。
- 原 CLS、128维 Token Alignment、temperature=0.1、compare_layer 和 mean/max pooling。
- 官方原始 Train 3042，不加入 Qwen C 数据、人工增强、WordNet 或检索样本。
- seed42，10轮，max_length128；Row batch16，Pair batch16组，Dev batch16。
- Row CE 权重 `[1,1,1,1.5]`。
- 三组合 Real Pair CE 权重 `[1,1,1.25]`，smoothing=0.05，系数0.5。
- 每轮完整 Row CE 后完整 Pair CE，没有联合辅助损失或额外训练步数。
- 每轮只在 Pair 阶段结束后评价；以 Dev pair-aware W-F1 选最佳，同分保留更早轮次。
- 配对组仍只允许 FB/BF/EE，单例保留模型四分类；N precision=1 / SoftCons=1 是结构效果，不是本次位置特征学出来的。

特征只接收当前 text1/text2，不读标签、pair_id、Dev 错误、历史预测或拟合的全局统计。
历史预测仅在训练结束后导出 fixed/broken 对比，不用于选模。

## 输出与检查

模型：`model/deberta_altn/run01/altn_seed_42/best_model.pth`。
结果：本目录 `outputs/run01/altn_seed_42/`。

只保存一个最佳大权重文件；写入临时文件成功后再替换最佳，降低写盘失败损坏旧最佳的风险。每轮只保留小型评分 CSV。
`epoch_summary.csv` 除分数外，还记录 F/B->E、E->F/B、F<->B、N 四类错误数。
最终导出相对原 0.9248 基线、历史 Lexical 0.930569 的修好/破坏清单；历史 seed42 的比较不能代替配对多种子验证。

运行 `test_cpu.py` 可验证特征、重复词和交换方向、与原 Lexical 的初始化/零特征训练一致性、两个阶段反向传播、保存重载、原有官方评分及随机状态恢复。只使用随机微型 DeBERTa，不加载真实大权重、不启动 GPU 训练。
