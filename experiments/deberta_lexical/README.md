# 原 DeBERTa + 轻量词汇特征分支

本目录从[原0.9248版](../deberta_original/README.md)改写；原版训练与评估已归入自己的目录，模型和历史预测单独保留。

## 开始运行

1. 可选：右键运行 `check_setup.py`。只检查 3042 条 Train、660 条 Dev、1282 组 Pair 和特征，不加载真实预训练模型。
2. 正式训练：右键运行 `train_deberta_lexical.py`。
3. 训练会逐轮预测并输出独立预测、pair-aware 的成绩，只保存 pair-aware Weighted F1 最好的完整模型。

本次默认只训练词汇分支版本，不自动训练额外对照或 A–E。训练前请结束其他占用 GPU 的训练进程。

参数在 `config.py`。初次可直接用默认值；后续新运行请将 RUN_NAME 改为 run02，避免覆盖旧结果。

当前默认 `RUN_NAME='run02_branch_lr'`：原主干、CLS 和 Alignment 保持学习率 `2e-5`，新增词汇分支单独使用 `1e-3`。这是在 run01 基础上只调整分支学习率的实验；run01 的权重、配置和结果保留。启动时会打印两组实际学习率，保存的 run_config.json 也分别记录。

可选的严格对照：把 `USE_LEXICAL_FEATURES=False`，再次运行训练入口，就会关闭分支、从相同预训练主干重跑 baseline。baseline 和 lexical 使用不同输出目录。也可通过 `--baseline` 参数关闭分支。

## 具体增加了什么

```text
原有 CLS 四分类分数
        +
原有 Token Alignment 四分类分数
        +
20 维词汇特征 → Linear(20,32) → ReLU → Linear(32,4)
        ↓
最终 F/B/E/N 四分类 logits
```

新增 804 个参数；最后一层权重和偏置初始化为 0，因此初始输出与原版一致。新增层的初始化隔离随机状态，不额外添加 dropout。分支随后通过原 Row CE 和 Real Pair CE 的梯度学习，没有额外辅助损失。

这是加入特征的单模型，不是多个模型集成，也不是用规则覆盖最终标签。

## 第一版的 20 个特征

- 保留全部词时的 Text1→Text2 覆盖率、反方向覆盖率、Jaccard。
- 忽略冠词 a/an/the 后的双向覆盖率、各侧独有词数量、双向包含标志、集合相等标志。
- 两侧词数、带方向的长度差。
- 两侧数字是否出现、数字集合重合度、两边都有数字时集合是否不同。
- 两侧否定是否出现、否定出现情况是否不同。

“去冠词覆盖率”只是辅助数值特征的名字，不是句法分析所得的内容词，也没有词形还原。不会修改送给 DeBERTa 的原始短语。

数字特征支持 zero 到 twenty 的简单数字词，以及普通阿拉伯数字；不是完整数量语义解析器。WordNet、词性、依存句法、词向量、训练集检索均未加入，本版不需要下载这些资源。

词汇覆盖率不是蕴含概率。不同词可能同义、相同词也可能有不同关系；分支输出由主任务监督学习，不按覆盖率或词数直接指定 E/F/B/N。可能只学到词面偏置，因此需要与对照比较，不能预先保证提升。

## 保持不变的内容

- 主干：MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli。
- 原有 CLS 分支、128 维 Token Alignment、温度 0.1、ReLU、初始化方式。
- 官方原始 Train 3042 条，不加入 C 或其他增强数据。
- float32、全参数训练、seed42、10 轮、max_length128。
- AdamW，原模型 lr2e-5，weight_decay0.01；run01 全部参数同学习率，当前 run02_branch_lr 的新增词汇分支 lr1e-3。
- Row batch16 行，Pair batch16 组，Dev batch16 行。
- Row CE 权重 F/B/E/N=[1,1,1,1.5]。
- 三组合 Pair CE 权重 FB/BF/EE=[1,1,1.25]，label smoothing0.05，loss 乘0.5。
- 每轮先完整 Row CE，再完整 Pair CE；两阶段各自更新参数。
- 同样的 pair-aware：完整组限制 FB/BF/EE，单例保留模型四分类预测。

正反两条数据各自使用对应方向的词汇特征。所有特征仅依赖当前 text1/text2，不读取标签、组大小、历史模型预测，也不拟合 Dev 统计。

新增逐轮评估时保存/恢复随机状态，避免评估改变下一轮 shuffle/dropout。只保留当前最佳的大权重，小型预测 CSV 每轮保留。历史0.9248是参照值，重跑 baseline 不保证恰好重现该数字。

## 输出

模型保存到项目根目录：

```text
model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth
```

里面是完整的主干 + CLS + Alignment + 词汇分支参数，不能拿 Qwen 的 Adapter 加载器加载。

本目录的结果：

```text
outputs/run02_branch_lr/lexical_seed_42/
    run_config.json
    train_features.csv
    dev_features.csv
    epoch_summary.csv
    best_summary.csv
    epoch_数字/
        predictions_with_scores.csv
        independent/  # 官方预测、分数和混淆矩阵
        pair_aware/
    comparison_with_original/
        comparison.csv
        fixed.csv
        broken.csv
```

`best_metrics.json` 在模型目录，记录最佳轮次。`best_summary.csv` 是该轮的两种解码成绩，不把不同轮次拼到一行。

训练后自动和原0.9248那份历史 pair-aware 预测比较，列出修好的样本和新增错误。历史预测只用于事后分析，不进入训练或新模型预测。

可选运行 `evaluate_best.py`，它会加载本组最佳完整权重重新预测；参数和特征名单从保存的配置读取，特征源码/Dev 哈希不一致时会报错。

## 验证与评价

`test_cpu.py` 使用随机一层微型 DeBERTa，检查原版初始化/输出/梯度一致性、特征方向、Row/Pair 反向、词汇分支能更新、随机状态恢复、保存加载和官方评分。不会加载真实大权重或训练 GPU，小模型分数没有实验意义。

主要看 pair-aware Weighted F1，以及 E/F/B/N 各类正确数。当前基线50条错误中36条为 E↔F/B，期望检查这里是否改善，并观察是否损害 N。SoftCons=1、N precision=1 在当前数据结构和解码规则下是结构保证，不是新增分支学会了完美推理。

结果仍是经过多次开发的 Dev、单种子的探索。比历史分数略高不自动等于稳定提升；后续需要同协议对照与种子复验。
