# DeBERTa + 原词汇分支 + WordNet

以已经跑到 0.930569 的 `deberta_lexical/run02_branch_lr` 配方为基础，在独立目录增加 WordNet 特征。原目录、模型和结果不修改。

## 直接运行

右键运行本目录的 `train_deberta_wordnet.py`。

它会使用官方原始 Train 3042 行、Dev 660 行，重新从 NLI 预训练主干训练 10 轮。每轮分别输出独立预测与 pair-aware 成绩，只保存 pair-aware Weighted F1 最好的完整模型。

先检查数据而不训练，可以运行 `check_setup.py`。只需查看代码即可运行，不需要把旧模型参数手工复制进来。

本机英文 WordNet 已下载到项目根目录 `.nltk_data/corpora/wordnet.zip`，版本 3.0、大小 10,775,600 字节。NLTK 已安装。新机器若缺词典，训练前会尝试下载；也可以先运行 `prepare_wordnet.py`。缺 NLTK 时在所用解释器中安装 nltk。

先结束其他 GPU 训练，再启动本入口。此目录的自动检查仅使用 CPU；没有替用户执行真实 GPU 训练。

## 这次加了什么

保留原 20 维特征，再加 12 维 WordNet 特征，总计 32 维：

| 类别 | 内容 |
|---|---|
| 查询覆盖 | 两侧差异词能在所选 WordNet 词义中查询到的比例 |
| 差异范围 | 两侧不同表面词占各自查询候选词的比例 |
| 同义 | 两个方向的差异词同义覆盖率 |
| 上下位 | Text1 更具体→Text2 更一般，以及反方向 |
| 反义 | 两个方向的反义覆盖率 |
| 已知关系覆盖 | 两个方向存在上述任一种关系的词占比 |

先去掉两侧共有的表面词，再查看差异词。例如 `black car / black vehicle` 会主要比较 car 与 vehicle，避免共同的 black 掩盖差别。这里不是删除 DeBERTa 输入中的词，Transformer 仍然读取原始完整短语。

原 20 维特征代码从已验证版本复制，检查时会逐条核对它们没有变化。

## 分支结构和初始化

```text
原 20 维特征 --原 Linear(20,32)--+
                                +--> ReLU --> 原 Linear(32,4)
新增 12 维  --新 Linear(12,32)--+
                                          ↓
                    加到 CLS + Token Alignment logits 上
```

原词汇分支的 804 个参数仍按相同顺序和形状初始化；新增 WordNet 投影无偏置，384 个参数，初始为零。

这样最初的输出与旧模型相同，避免直接扩大输入矩阵后改变旧特征权重的初始化。新增层同样隔离初始化随机状态，不新增 dropout，后续由原 Row CE / Pair CE 更新参数。

这是一套共享主干的单模型，不是模型集成，不用 WordNet 规则强制修改标签。

## 保持原配方

- MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli，CLS + 原 Token Alignment。
- 原模型学习率 2e-5；原词汇分支和新增 WordNet 投影共同使用 1e-3。
- AdamW，weight_decay=0.01；不新增 scheduler 或辅助损失。
- 原始 Train 3042 行，不叠加 A–E 增强。
- seed42，float32，全参数训练，10 epochs，max_length128。
- Row batch16 行、Pair batch16 组、Dev batch16 行。
- Row CE 权重 F/B/E/N=[1,1,1,1.5]。
- Real Pair CE 权重 FB/BF/EE=[1,1,1.25]、label smoothing0.05、loss 乘0.5。
- 每轮先完整 Row CE，再完整 Pair CE；预测后使用原 pair-aware。
- 逐轮评估恢复随机状态，避免改变后续训练的 shuffle/dropout。

数据和模型各组参数在 `config.py`。再次运行请把 `RUN_NAME` 改为 run02，避免覆盖已有结果。

## 词义处理的边界

不额外下载 POS 标注器或语义模型。每种词性 n/v/a/r 最多使用 WordNet 返回的第一个词义。这能限制全部词义遍历引入的噪声，但**不是真正的上下文词义消歧**，仍可能选错。

上下位只沿 hypernym / instance_hypernym 向上搜索最多 4 层，不把共同祖先当成互相蕴含。例如 dog 和 cat 都属于 animal，不会因此被当成 dog 蕴含 cat。反义关系通过 lemma 的 antonyms 查询。同义词用所选 synset 的交集作为证据。

词典查询失败不会报成某个任务标签；特征同时保留查询覆盖信息，任务分类器继续学习最终输出。短语整体、数量、否定、语境、专名、多词表达等不能只靠词典查询解决。

这些是词级证据，不是短语蕴含金标。查询覆盖高也不等于词义或关系正确，不保证一定提升 Dev。

不读取 Dev 标签来提取词汇关系，不从 Train/Dev 拟合词典或标准化统计。审计表中的 label 仅用于人工检查，不传给模型特征函数。

## 检查结果与审计

`check_setup.py` 输出到 `prepared/`：

- wordnet_coverage.csv：词典查询覆盖、同义/上下位/反义关系触发数量。
- train_features.csv：32 个数值特征。
- train_wordnet_audit.csv、dev_wordnet_audit.csv：差异词、所选 synset、具体词对关系。

本次初步检查：Train 有 574/3042 行、Dev 有 123/660 行触发至少一种所选词义关系。这是覆盖统计，不是关系准确率，也没有据此修改标注。

## 输出位置

最佳完整模型：

```text
model/deberta_wordnet/run01/wordnet_seed_42/best_model.pth
```

本目录的结果：

```text
outputs/run01/wordnet_seed_42/
    run_config.json
    wordnet_coverage.csv
    train_features.csv
    dev_features.csv
    train_wordnet_audit.csv
    dev_wordnet_audit.csv
    epoch_summary.csv
    best_summary.csv
    epoch_数字/predictions_with_scores.csv
    comparison_with_original/   # 与原 0.924794 比较
    comparison_with_lexical/    # 与 0.930569 词汇分支版比较
```

两个比较目录均包含 comparison.csv、fixed.csv、broken.csv。训练后会自动输出修好与新增错误数。历史预测只参与事后分析，不进入新模型训练或解码。

最佳轮次记录在模型目录 best_metrics.json。运行 `evaluate_best.py` 可以按保存的配置和特征定义重新加载最佳权重，不训练。完整模型包含新增投影，不能用旧的 Adapter 加载器加载。

保存配置包含特征名单、源码校验和、WordNet 版本、NLTK 版本、词典文件校验和。重新预测时会核对，避免训练和推理使用不同特征。

## 可选对照

`config.py` 的 `USE_WORDNET_FEATURES=False` 会保留原 20 维词汇特征，把后 12 维置零，并保留相同模型结构。输出目录是 lexical_control_seed_42，不覆盖 WordNet 版。

也可以使用 `--no-wordnet` 参数。这个开关只关闭 WordNet 输入，不是关闭整个词汇分支。

当前先跑 WordNet 版即可。若想正式声称 WordNet 的收益，需补同协议对照和种子复验，不用继续从反复使用过的 Dev 上无限挑特征。

## CPU 验证

`test_cpu.py` 使用真实 WordNet 和随机初始化的一层微型 DeBERTa。检查关系方向、同义/反义、共同祖先、原20维一致、初始化和梯度一致、优化器分组、WordNet 投影实际更新、完整保存重载和官方评分。真实大权重和 GPU 训练没有在检查中运行，小模型成绩没有实验意义。

实现参考：[NLTK WordNet 接口](https://www.nltk.org/howto/wordnet.html)、[FBK-HLT-NLP 的词汇关系特征](https://aclanthology.org/S16-1121/)。本实现并非复刻 FBK 的完整系统。
