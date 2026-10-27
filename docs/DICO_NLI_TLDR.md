# DiCo-NLI 研究记录：一页摘要

## 最终方案已经确定

**DeBERTa-v3-base NLI + Token Alignment + Real Pair CE + Pair-aware inference。**

任务是对英语短语对区分 F（正向蕴含）、B（反向蕴含）、E（等价）、N（其他）。最后用官方完整训练集 **3042行**，评估官方 **DEV 660行**，不加入增强样本。

| 最终指标 | 数值 |
|---|---:|
| Weighted F1 | **0.9247935854** |
| Accuracy | 0.9242424242 |
| SoftCons / HardCons | 1.0000 / 0.9241877256 |
| 最佳epoch / seed | **7 / 42** |

## 最终做了什么

NLI编码器同时提供CLS全局分类和token比较特征。后者先投影到128维，做双向软对齐，再比较原表示、对齐表示、差和乘积，mean/max池化后输出4个logits，与CLS logits相加。每轮先训练行级四分类，再按真实 `pair_id` 训练FB/BF/EE组合。推理时对完整pair联合选择；singleton仍由模型四分类，**没有“无pair就判N”的规则**。

## 中间尝试了什么，为什么没有继续

尝试过Large/ModernBERT、原型与分层头、NULL、Conv、attention pooling、辅助任务、Boundary、双向二元蕴含、不同训练调度和数据增强。很多变化在减少F/B→E时增加了E→F/B。纯二元蕴含保存F1=.8520，Entailment辅助=.8837，均未采用；后者当前实现还取消了Pair训练，不能当作严格单变量失败证据。

NULL-M曾有单次.9290，manual曾有.9335；前者有限seed未建立稳定优势，后者存在历史DEV介入来源问题。**最终选定.9248不等于历史所有数字的最大值。** 原token与最近关闭增强的控制运行预测完全相同；当前24对train-only增强反而降到.9148。

## 还剩什么问题

50条错误中：F/B→E **20**、E→F/B **16**、F↔B **6**、N错误 **8**。72%集中于等价和单向蕴含边界。需要判断“差异是否改变信息包含关系”，而不只是发现差异。SoftCons=1由硬解码保证，不代表语义全对。

## 下次回来先看哪里

- [完整研究/实验记录](history/DICO_NLI_FINAL_EXPERIMENT_RECORD.md)：方法、结果、失败分支、代码配置和产物索引。
- [最近复现脚本](../experiments/deberta_original/train.py)：`USE_AUGMENTATION=False`，compare为ReLU；原名token脚本当前已经是GELU，勿混用配置。
- [最终权重](../model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth) / [最终预测](../model/deberta_nli_base_token_alignment_official_full_control_seed_42/dev_predictions_pair_aware.csv)。

`clean`是官方train内部重新划分的2430/612，**不是官方DEV**。最终保留官方原始数据，其train/DEV存在少量规范化文本对重合，完整记录已披露。

若未来继续，先固定这套配置，在多个seed及独立评估上确认效果；语义研究优先考虑仅由训练数据构造、经核验的“等价改写 vs 单向限定”对照监督。当前不再堆结构或继续扫描系数。
