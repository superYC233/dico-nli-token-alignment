# 实验目录索引

按用途查找入口；每个实验自己的代码和outputs放在同一目录。部分方法有交叉，这里按主要用途归类。

当前优先查看：[五个融合方案](ensemble_selected/README.md)、[原Lexical版](deberta_lexical/README.md)、[数据增强](deberta_manual_aug/README.md)。完整发展过程和成绩见[项目README](../README.md)。

`generative_nli.py`是多个生成式实验共用的工具模块，`__init__.py`用于包导入。

## 基线与不同编码器

| 目录 | 内容 |
|---|---|
| [deberta_original](deberta_original/README.md) | 原0.9248版，独立的训练与评估入口 |
| [baselines](baselines) | 早期BERT、DeBERTa、Large和ModernBERT基线 |
| [deberta_base_lexical](deberta_base_lexical/README.md) | 未经NLI微调的DeBERTa-base＋Lexical |
| [deberta_nli_large](deberta_nli_large/README.md) | NLI初始化的DeBERTa-large |
| [deberta_nli_xsmall](deberta_nli_xsmall/README.md) | 较小NLI编码器 |
| [deberta_zeroshot](deberta_zeroshot) | 另一种NLI初始化checkpoint |
| [modernbert_nli_base](modernbert_nli_base/README.md) | ModernBERT的NLI初始化对照 |

## 对齐与词汇特征

| 目录 | 内容 |
|---|---|
| [alignment](alignment) | 原Token Alignment、NULL和均匀对齐等历史版本 |
| [auxiliary](auxiliary) | 方向、类型、分层读出等早期辅助任务 |
| [deberta_lexical](deberta_lexical/README.md) | 原Alignment模型增加20维Lexical分支 |
| [deberta_altn](deberta_altn/README.md) | 差异位置和连续差异片段特征 |
| [deberta_wordnet](deberta_wordnet/README.md) | WordNet关系统计特征 |
| [deberta_kim](deberta_kim/README.md) | 对齐时注入局部词对关系 |
| [deberta_kim_gate](deberta_kim_gate/README.md) | 为局部知识增加上下文门控 |
| [deberta_lexical_composition](deberta_lexical_composition/README.md) | 组合和局部比较特征实验 |

## 数据增强

| 目录 | 内容 |
|---|---|
| [augmentation](augmentation) | 早期人工边界及Train-only增强 |
| [deberta_abcde](deberta_abcde/README.md) | 原样、复制和反向增强五臂对照 |
| [deberta_original_aug](deberta_original_aug/README.md) | 原Alignment模型＋scope200 |
| [deberta_lexical_aug](deberta_lexical_aug/README.md) | Lexical模型＋scope或relation增强 |
| [deberta_manual_aug](deberta_manual_aug/README.md) | relation48、局部对照、语义对照及相关训练 |
| [deberta_family_aug](deberta_family_aug/README.md) | 大批家族式对照增强 |
| [deberta_closure](deberta_closure) | 闭包／E改写88和归档的旧生成数据 |
| [qwen_scope_aug](qwen_scope_aug/README.md) | Qwen加入scope增强 |

## 阶段诊断与泛化分析

| 目录 | 内容 |
|---|---|
| [deberta_lexical_stage_probe](deberta_lexical_stage_probe/README.md) | 分别检查Row后与Pair后的词汇版预测 |
| [deberta_original_aug_stage_probe](deberta_original_aug_stage_probe/README.md) | 增强原版的阶段诊断 |
| [deberta_generalization_audit](deberta_generalization_audit/README.md) | Train／Dev泛化和文本重合分析 |
| [deberta_minimal_contrasts](deberta_minimal_contrasts/README.md) | 最小差异对照探针 |
| [deberta_lexical_reliability](deberta_lexical_reliability/README.md) | 词汇版跨seed稳定性验证 |

## 训练目标、冻结与优化

| 目录 | 内容 |
|---|---|
| [pair_training](pair_training) | 早期Pair CE、Real Pair CE、一致性和权重对照 |
| [deberta_lexical_mixed_pair](deberta_lexical_mixed_pair/README.md) | 混合Row／Pair、反向KL、依据监督和回放 |
| [deberta_lexical_row_pair](deberta_lexical_row_pair/README.md) | Pair批次同时做四类Row监督 |
| [deberta_pair_hinge](deberta_pair_hinge/README.md) | 含N负组合的结构hinge |
| [deberta_reversal_kl](deberta_reversal_kl/README.md) | 反向四类一致性训练 |
| [deberta_row_anchor](deberta_row_anchor/README.md) | 限制Pair训练偏离Row预测 |
| [deberta_row_ce_frozen](deberta_row_ce_frozen/README.md) | 冻结部分编码器，仅做Row CE |
| [deberta_frozen_n_head](deberta_frozen_n_head/README.md) | 冻结底座后训练N判别头 |
| [deberta_stage_average](deberta_stage_average/README.md) | 固定轮次窗口的权重平均 |
| [deberta_lexical_child_tuning](deberta_lexical_child_tuning/README.md) | 限制部分梯度更新的正则实验 |

## 表示融合与联合分类结构

| 目录 | 内容 |
|---|---|
| [deberta_feature_fusion](deberta_feature_fusion/README.md) | CLS、Alignment、Lexical消融和特征拼接 |
| [deberta_lexical_joint](deberta_lexical_joint/README.md) | 联合分类结构 |
| [deberta_lexical_pooler](deberta_lexical_pooler/README.md) | 复用NLI pooler |
| [deberta_joint_pair](deberta_joint_pair/README.md) | 联合Pair头 |
| [deberta_joint_pair_frozen](deberta_joint_pair_frozen/README.md) | 冻结底座的联合Pair头 |
| [deberta_bidirectional](deberta_bidirectional/README.md) | 双向编码、联合分类及PAWS／SiFT等实验 |
| [deberta_local_semantics](deberta_local_semantics/README.md) | 局部语义和外部辅助信号 |

## Qwen与其他生成式模型

| 目录 | 内容 |
|---|---|
| [qwen_lora](qwen_lora/README.md) | Qwen3-0.6B主线、理由、六类、检索和词级比较 |
| [qwen2p5_3b](qwen2p5_3b) | Qwen2.5-3B和早期Real Pair CE脚本 |
| [qwen25_3b_reason](qwen25_3b_reason/README.md) | Qwen2.5-3B的C＋reason |
| [qwen3_1p7b](qwen3_1p7b/README.md) | Qwen3-1.7B训练与预测 |
| [qwen3_1p7b_lexical](qwen3_1p7b_lexical/README.md) | Qwen3-1.7B加入词汇分支 |
| [qwen3_4b_instruct_2507](qwen3_4b_instruct_2507/README.md) | Qwen3-4B-Instruct-2507 |
| [smollm2_1p7b](smollm2_1p7b/README.md) | SmolLM2-1.7B对照 |
| [qwen_mnli_transfer](qwen_mnli_transfer/README.md) | MNLI监督迁移后再做DiCo训练 |

## 多模型预测

| 目录 | 内容 |
|---|---|
| [ensemble_selected](ensemble_selected/README.md) | S01、S11、S12、S14、S16完整训练和预测入口 |
| [model_ensemble](model_ensemble/README.md) | 已有成员分数搜索、融合计算和详细报告 |
