# 项目内其他模型权重清理候选

核查日期：2026-10-20。仅扫描 G:/pycode/SemEval/Task-2-DiCo-NLI-main。

没有删除或移动任何文件。本清单列出当前“DeBERTa 词汇分支 + 阶段诊断”不使用的旧实验权重。保留 CSV、JSON、代码与日志。若还需要直接加载旧模型复验，可先把对应权重转存到有空间的磁盘。

## 优先保留

- [当前最佳词汇分支 0.930569](../model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth)
- [原正式基线 0.924794](../model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth)
- 当前阶段诊断的代码和 CSV。失败保存的 run01/best_model.pth 只有 53248 字节，不是可用模型。

## 上一轮清单之外的候选

空间为 GiB，按实际 .pth 文件大小汇总；不含预测表等小文件。历史旧实验分数来自项目实验记录，两个词汇分支运行的分数另核对了各自 best_metrics.json。

| 权重所在目录 | 文件数 | 权重空间 | 说明 |
|---|---:|---:|---|
| [deberta_lexical/run01/lexical_seed_42](../model/deberta_lexical/run01/lexical_seed_42) | 1 | 0.686 | 词汇分支第一版，Dev W-F1=0.921383，已被 run02 替代。 |
| [deberta_lexical/run03_branch_lr5e4/lexical_seed_42](../model/deberta_lexical/run03_branch_lr5e4/lexical_seed_42) | 1 | 0.686 | 词汇分支较低学习率试验，Dev W-F1=0.926365，未超过 run02。 |
| [deberta_nli_base_token_alignment_official_dev_aug24_seed_42](../model/deberta_nli_base_token_alignment_official_dev_aug24_seed_42) | 3 | 2.057 | 早期 aug24 增强试验，当前完整官方 Train 主线未使用。 |
| [deberta_nli_base_prototype_k1_seed_42](../model/deberta_nli_base_prototype_k1_seed_42) | 1 | 0.685 | 旧原型分类头 k=1，历史官方 Dev W-F1=0.908659。 |
| [deberta_nli_base_prototype_k3_seed_42](../model/deberta_nli_base_prototype_k3_seed_42) | 1 | 0.685 | 旧原型分类头 k=3，历史官方 Dev W-F1=0.898678。 |
| [deberta_nli_base_token_alignment_attn_pool_seed_42](../model/deberta_nli_base_token_alignment_attn_pool_seed_42) | 1 | 0.686 | 旧 attention pooling 试验，历史官方 Dev W-F1=0.919820。 |
| [deberta_nli_base_token_alignment_compose_seed_42](../model/deberta_nli_base_token_alignment_compose_seed_42) | 1 | 0.686 | 旧 compose 分支，历史官方 Dev W-F1=0.918678。 |
| [deberta_nli_base_token_alignment_group_lr_seed_42](../model/deberta_nli_base_token_alignment_group_lr_seed_42) | 1 | 0.686 | 旧分组学习率试验，历史官方 Dev W-F1=0.919161。 |
| [deberta_nli_base_hierarchical_manual_seed_42](../model/deberta_nli_base_hierarchical_manual_seed_42) | 1 | 0.685 | 旧层级分类试验，历史官方 Dev W-F1=0.883999。 |
| [deberta_real_pair_ce_direction_interleaved_seed_42](../model/deberta_real_pair_ce_direction_interleaved_seed_42) | 1 | 0.685 | 旧交替训练试验，历史官方 Dev W-F1=0.883829。 |
| [deberta_real_pair_ce_weights_scheduler_seed_42](../model/deberta_real_pair_ce_weights_scheduler_seed_42) | 1 | 0.685 | 旧学习率调度试验，历史官方 Dev W-F1=0.883898。 |
| [deberta_real_pair_ce_weights_type_aux_w0p3_seed_42](../model/deberta_real_pair_ce_weights_type_aux_w0p3_seed_42) | 1 | 0.685 | 旧类型辅助损失试验，历史官方 Dev W-F1=0.892869。 |
| [deberta_real_pair_ce_weights_nli_base_lr5e6_ep15_seed_42](../model/deberta_real_pair_ce_weights_nli_base_lr5e6_ep15_seed_42) | 1 | 0.685 | 旧小学习率长轮数试验，历史官方 Dev W-F1=0.889658。 |
| [deberta_real_pair_ce_weights_nli_base_n1p75_seed_42](../model/deberta_real_pair_ce_weights_nli_base_n1p75_seed_42) | 1 | 0.685 | 旧 N 权重 1.75 试验，历史官方 Dev W-F1=0.890178。 |

以上 14 组共可释放 **10.96 GiB**。只删除对应目录内的 .pth 文件即可。

## 已核实的重复权重

- 可清理副本：[deberta_nli_base_token_alignment_seed_42/epoch_7.pth](../model/deberta_nli_base_token_alignment_seed_42/epoch_7.pth)。
- 保留主副本：[official_full_control_seed_42/epoch_7.pth](../model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth)。
- 本次逐文件计算 SHA-256，两个文件内容完全相同。删除副本可再释放约 **0.686 GiB**，但使用旧副本路径的历史脚本需要改为保留的主副本路径。

上述候选加该重复副本共约 **11.65 GiB**。

## 本次未列为优先删除

NULL-M、Token Alignment seed=7、contrast 等历史对照权重先保留，避免影响多种子和方法比较。它们是否长期保留由后续复验需求决定；不能仅凭分数未达当前最佳就认定全部无用。
