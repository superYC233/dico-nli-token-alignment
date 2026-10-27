# Train理由与依据词辅助监督

右键运行本目录 `train_deberta_reason.py` 即可训练。配置集中在 `config_reason.py`；以后改 `RUN_NAME` 复用同一套代码。

## 本次具体做什么

- 模型：原DeBERTa NLI初始化 + Token Alignment + 20维Lexical；全参数训练。
- 数据：官方Train **3042行**；沿用其原文、标签和分组。
- 现已为**1760个Train组、全部3042行**补齐reason：保留原100组说明，另写1660组说明；只对原本已有的反向行互换依据位置。
- 全部3042行照常进行关系分类训练。`supported`的1756行参与依据监督；`needs_context`的1103行和`label_question`的183行跳过依据辅助损失，保留原金标与Row/Pair训练。没有新增反向N、人工短语或外部数据。
- 按组打乱混合原真实Pair和N单例；每轮220步，每条原行一次，每个真实Pair一次。
- 按**independent F1**保存最佳，同时保存同一轮pair-aware成绩。权重仅在刷新最佳时覆盖`best_model.pth`。

这次以原mixed版作底版，目标是检验额外依据监督能否帮助单行泛化。与0.938的两阶段、3290行增强配方存在训练安排和数据差异，不能将两者的分差全部归因于理由监督。

## CSV在哪

- `reason_annotations.csv`：1760组理由、状态和依据片段，片段用`|`分隔；训练读取这个源文件。
- `reason_annotations_pilot100.csv`：保留最初100组版本。
- `reason_semantic_review.tsv`：1660组新增说明的可修改来源，包含可支持、需要语境、标签待核验三种判断。
- `complete_train_reasons.py`：合并旧100组与新增1660组；只读Train，并检查Train内部重复输入的标签冲突。
- `outputs/reason_prepared_train_only/train_with_reasons.csv`：已生成的完整3042行训练表。
- `outputs/reason_prepared_train_only/reason_examples.csv`：已生成的3042行完整理由表，含原文、金标、reason、变化类型、状态和字符位置。
- `outputs/reason_prepared_train_only/reason_review_needed.csv`：1103行需语境说明和183行标签疑点，供人工复核。
- `outputs/reason_prepared_train_only/evidence_token_audit.csv`：依据片段对应的实际分词。
- `outputs/reason_prepared_train_only/manifest.json`：来源、覆盖情况与文件哈希。

标注依据是本次读取的Train原文及Train标签，没有读取Dev进行补齐。它们属于**助手撰写、尚未经过独立人工复核的新增标注**，不应在论文中写成官方理由或已复核的人类标注。`label_question`是待核验的疑点，不是已证实的官方标注错误；`needs_context`也不代表该行必然无法学习。

原100组的依据片段由助手选取；新增组的片段通过原文词面差异定位，另以`evidence_source=surface_difference_draft`标明。文字理由是逐组说明，片段定位是可复现的辅助草稿，未宣称这些片段已经通过人工语义审核。

这些是原Train的依据标注，因此继承原数据已经存在的Train/Dev文本重合；本次没有宣称官方数据之间绝对无重合，也没有新增文本泄漏检查的保证。

## reason怎么被训练使用

原文仍作为唯一文本输入：

```text
Text1、Text2 → 同一次DeBERTa编码
                    ├→ 原CLS + Alignment + Lexical → F/B/E/N
                    └→ 新1538参数线性头 → 每个token是否为所标依据
```

损失：

```python
loss = row_loss + 0.5 * pair_loss + 0.1 * evidence_loss
```

CSV中`evidence_spans1/2`的字符位置通过fast tokenizer映射到token；特殊token、padding及非supported行忽略。每个可用样本的token损失先取均值，再对本批可用样本平均。已有词汇特征始终只从原Text1、Text2提取。

**真正的辅助训练目标是依据词位置。** reason文字与operation保留作语义说明和审核，这一版没有对整段英文解释进行文本生成或语义蒸馏。未选中的词仍完整提供给分类器；所选片段是稀疏判别线索，不宣称它们构成完整逻辑证明。

辅助梯度流入同一个DeBERTa主干；这能提供额外监督，但不保证分类器会因果性地使用这些线索，也不自动消除所有Pair目标问题。Dev F1是否改善需要实际训练检验。

依据标注监督的背景可参考[ERASER，ACL 2020](https://aclanthology.org/2020.acl-main.408/)。这里是本项目的简单辅助任务，并非该论文系统的复现。

## 怎么运行

默认训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_mixed_pair/train_deberta_reason.py
```

全量版本默认RUN_NAME为`run09_reason_all_seed42`。模型保存在`D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run09_reason_all_seed42/`，指标和逐条预测在本目录`outputs/run09_reason_all_seed42/`。旧run08对应最初100组，不覆盖其结果。

同代码零依据权重对照，其他设置、样本和批次保持一致：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_mixed_pair/train_deberta_reason.py --baseline
```

它使用`run09_reason_all_seed42_baseline`，不会覆盖理由版。比较两版的`best_summary.csv`，每张表两项指标均来自同一轮；不要拼接不同轮次的独立/结构最高分。

已生成全部CSV。修改新增说明后，先运行`complete_train_reasons.py`合并源标注，再运行`train_deberta_reason.py --prepare-only`更新完整CSV和token审计。

训练完成后重新预测Dev：`train_deberta_reason.py --evaluate`。此路径只读取最佳权重、tokenizer及Dev原文，完全不读取Train或reason文件。

## 已完成检查

已执行全量prepare-only和`check_reason_setup.py`，未训练。通过：原3042行及标签不变、全部理由非空、可用依据都存在且未被截断、待审核行辅助目标忽略、原始分词不变、特殊位置忽略、每轮行/Pair覆盖正确、辅助梯度到达编码器、零权重分类梯度与原模型一致、无标注时损失为0、同一次编码提供分类与依据输出、小模型保存读取一致。

另外已加载本机真实NLI主干做CPU前向检查：分类输出`[2,4]`、依据输出`[2,32,2]`，关闭依据输出时分类logits完全相同。没有创建优化器或做参数更新；这不是准确率测试。
