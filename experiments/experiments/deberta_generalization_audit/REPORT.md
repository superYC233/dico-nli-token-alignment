# DeBERTa 泛化、标签与语境审计

日期：2026-10-21。仅评估已有权重，没有新增训练，没有修改原数据、训练代码或权重。

## 1. 先说判断

当前问题是**明显的训练集拟合与泛化落差、局部表面形式敏感性、类别分数偏置，以及部分语境/标注口径问题共同存在**。不能概括成“模型完全没学会”，也不能把所有错误归为标注噪声。

下一步优先做经过审核的、从 Train 出发的最小对照诊断；暂不增加模型分支、Pair CE 或训练轮数。最明确的突破口是冠词/同义释义与修饰范围的泛化。任何增强都必须先核验保义，不能把冠词、介词、数量词替换一律视为不改变标签。

## 2. 本次查的模型

- `experiments/deberta_row_ce_frozen` 的 epoch 8。
- 冻结 embedding 和底部 6 层，训练顶部 6 层与原 Token Alignment/分类头。
- 只有四分类 Row CE，N 权重 1.5；没有训练期 Pair CE，没有词汇分支。
- 按 Dev **independent W-F1** 选模。
- 推理固定 batch 16、float32、长度上限 128。
- 重新加载最佳权重后，Dev 的 independent、pair-aware 预测与历史 epoch 8 都是 **0 条差异**。

| 指标 | Train | Dev |
|---|---:|---:|
| 独立正确数 | 3034/3042 | 559/660 |
| 独立准确率 | 99.74% | 84.70% |
| 独立 W-F1 | 0.997371 | 0.848510 |
| F 正确 | 878/881 | 167/190 |
| B 正确 | 880/881 | 172/190 |
| E 正确 | 799/802 | 147/174 |
| N 正确 | 477/478 | 73/106 |

这是**同一个模型**的实测差距。此前另一个联合损失模型的 Train 99.57% 没有被混用。

Dev 独立错误 101 行，来自 74 个组：非 N→N 42；N→F/B 20；N→E 13；E→F/B 8；F/B→E 12；F↔B 6。

pair-aware 后正确 587/660，W-F1 0.887296。改动 49 行，修好 32，破坏 4，改后仍错 13。剩余错误 73 行：N 33，E→F/B 18，F/B→E 16，F↔B 6。原始分类错误与结构解码后的错误必须分开讨论。

## 3. Train 答对，换形式后答错的实际证据

检索器只在 Train 上拟合，排除了完全相同和反向相同的文本对。找到 101 个错误行各自的同标签、Train 答对的最近类比，其中 29 行相似度至少 0.45，12 行共享一个完全相同的短语。这些是自然类比，不能全部当成受控单变量实验。

| Train：模型答对 | Dev：模型答错 | 观察 |
|---|---|---|
| `Woman / an adult female` → E | `A woman / An adult female` → B | 冠词/释义不稳 |
| `on the tracks / down the track` → N | `on the tracks / down the tracks` → E | 单复数近邻迁移失败；同时存在源关系歧义 |
| `on a crowded street / on a city street` → N | `on a crowded street / on a public street` → E | 属性替换后忽略区别 |
| `on the shore / along the beach` → N | `on the shore / on beside the beach` → E | 介词与不完整表达敏感 |
| `at the city train station / in a station` → F | `at the train station / at the station` → E | 限定关系没有稳定迁移 |
| `over grass / across the grass` → E | `through grass / across the grassy area` → N | 多项词汇、结构变化叠加 |

### 冠词例子另做了逐因素探针

使用同一权重，比较 `Woman/woman/A woman/a woman` 与 `an adult female/An adult female`，每种都测正反两向，共 16 行。

- 不加冠词：只改大小写，8/8 预测 E。
- 加 `A/a`：8/8 转成 B/F。
- 例如 `Woman / an adult female` 的 P(E)=0.9847；`A woman / an adult female` 为 0.3267；`A woman / An adult female` 为 0.0297。

这支持**这个例子中，冠词变化会触发错误的单向关系判断**。它不证明所有错误都由长度启发式导致。中间人工变体仍需保义审核，未计入正式 Dev 成绩，也未加入训练。

## 4. 标签冲突：已确认与待审核分开

### 确认的任务内输入冲突

Train 中相同有序输入 `South Africans / South Africa` 同时被标为 F 和 N。模型两条都预测 N，因此不可能同时答对。

经过精确文本、规范化文本、反向关系及实际 token 输入碰撞检查，发现的硬冲突仍是这一处。它至少造成 1 条不可避免的 Train 错误，**不足以解释 101 条 Dev 错误，也不能据此推导 Dev F1 的理论上限**。

若以后做清洁 Train 对照，应隔离相关两个 pair_id 的完整 3 行，不能只删 F 留下孤立的 B。本次没有删除或改标。

### 更完整的 PhrasIS 源关系检查

分开下载 headlines/images 四份公开源标注，其记录多重集合与本地 train/test all 文件完全一致，共 10214 条。

保留全部匹配，而非只取第一个匹配后发现：

- 14 个任务行有多个源关系，其中 Train 13、Dev 1。
- N 中 `to release / Releases`、`are standing / is standing`、Dev 的 `on the tracks / down the tracks` 都有 EQUI 与 SIMI 的源记录。
- `South Africans / South Africa` 有 FORW 与 SIMI 源记录。
- `Miss America crown / Miss America` 及反向，共 2 个 Train 行的任务方向与这份公开源记录的方向不一致。
- 518 个 Train 行和 114 个 Dev 行没有找到完全匹配的源文本对；主要来自 F/B/E，全部 N 都匹配到了。

**源记录不匹配不是标签错误证明。源关系冲突也不等于任务内部硬冲突**：它们可能来自不同原句语境、多义词或不同发布版本。应保留证据再核验。此前“全部 N 溯源无歧义”的表述，需要限定为在原先仅查 N 来源关系的范围内成立。

## 5. 依赖语境和标签口径的样本

官方说明短语来自新闻/图片描述的 iSTS 切块：[任务说明](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/)。源文件见 [公开 PhrasIS 仓库](https://github.com/Jonapa/assessing_logical_coherence)。

本次按领域、源文件、原句编号和 token 位置恢复可用语境。恢复结果含 `[GAP]`，不保证完整原句；没有臆造缺词。分领域后未发现 token 位置冲突，4 处原始跨度异常另列。

| 样本/任务金标 | 需要核验什么 |
|---|---|
| `as national security adviser / as his security adviser`：F | his 的指代、具体职位是否 national，需要新闻背景 |
| `Saudi Women / Saudis`：B | 常见集合包含方向更像 F；源句一侧是被允许参赛的 women，另一侧 Saudis 是允许者，疑似角色投射/方向问题 |
| `with a black top and a necklace / with a big necklace`：F | 第一段没有 big 的大小信息，不能按“多几个词”解释为严格蕴含 |
| `over police crackdown / over police violence`：E | 两词不是普遍等价，可能依赖同一新闻事件 |
| `to set out / to announce`：E | set out 的多义和省略宾语，需要上下文 |
| `Australian senator / Australian lawmaker`：N | 常见词义有包含关系，但源 SIMI→N；不能直接诊断为模型缺词义 |
| `may be trapped / trapped`：N，模型 F | 模型把“可能”推成“事实”，严格语义下也不成立；不能用口径问题替全部错误开脱 |

对全部 74 个错误组做了一次研究性初审：42 组标为需要语境审核，33 组标为需要标签口径审核。**这是候选数，不是已确认噪声率；未经第二标注者裁决，未修改金标。**

## 6. 其他发现

1. **不是输入被截断。** Train/Dev 最长分别 26/21 个模型 token，128 上限下截断数均为 0，无空文本。
2. **错误往往很自信。** Dev 平均最大概率 0.9644，但准确率 0.8470；101 个错误里 66 个置信度超过 0.9。错误置信度均值 0.8949、中位数 0.9712。训练拟合好不等于预测概率可靠。
3. **不是单纯没见过词。** Dev 有 365 行没有 Train 未见词，仍错 47 行；两段短语都曾在 Train 出现的 67 行里仍错 6 行。“Train 未见”也不代表预训练模型词表不认识该词。
4. **高重叠并非已证实的唯一瓶颈。** 当前 N 的低、中、高重叠切片正确率约 65.8%、70.3%、71.0%，没有“越重叠越差”的单调趋势；这些切片也不是因果对照。
5. **存在跨集重复。** 7 个精确有序文本键跨 Train/Dev 重合；规范化后 13 个键，涉及 15 个 Dev 行，当前都答对。剔除这 15 行的诊断 W-F1 为 0.844988，官方完整 Dev 仍为 0.848510。重复不是整个成绩差距的解释。
6. **同原句不同切块可跨集。** 246 个 Dev 行至少有一个匹配源语境在 Train 出现；其中 231 行只有一个匹配源语境。官方按“源短语对”分组，不承诺整篇新闻/原句隔离，不能直接叫违规泄露。
7. **更多轮次未解决稳定难点。** 51 个 Dev 行在 10 轮都错，448 行始终对，176 行至少改过一次预测。前者混合真实难例、语境问题及口径问题，不能自动全用于困难样本训练。
8. **原模型不是完全不懂 F/B/E。** 原 epoch7 虽有 307 条非 N→N，但排除 N 后单行 F/B/E 排序可答对 503/554；当前模型为 514/554。当前 pair-aware 也答对 514 行，但与单行排除 N 的具体答案有 12 行不同。
9. **原版 N 98/106 不能独立证明 N 语义边界更强。** 原版总共预测 N 405 次，当前 115 次。当前 N 的 ROC-AUC/AP 为 0.9237/0.7385，原版为 0.8720/0.6851。这是固定 Dev 的描述性排序比较，不是跨种子结论。

## 7. 接下来做什么

**第一步：先裁决，不训练。** 从本次初审表中选取最有代表性的语境/口径案例，区分：模型可合理判断的错误、需要原句的信息不足、标签政策或源方向争议。优先核验 `Saudi Women/Saudis`、N 的上下位样本、相同短语对的 EQUI/SIMI 两种源记录。不要批量改 Dev 标签。

**第二步：固定 Train-only 最小对照诊断。** 从高置信 Train 样本出发，覆盖冠词与同义释义、真正增加限制的修饰词、并列修饰范围、模态与核心对象差异。每次只改一个因素，人工确认标签；保留原句和新句的判定依据。不要照搬本次 Dev 错例去补训练，更不要默认任意冠词或介词替换都保义。

**第三步：只做一个针对性干预。** 若诊断确认同一模式反复失败，再考虑加入审核过的 Train 最小对照，保持当前冻结范围、Row CE、学习率和选模规则不动；用等量复制/随机样本作预算对照。先看独立预测及现象切片，再附 pair-aware 成绩。

暂不推荐：再加 Pair CE、再加知识分支、继续堆轮次或把所有高重叠 N 一律加权。当前证据尚不能保证这些措施对症。

评估边界：单种子，Dev 已反复使用；自然近邻不等于受控反事实；人工初审不等于重标金标。最终泛化仍需独立评估集。

## 8. 主要文件

所有明细在 `outputs/row_ce_frozen_epoch8/`：

- `reviewed_error_groups.csv`：74 个错误组的初审、正反向答案、源语境。
- `train_correct_dev_wrong_analogues.csv`：Train 对/Dev 错的 101 行自然类比。
- `woman_surface_probes.csv`：16 行冠词/大小写逐因素探针。
- `label_conflicts.csv`：不同检查方法下的冲突明细，同一冲突会出现多次，不能重复计数。
- `source_matches_with_context.csv`、`source_mapping.csv`：全部源匹配与歧义。
- `train_errors.csv`、`dev_independent_errors.csv`、`dev_pair_aware_errors.csv`：错误原始分数。
- `summary.json`、`validation.json`：量化汇总和官方评分验证。

复现脚本：`export_predictions.py` → `prepare_sources.py` → `analyze.py` → `review_errors.py` → `probe_surface.py` → `validate_audit.py`。均不训练模型；权重仍从原位置只读加载。
