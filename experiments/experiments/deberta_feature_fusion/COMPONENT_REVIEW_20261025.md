# 四个组件、融合版结果与下一步判断

核对日期：2026-10-25。只读取代码、日志和已有预测，运行了小型 CPU 数学核验；没有训练模型，没有改动训练配置。

**主要结论**

“E 与 F/B 的边界需要更好的语义监督”是合理的研究方向；但“四个组件都只管一致性”“主干已被证明达到语义上限”“只有加数据才可能有效”都超出了目前证据。

需要区分两个问题：

- 单行预测在 Pair 阶段退化：最直接的设计问题是用同一个四分类输出承受三组合条件分类目标，并连续执行一整段 Pair-only 更新。
- 结构解码后的残余错误：主要是 E 与 F/B 的边界。现有对齐/比较、任务数据覆盖、片段语境缺失都可能影响它，尚不能定位成一个已被证明的唯一原因。

**1. 最新融合版的最终结果，不是第 2 轮的中间结果**

数据：官方 Train 3042 行、Dev 660 行；没有使用 3290 行增强配方。

| 模型 | 最佳轮 | independent W-F1 | pair-aware W-F1 | PA F/B/E 正确 | PA N 正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 Lexical | 3 | 0.468191 | 0.930569 | 516/554 | 98/106 |
| 对齐后增加 CNN | 8 | 0.725116 | 0.903072 | 506/554 | 90/106 |
| 本次特征融合 | 10 | 0.507145 | 0.916871 | 508/554 | 97/106 |

原 0.938 增强配方和这三版不能当作同一数据设置：相关原配方使用 3290 行，而且不是本表的 Lexical 配方。

最新结果来自：

- `outputs/run01_feature_fusion/feature_fusion_seed_42/best_summary.csv`
- `outputs/run01_feature_fusion/feature_fusion_seed_42/epoch_10/`
- `outputs/run01_feature_fusion/feature_fusion_seed_42/comparison_with_lexical/`

相对原 Lexical，最新融合版修好 13 行、破坏 22 行，净少对 9 行。按金标拆分：

| 类别 | 修好 | 破坏 | 净变化 |
| --- | ---: | ---: | ---: |
| F | 3 | 4 | -1 |
| B | 3 | 4 | -1 |
| E | 6 | 12 | -6 |
| N | 1 | 2 | -1 |

主要损失在 E，但这不等于已经确定某一种 E 机制。单个 seed、各自按 Dev 取峰值的比较，也不能证明所有融合结构必然无效。

**2. 同一轮中，Pair 阶段究竟改变了什么**

最新融合版第 10 轮：

| 指标 | Row 后 | Pair 后 |
| --- | ---: | ---: |
| independent W-F1 | 0.833426 | 0.507145 |
| pair-aware W-F1 | 0.875556 | 0.916871 |
| 独立 F/B/E 正确 | 466 | 236 |
| 非 N 被独立判为 N | 58 | 307 |
| N 正确 | 81 | 97 |
| PA F/B/E 正确 | 498 | 508 |

Pair 阶段确实把组合判断做得更好，PA F/B/E 多对 10 行；同时单行判别显著退化。这两件事同时成立，不能把结构增益全说成“模型什么也没学会”，也不能用结构增益证明四分类已经学好。

十轮中，Pair 后 independent 比 Row 后差的有 8 轮，pair-aware 变差的有 7 轮。Pair 阶段并非总会改善结构分，也并非必然把输出推向 N。

本版 Row 后最高 independent 为第 3 轮 0.846285；Pair 后最高 independent 为第 6 轮 0.847296。不能把这两个状态和第 10 轮最高结构分拼成一个模型的能力。

**3. 四个组件真正负责什么**

| 组件 | 当前实现的职责 | 主要局限 | 当前证据 |
| --- | --- | --- | --- |
| Pair CE | 用金标 FB/BF/EE 教组合分类 | 不约束四分类的全部相对分数；Pair-only 阶段缺少同步 Row 监督 | 第 10 轮单行与结构表现分离，且有精确不变性 |
| Token Alignment | 对上下文化 token 做软匹配，再学习局部比较并汇总 | 无 NULL，缺少显式短语/作用域关系；局部匹配缺少直接标注 | 有小幅读出贡献，但 CNN/融合没有超过原版 |
| Lexical | 根据覆盖率、长度、数字、否定等特征学习四类修正 | 不直接提供词义；可能学到启发式，也可能补充有用信号 | 去掉它略降，原无 Lexical 模型也塌陷 |
| pair-aware | 推理时在三种允许组合中选最高分 | 隐藏单行 N/非 N 和 F/B 偏置；依赖分组信息和候选支持集 | 强化结构一致性，但没有任何反向传播 |

“没有专门的词义模块”不等于“没有语义学习”。DeBERTa 的上下文化表示、`compare_layer`、CLS 分类器，都能在金标监督下学习语义关系。问题是这些监督能否在小数据条件下覆盖并泛化到需要的判断。

Pair CE 也不是只教“合法”。`CE([S_FB,S_BF,S_EE], gold_combination)` 明确教哪一个金标组合正确。它与只奖励三个合法组合总概率的损失不同。

Token Alignment 比较的是同一输入中 Text 1 和 Text 2 的 token，并不是在对齐数据表里两条反向记录；反向记录是在 Pair CE 中组合的。

**4. Pair CE 的盲区：可以精确证明，但不能把“允许漂移”说成“必然向 N 漂移”**

代码入口：

- `experiments/deberta_lexical/evaluation.py:get_pair_logits`
- `experiments/deberta_feature_fusion/train_deberta_feature_fusion.py:train_pairs`
- `experiments/deberta_lexical/train_deberta_lexical.py:train_pairs`

当前组合分数为：

```text
S_FB = z0[F] + z1[B]
S_BF = z0[B] + z1[F]
S_EE = z0[E] + z1[E]
```

不变性一：每行 F/B/E 同时加上一个数，N 不变。三个组合分数获得相同的加数，所以组合 CE 不变，但单行可能从 F/B/E 变为 N。

不变性二：所有行 F 加 c、B 减 c，E/N 不变。FB/BF 的加减抵消，EE 不变。因此四分类中的 F/B 整体偏置也没有被这个目标识别。

CPU 双精度核验还使用了当前实际 Pair 权重 `[1,1,1.25]` 和 label smoothing=0.05：

```text
f = log(58)/2 = 2.0302215
e = log(1.25)/2 = 0.1115718
z0 = [f, 0, e, 3]
z1 = [0, f, e, 3]
gold combination = FB

组合概率 = [0.962655602, 0.016597510, 0.020746888]
Pair CE = 0.185835932
最大梯度绝对值约 1e-17
两行独立 argmax = N, N
两行 F/B/E 全部减 10：Pair CE 仍为 0.185835932
```

这说明在该例的最优组合分布处，两行仍可全判 N。增加训练轮次、减小 Pair loss 的显示值，都不能从这个目标中恢复缺失的监督。

但是，Pair CE 对这些方向不敏感，不代表其 logit 梯度主动沿零空间移动。共享表示、参数到 logit 的映射、Adam 动量、权重衰减以及两个阶段的交替，决定实际参数如何变化。N logit 的直接 Pair 梯度为零，也不代表训练后 N logit 一定不变。

当前 `0.5 * Pair CE` 是一整段独立 optimizer.step，不是每一步都有 `Row CE + 0.5 * Pair CE`。用 AdamW 时，也不能把“乘 0.5”理解成参数位移精确减半。

**5. Token Alignment 和 Lexical：有局限，但不是目前最强的塌陷解释**

原 Lexical 的固定权重读出诊断：

| 保留的读出分支 | PA W-F1 | PA F/B/E 正确 | N 正确 |
| --- | ---: | ---: | ---: |
| 完整模型 | 0.930569 | 516 | 98 |
| 去掉 Alignment | 0.926004 | 512 | 99 |
| 去掉 Lexical | 0.927500 | 514 | 98 |
| 仅 CLS | 0.919685 | 508 | 99 |

来源：`experiments/deberta_lexical_composition/outputs/run01_composition/composition_seed_42/branch_ablation_summary.csv`。

这是在已联合训练权重上移除读出分支，不是重新训练消融，不能直接解释每个组件对整个训练过程的因果贡献。但它至少不支持“Lexical 明显拖累当前模型”或“Alignment 完全无用”。

同一旧权重上把软匹配临时替换成最大项匹配，660 行 independent 和 PA 预测均没有变化。这个诊断没有训练新模型，也不是完整的 ColBERT 实现；它只能说明“把 softmax 改成硬选一个词”目前不是有证据的增益来源。

Alignment 的真实局限包括：

- 每个 token 必须把匹配概率分给对方 token，没有“无合适匹配”的 NULL 选项。
- 每个 token 的匹配独立归一化，多个词可落在同一个词上，没有短语整体覆盖保证。
- mean/max 汇总只保留部分局部证据；作用域关系没有显式结构监督。但输入 token 已上下文化，不能说它完全不知道顺序或结构。
- 对齐得分与“同义/上下位/无关”的语义关系不是同一变量，没有对齐金标或局部关系金标来保证二者一致。

Lexical 的覆盖率、长度等可能成为捷径，但判定它有害需要错误类型与特征依赖的证据。中心词不同也不能直接判 N，例如上下位词本就可能具有不同中心词。

**6. 最新融合版的错误，以及对“只缺语义数据”的修正**

最终 PA 错误共 55 行，涉及 32 个 `pair_id`：

| 真实到预测 | 行数 |
| --- | ---: |
| E → F/B | 22 |
| F/B → E | 16 |
| F ↔ B | 8 |
| N → F/B | 3 |
| N → E | 6 |

E 与 F/B 混淆占 38/55，约 69%。这支持优先研究“改写等价”与“真正限制范围”之间的区别，不支持直接推出“全是词典知识不足”。

两个方向都有错误，也不构成排除阈值、校准或类别权重影响的数学证据。

例子：

- `A beagle / the dog of the beagle breed`：原 Lexical 答 E 正确，新融合答 B。
- `A woman / An adult female`：原 Lexical 答 E 正确，新融合答 B。
- `a chocolate bar / a candy bar`：原 Lexical 答 F 正确，新融合答 E。
- `on the ground / on the grass`：原 Lexical 的反向行答 B 正确，新融合答 E。

模型有时已经能答对，换训练或读出后又丢掉。不能把所有剩余错误都解释为“从未获得过对应知识”。

`of some runway / from the airport`、新闻事件片段等，还可能需要原始语境或数据标注约定。给 WordNet 加一个 runway 与 airport 的关系，不能直接推出两个介词短语的四分类金标。也不能仅凭孤立片段宣布金标错误。

“各实验 F/B/E 都在 510–520”也需要限定为部分较高结果：本次融合是 508、CNN 是 506，不是所有实验都在这个区间。很多改动未突破，说明当前路线收益有限，但不是主干能力上限的证明。

**7. 论文、作者代码与实际适配程度**

以下均核对了论文页面与相关源码入口。来源中的结果不等于 DiCo 上已验证的收益。

| 路线 | 论文与年份 | 作者仓库、关键文件 | 能解决什么 | 适配判断与成本 |
| --- | --- | --- | --- | --- |
| 分开 Row 与 Pair 的任务读出 | [Multi-Task Deep Neural Networks for Natural Language Understanding, ACL 2019](https://aclanthology.org/P19-1441/) | [namisan/mt-dnn](https://github.com/namisan/mt-dnn)，[mt_dnn/matcher.py](https://github.com/namisan/mt-dnn/blob/master/mt_dnn/matcher.py)，`scoring_list` | 共享编码器、不同任务用各自输出头；不再直接让组级条件目标修改同一个 Row 分类器 | 对训练职责分离适配高；对提高 DiCo 泛化只属待验证。复用双向特征时不必额外跑主干；新增小型 Pair MLP |
| 共享/专用参数组织参考 | [LibMTL: A Python Library for Deep Multi-Task Learning, JMLR 2023](https://jmlr.org/papers/v24/22-0347.html) | [median-research-group/LibMTL](https://github.com/median-research-group/LibMTL)，`LibMTL/architecture/HPS.py`、`abstract_arch.py` | 清楚区分共享 encoder 与任务 decoder | 是实现框架，不是新发现的涨分算法。HPS 本身不是 2023 年新算法；无需整库迁入 |
| 可微稀疏匹配 | [Sparse Sequence-to-Sequence Models, ACL 2019](https://aclanthology.org/P19-1146/) | [deep-spin/entmax](https://github.com/deep-spin/entmax)，[entmax/activations.py](https://github.com/deep-spin/entmax/blob/master/entmax/activations.py)，`entmax15` | 允许无关匹配权重精确为零，仍可反向传播 | 可替换 `_align` 中归一化；不会自动提供 NULL 或词义监督。额外为排序等张量操作，无新增主干前向；当前硬匹配诊断无预测收益，优先级不高 |
| 词汇捷径去偏 | [Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases, EMNLP 2019](https://aclanthology.org/D19-1418/) | [chrisc36/debias](https://github.com/chrisc36/debias)，[debias/bert/clf_debias_loss_functions.py](https://github.com/chrisc36/debias/blob/master/debias/bert/clf_debias_loss_functions.py)，`ReweightByInvBias`/`BiasProduct` | 先单独识别已知捷径，再改变主模型的学习信号 | 只有证实 Lexical 捷径有害后才合适。已有联合训练残差分支不等于一个校准好的 bias-only 模型；要先拟合/审计偏差模型，不能直接拿现有分支输出套用 |
| 局部语义关系再组合 | [Neuro-symbolic Natural Logic with Introspective Revision for Natural Language Inference, TACL 2022](https://aclanthology.org/2022.tacl-1.14/) | [feng-yufei/NS-NLI](https://github.com/feng-yufei/NS-NLI)，[sampling_model_fix3.py](https://github.com/feng-yufei/NS-NLI/blob/main/sampling_model_fix3.py)，`local_relation_layer`；另有 `logic_table.py` | 显式局部自然逻辑关系，再组合成整体推断；比单纯匹配更接近“这个改变是否改变意义” | 研究方向适配中等、工程直接适配低。原实现有 GPT-2、七关系、投影/分块与强化学习；不是换一个层即可适配 DiCo 四类。需要数据/关系映射与更大改造，不能承诺训练成本或增益 |
| CRF/结构化预测实现参考 | [Torch-Struct: Deep Structured Prediction Library, ACL 2020 demos](https://aclanthology.org/2020.acl-demos.38/) | [harvardnlp/pytorch-struct](https://github.com/harvardnlp/pytorch-struct)，[torch_struct/linearchain.py](https://github.com/harvardnlp/pytorch-struct/blob/master/torch_struct/linearchain.py) | 实现结构化分布与约束推断 | 把同一三组合、同一加和势函数包装成 CRF，不会补出 N 监督，也不会消除上述不变性；不建议作为这次修复 |

第一条不是作者已经验证过的 DiCo 方案，属于根据任务共享/专用读出原则做的适配。它也不是仓库从未写过的想法：`experiments/pair_training/deberta_large_pair_head.py` 已有 Row 四分类头与 Pair 三分类 MLP，但用 large、CLS 特征和先 Row 后 Pair 的训练。该源码的存在不能当作受控结果，也不能把“再写两个头”宣传成没试过的必胜方法。

两个头只隔离输出参数，共享 encoder 仍会收到两个任务梯度。若停止 Row 监督后长时间只训练共享 encoder，单行仍可能变差。因此它是一条职责更清楚的改造思路，不是梯度冲突或泛化问题的自动解。

**8. 下一步该怎么选**

如果目标是解释 independent 为什么塌陷，先处理 Pair-only 对同一 Row 输出的更新方式；不能把训练期问题全归为“缺人工样本”。

如果目标是提高最终 PA 分数，现有证据支持把精力移向 E/F/B 的判别覆盖，而不是继续只改一致性损失。但不建议立刻复制扩大上一批模板，原因是新增 768 行没有验证线性收益，旧 scope 的一次增益也没有证明“每 200 行加 1 分”。

具体顺序：

1. 保留原 Lexical/.938 的代码和结果作为基准，本次融合记为当前单 seed 下未成功的改造。
2. 先审计已有人工家族：标签是否严格成立；是否真在只改一个信息因素；是否已大量覆盖 `形容词 + 名词 / 名词 + 定语从句`；改写等价、真限制、交叉属性和事件状态是否都有对照。家族数量不是独立语义规则的数量。
3. 只为 Train 暴露的覆盖缺口补新的、可独立判断的对照；别从 Dev 的具体词串改写生成。已有 `a blue car / a car that is blue` 模式，再换颜色不一定增加边界信息。
4. 同样本数、同类别构成、近似同训练预算，比较新内容与 Train 复制对照。最好把人工家族整体留一部分作诊断，避免在同模板的新词上把记忆当泛化。
5. 报告同一 checkpoint 的 independent、三组合解码与修复/破坏数量。若单行/结构都有要求，不能只用 decoded F1 选一个塌陷态，再引用别轮的独立分。

生成器若反复按 Dev 错误修改，即使文本完全不重复，也属于 Dev 引导的开发，不能宣称 Dev 从未参与方法设计。原句去重与方法选择独立性是两种不同的隔离。

**9. 不应再次当作未试新方法的改动**

- 相同三组合上套 log_softmax、CRF，不能提供新监督。
- 加性分数上的全 16 组合 CE 在无权重、无额外耦合项时等于两行 Row CE 之和；不是新组级推理能力。
- 16 组合 hinge、混合训练、NN 反向训练、SAM、SiFT、梯度投影/约束等已有实验，应按各自实际结果引用。
- 交换一致性约束可以由全答 N 满足，不能替代正确性监督；但四类 KL 并非一般地对 F/B/E 相对 N 的共同平移不变。
- 同一组内把候选限制为 FB/BF/EE，是利用当前数据的候选支持集。NN 在逻辑上同样合法，不能把排除 NN 说成标签定义的必然推论。

pair-aware 自身不更新模型，SoftCons=1 有一部分是解码保证。结构分仍评价模型对 FB/BF/EE 的相对排序，不能把它全部称为“假分”，也不能把它当作单行四分类能力。

**最终判断**

训练设计最需要解释和修正的是 Pair CE 对共享四分类输出的使用方式；剩余结构错误最值得研究的是 E/F/B 语义边界。两者不是同一个瓶颈。数据增强是有针对性的候选路径，但目前没有证据证明它是唯一有效路线，也没有依据保证下一批数据或某篇方法达到 0.95。
