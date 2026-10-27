# DeBERTa + Token Alignment + Lexical：独立能力与Pair训练的研究复核

核对日期：2026-10-25。本轮只读取已完成实验的预测、代码与公开论文/作者仓库，运行CPU统计；不训练，不修改现有训练入口。

复算脚本：[analyze_independent_research.py](analyze_independent_research.py)。结果：[分析输出目录](outputs/independent_research_20261025)。

**结论先行**

最大问题有两个层次：旧Pair-only阶段造成的四类读出退化，以及恢复四类监督后仍存在的真实泛化错误。前者已能通过持续Row监督缓解，后者不能继续用结构分的错误分布替代分析。

在已核实、且可读代码中尚未找到严格实现的新候选里，若只选一个训练方法，优先 **Child-Tuning-D**；它是针对小样本微调泛化的参数坐标选择方法，不是新的Pair损失。Mixout次之，结构教师边缘蒸馏为有条件备选。R3F与已试R-Drop/SiFT的目标接近，优先级较低。Dataset Cartography作为同轮诊断记录，而非承诺涨分的算法。

这些排序是适配判断，不是对DiCo提升的概率估计，更没有0.95保证。

**1. 先纠正总结中会改变决策的事实**

1. **independent与pair-aware并不要求相反的N分数。** 真实N应被独立选为N，真实F/B/E应压过N；三组合Pair训练与解码只是没有评价后一个要求。若两行都独立正确，金标组合加和也一定最高，两个目标可以同时达到100%。
2. **506—522/554是部分结构解码结果的观察范围，不是独立语义能力或数学天花板。** 最高522/554本身约94.22%，也不是统一的93.9%。旧Lexical结构后516，但独立只有224；稳定混合模型独立498，结构后514。不能混用。
3. **N正确数81不是判别上限，96也不是独立F1达到0.87的必要条件。** 正确数取决于N召回、误报及其余类的表现。一个支持数仍为190/190/174/106、N只对81的构造混淆矩阵可以得到W-F1=0.894912。它是反例，不是某个已训练模型。
4. **0.938145对应3290行、无Lexical。** 不是3442行的Lexical模型。Lexical+3042的旧结果是0.930569；Lexical+3242为0.929192；新Lexical+3290是另外一次实验，不能提前与历史最好结果合并。
5. **解码增益要说明比较起点。** 原0.938权重的三级读出为0.547177→0.899494→0.938145。前一步是在配对行排除N，后一步才是使用反向分数相加；对应+35.23与+3.87个百分点，并非统一的“+5.3”。
6. **NLI初始化与增强的单点正结果，不等于已验证的稳定因果。** raw实验实际启用了Lexical，更近的对照应是NLI Lexical的0.930569，而非无Lexical的0.9248。200/248增强还改变步数与输入分布，需与匹配复制及种子对照区分。

“Train独立0.977→0.637”发生在同一轮切换训练目标后，首先是训练目标/读出的退化；“Train约0.98、Dev约0.85”才是常见的泛化差距。这不是一个问题的两种说法。

**2. 以Lexical模型的真实独立错误为分析对象**

| 已完成实验与状态 | 独立W-F1 | 同轮结构W-F1 | 独立正确 | 独立F/B/E正确 | N正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧两阶段Lexical，ep3 | 0.468191 | 0.930569 | 322 | 224 | 98 |
| 混合Lexical run01，ep2 | 0.850565 | 0.873506 | 563 | 498 | 65 |
| OOF难例加权Lexical，ep7 | 0.852621 | 0.892869 | 562 | 486 | 76 |

run01 ep2只有逐轮预测记录，保存权重的是按结构指标选出的另一个epoch，不能用那个模型文件部署ep2的成绩。OOF run06明确按independent选模。

混合Lexical ep2的独立混淆矩阵：

| Gold / Prediction | F | B | E | N |
| --- | ---: | ---: | ---: | ---: |
| F | 165 | 4 | 11 | 10 |
| B | 1 | 175 | 7 | 7 |
| E | 6 | 2 | 158 | 8 |
| N | 9 | 16 | 16 | 65 |

97条错误分成：

- N与非N边界66条：41条N漏判、25条非N误判N，占68.0%。
- E与F/B混淆26条。
- F/B方向混淆5条。

OOF模型的98条错误中，N边界67、E与F/B为28、方向3，整体形状相似。因此，以独立能力为目标时，不能继续说“剩余主要只有30条E↔F/B”。那是另一模型、另一解码方式的统计。

**3. N错在哪里：SIMI/REL边界，且高置信错误同时出现在两个方向**

| 模型 | SIMI正确/81 | REL正确/22 | OPPO正确/3 |
| --- | ---: | ---: | ---: |
| 混合Lexical ep2 | 48 | 14 | 3 |
| OOF Lexical ep7 | 58 | 15 | 3 |

子类来自仓库已有PhrasIS映射审计，仅作分析，不进入训练。

混合模型高置信N漏判包括：

- `with network-sharing deal / network-sharing , spectrum deal`，金标N，预测B，置信度0.955。
- `on Syria abuse / on Syrian chemical arms`，金标N，预测B，0.940。
- `Opens / begins operation`，金标N，预测E，0.938。
- `performs / to do`，金标N，预测E，0.926。

反向错误也存在：`to announce / to set out`金标E却预测N；`ginger and white cat / orange and white cat`也被预测N。

这不是只要把N整体抬高就能解决。既有训练语义判据、片段的具体词义、事件/论元结构以及PhrasIS的SIMI/REL映射约定，都可能影响这些例子。孤立片段不足以宣布金标错误，更不能把一般词典上下位直接改写为该任务标签。

OOF相对混合模型的N正确数65→76，同时非N→N从25→37，全部正确数563→562。N的ROC-AUC从0.913426降到0.906512；固定按p(N)取前90条时命中67→66。这一比较没有显示N排序整体改善，也不能据召回提升单独宣布语义边界学好了。

**4. Pair结构的确切问题与它修不了的事**

当前三组合：

```text
S_FB=z0F+z1B
S_BF=z0B+z1F
S_EE=z0E+z1E
```

存在两个精确不变方向：

- 每行F/B/E同时加c，N保持不变，组合softmax不变；单行的N/非N判断可改变。
- 所有行F加d、B减d，EE保持，三个组合分数都不变；单行可能整体偏F或偏B。

Pair CE对N logit的直接梯度为0，但共享参数、Adam状态和weight decay仍能改变N输出。对logit求梯度时，F/B/E梯度和为0；不能说这个损失必然主动沿整体平移方向压低F/B/E。正确说法是：它不识别这些方向，参数耦合和分阶段更新允许四分类读出退化。

若每行独立金标均最大，对任何不同候选(a,b)：

```text
S_gold-S_ab=(z0_gold0-z0_a)+(z1_gold1-z1_b)>0
```

因此独立全对可以保证组合全对，反向不成立。问题是目标监督不充分及其优化路径，不是任务标签存在必然冲突。

混合Lexical ep2中，277个真实配对组的独立预测为：

| 组合性质 | 组数 | 其中错误行 |
| --- | ---: | ---: |
| 已是FB/BF/EE | 244 | 12 |
| NN：交换一致，但语义错误 | 8 | 16 |
| 交换不一致 | 25 | 28 |

这些统计不包括41条N单例漏判。交换一致性KL对NN可为0，也不能改变已经一致但选错关系的组。只优化一致性不足以覆盖当前独立错误。

当前混合模型的pair-aware修好23条、破坏7条，净多对16条；不能说结构解码只会修不会破坏。

在无权重/无平滑的简化情形，若合法组合总概率为M：

```text
Pair CE = RowCE0 + RowCE1 + log(M)
Semantic loss = -log(M)
Pair CE + Semantic loss = RowCE0 + RowCE1
```

所以同候选上的posterior projection、合法概率约束、在线约束蒸馏，在这里有明显的代数重合；换名称并没有增加语义监督。实际加权平滑版不能直接套同一个数值分解，但上述不变性仍成立。全16组合加性CE、同三状态CRF、已试hinge/约束，也不应被重新包装为全新能力来源。

**5. 校准空间不是学习上限**

固定混合ep2的全部F/B/E排序，只扫描N的统一logit偏移，使用Dev金标回顾性计算：

- 当前W-F1=0.850565；理想选择该偏移可达0.861985，N命中73、N误报30。
- 若偏移直到N命中96，需要预测247条N，其中151条为误报，W-F1仅0.767764。

这不是建议用Dev金标挑阈值提交，也不是实际训练提升。它说明两个结论：一部分读出差距可能是校准；强行规定N必须96并不合理。真正提升仍要改善排序、判据及其他类别，同时用独立数据校验。

**6. 检索后保留的方法与实现入口**

| 优先级 | 论文 | 已核验作者代码 | 对当前问题的关系 | 额外成本 |
| --- | --- | --- | --- | --- |
| 1，泛化训练候选 | [Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.749/) | [RunxinXu/ChildTuning](https://github.com/RunxinXu/ChildTuning)，[ChildTuningD.py](https://github.com/RunxinXu/ChildTuning/blob/main/ChildTuningD.py)的calculate_fisher；[ChildTuningOptimizer.py](https://github.com/RunxinXu/ChildTuning/blob/main/ChildTuningOptimizer.py) | Fisher重要度选择参数坐标，限制小样本过度适配；不是解决Pair零空间本身 | 额外一次Train前后向估重要度；正常每步不增加主干前向，增加掩码运算和存储 |
| 2，泛化/稳定性备选 | [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models，ICLR 2020](https://arxiv.org/abs/1909.11299) | [bloodwass/mixout](https://github.com/bloodwass/mixout)，[mixlinear.py](https://github.com/bloodwass/mixout/blob/master/mixout/mixlinear.py)、[mixout.py](https://github.com/bloodwass/mixout/blob/master/mixout/mixout.py) | 随机混入初始NLI参数，限制遗忘；不同于每轮Row后的参数锚定 | 无额外主干前向；增加参考权重和掩码内存、逐元素计算 |
| 3，有条件结构迁移 | [Structural Knowledge Distillation: Tractably Distilling Information for Structured Predictor，ACL 2021](https://aclanthology.org/2021.acl-long.46/) | [Alibaba-NLP/StructuralKD](https://github.com/Alibaba-NLP/StructuralKD)；README确认CRF⇒MaxEnt及train.py入口；具体后端函数未核验 | 冻结结构教师，把组合边缘分布传给独立学生；不同于Row缓存KL，但不增加N金标信息 | 若已有Train logits，可离线生成；否则一次教师Train推理，再训练一个学生；OOF教师会显著加成本 |
| 4，低于上述候选 | [Better Fine-Tuning by Reducing Representational Collapse，ICLR 2021](https://arxiv.org/abs/2008.03156) | [facebookresearch/fairseq/examples/rxf](https://github.com/facebookresearch/fairseq/tree/main/examples/rxf)，[sentence_prediction_r3f.py](https://github.com/facebookresearch/fairseq/blob/main/examples/rxf/rxf_src/sentence_prediction_r3f.py) | embedding小噪声与四类对称KL；接近已试R-Drop/SiFT目的，没有新的词义监督 | 两次主干前向，两图参与反向；不能把一次backward调用说成只有一图成本 |
| 同轮诊断，非涨分算法 | [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.746/) | [allenai/cartography](https://github.com/allenai/cartography)，[selection_utils.py](https://github.com/allenai/cartography/blob/master/cartography/selection/selection_utils.py)、[train_dy_filtering.py](https://github.com/allenai/cartography/blob/master/cartography/selection/train_dy_filtering.py) | 记录Train金标概率、波动、正确/遗忘轨迹，区分难例与阶段退化；不能自动认定错标 | 在线记录几乎无额外前向；统一epoch末推理则每轮加一次Train推理 |

Papers With Code当前入口实际重定向到[Hugging Face论文页](https://paperswithcode.com/)；方法事实与实现入口以原论文和作者仓库核对，不依赖聚合站的二手结论。

**7. 为什么Child-Tuning-D排在第一，而不是再换一个Pair损失**

[原论文Table 2](https://aclanthology.org/2021.emnlp-main.749.pdf)包含用5k MNLI/SNLI微调BERT-large、在其他NLI集测试的实验，D版本域外平均准确率分别提高1.53/0.81点；SNLI→SICK仍有-0.08。小样本与NLI迁移证据较贴近，但模型和标签空间不同，不能推算DiCo涨幅。

它与旧方法的区别：

- 冻结底6层按整层指定；Child-D依据Train梯度重要度选择层内坐标。
- LoRA把更新限制在低秩子空间；Child-D保留原参数形式，选择部分坐标的数据梯度。
- 它不更改Token Alignment、Lexical、四分类头或推理协议，不需要外部训练集。

直接接入时只借用Fisher估计/掩码逻辑，不整份复制2021版自定义Trainer；原代码依赖旧Transformers私有接口。为避免把Adam实现差异混入对照，应保留当前AdamW、eps、两组学习率等设置。

需要处理的细节：

1. 重要度只从Train的完整四分类Row CE计算，不能只用FB/BF/EE的Pair CE，更不能读Dev标签。
2. 作者估计使用逐批平方梯度，并在累计前逐参数裁剪；它是重要度近似，不是精确逐样本Fisher。
3. 名称筛选应限定`deberta_model.encoder.layer.`；原脚本简单匹配`layer`，不能照搬后误把新增compare_layer也遮掉。CLS/Alignment/Lexical头保持可学，embedding按固定方案处理并披露，不能把保留比例说成整个模型训练参数比例。
4. 原作者优化器在遮罩后仍做weight decay，已有动量也可能产生位移，因此“梯度遮罩”不等于所有被遮参数严格冻结。
5. Fisher准备阶段使用独立数据迭代器，恢复主训练随机状态，清空梯度，不执行optimizer.step，避免额外训练或shuffle扰动被混进方法收益。

该方法不保证消除梯度冲突。例如g_row=[2,1,-3]、g_pair=[-1,2,1]，原内积-3；掩码[1,0,1]后内积-5。因此应在持续四类监督的训练底座上研究其泛化，而不是指望它替代Row约束。

**8. Mixout、R3F与已试锚定的区别和限度**

Mixout参考点应是固定的初始NLI backbone，不是每轮Row结束的参数。之前run06是RecAdam-inspired近端Row锚定，不能称严格复现过Mixout或原RecAdam。Mixout能减轻对初始NLI知识的遗忘，但也可能限制需要的任务适应。

作者旧MixLinear的target不是注册buffer，移植到当前PyTorch时要处理设备、dtype和保存，不能直接复制旧模块假定无兼容问题。仅替换预训练backbone的Linear；新随机CLS/Alignment/Lexical头不应无差别混回随机初值。

R3F必须让噪声输入仍保留原token IDs来计算Text1/Text2 mask，Lexical特征保持相同。全答N且对噪声稳定时，KL也可以接近0，因此它不能单独补齐当前N边界。

**9. 结构蒸馏：能转移什么，不能转移什么**

冻结教师给出q=softmax(S_FB,S_BF,S_EE)，两行学生目标为：

```text
q0=[qFB,qBF,qEE,0]
q1=[qBF,qFB,qEE,0]
```

N单例仍按真实Row金标监督，推理时学生独立四分类，不强制单例N。此公式不需要安装完整StructuralKD；属于两节点任务的适配。作者README有CRF⇒MaxEnt入口，但本轮未取回具体后端模型实现，不能声称已核实可直接运行其完整NER配置。

对仓库旧0.938教师的1380个Train组合，离线复算：

| 温度 | 平均最大组合概率 | 最大概率>0.98的组比例 |
| --- | ---: | ---: |
| 1 | 0.984009 | 80.65% |
| 2 | 0.890529 | 2.75% |
| 4 | 0.689647 | 0% |

不能仅凭T=1接近one-hot就武断判定无可传递软信息。但在非N行上，N目标为0与原硬标签CE相同；新增信息仅是F/B/E的软偏好。若教师已错或只是拟合Train，蒸馏可能复制错误，不能期待它凭空解决N的词义或语境问题。

若用当前学生自己即时刷新q，部分目标又退化为已试的合法概率约束；需要冻结独立教师，必要时用Train内OOF减少自我拟合。旧教师使用了3290行增强时，学生若只用3042行，也必须披露教师额外数据来源。

**10. 数据侧该做什么，什么不值得重试**

Dataset Cartography应记录稳定Row/mixed模型的四类金标概率，不能把纯Pair阶段造成的分数漂移当成样本天生歧义。低置信样本可能是难语义、语境不足、任务约定或错标，不能自动删掉。保留容易样本，避免对3042行做大比例过滤。

JTT的“早期错例再加权”与当前OOF难例加权已有明显重合；后者已跑出0.852621及对应副作用，不应换个名称重复当新方法。更多同模板增强已被匹配复制对照约束，不建议继续按固定斜率扩量。

原始文件审计：按casefold与空白归一化，官方Train与Dev有7种相同无序文本对，覆盖8个Dev pair_id、15行；Train还有同一有序输入`South Africans / South Africa`分别为F和N的记录。这些是官方输入已有的现象，不是本轮新增的泄漏，不能据此修改Dev金标或直接宣称全部错误是噪声。论文需披露，并可补去重子集敏感性分析，官方全量成绩仍照常报告。

**11. 下一步的具体决策**

正在运行的Lexical+3290两阶段配置保持原样，不在这轮研究中改动。

若下一次只做一个新方法实验，建议选择已达到独立约0.85的Lexical混合训练底座，在相同3042行、相同损失/学习率/组批次上仅增加Child-Tuning-D。持续Row CE是为了不让结构盲区掩盖泛化效应；它不是重新发明一个混合训练方法。数据与架构仍使用现有版本，不叠加新增强、SAM或新语义头。

建议固定一个保留比例作为起点，例如encoder block坐标20%；这是待验证超参数，不是论文已确定的本任务最优值。Fisher只看Train，新增诊断记录各类错误、N误报/漏报、N-AUC/AP和稳定的局部对照，选模与主比较使用同一checkpoint的independent。

第一次以一组训练看方向，不声称显著或80%成功率；若有明确信号，再对基线/方法补相同种子。不能把0.547→0.85的退化修复当成Child-D新提升，也不能把其他轮/其他模型的结构成绩拼入新模型。

如果必须坚持“末尾完整Pair-only”并仍按pair-aware选模，independent改善就不应成为该实验的主要结论；那样的选择目标仍会允许保存四分类退化但结构高分的状态。
