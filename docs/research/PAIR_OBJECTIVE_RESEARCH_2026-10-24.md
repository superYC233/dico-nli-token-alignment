# DeBERTa 0.9248 / 0.938：目标欠约束、阶段漂移与可借鉴方法

当前项目入口见[项目README](../../README.md)，本页保留2026-10-24的研究记录。

检索与本地核对日期：2026-10-24。只读论文、源码、训练配置和已有预测；未修改训练代码、未训练。

检索覆盖arXiv、ACL Anthology、论文正式会议页、Papers with Code与GitHub。Papers with Code部分页面本轮无法稳定打开，不能把搜索索引中的“有代码”当成已经读过实现。下列代码入口优先给作者仓库；复现、后续论文实现和无法验证的路径均单独说明。

## 1. 两个模型到底有什么问题

| 同一官方Dev660行 | 原0.9248，ep7 | 原0.938，ep2 |
|---|---:|---:|
| 官方Train行数 | 3042 | 3042+scope200+relation48=3290 |
| independent W-F1 | 0.528004 | 0.547177 |
| pair-aware W-F1 | 0.924794 | 0.938145 |
| 独立非N→N | 307 | 297 |
| 独立F/B/E正确 | 239 | 248 |
| pair-aware F/B/E正确 | 512 | 518 |
| N正确 | 98 | 101 |
| pair-aware错误行 | 50 | 41 |

两版主干均为NLI初始化DeBERTa-v3-base，CLS四分类加Token Alignment残差，全参数微调。先完整Row四分类CE，再完整三组合Pair CE。Row权重[1,1,1,1.5]，Pair权重[1,1,1.25]、smoothing0.05、Pair系数0.5，lr2e-5、batch16、10轮。

原版实现入口：`train.py::train_model_real_pair_ce`；0.938入口：`experiments/deberta_manual_aug/train_relation_manual.py::train_staged`。
当前根目录train.py已有后续维护，不把当前源码每一行等同于历史运行；历史成绩以逐条预测和保存配置为证。

| 结构解码后的错误 | 0.9248 | 0.938 |
|---|---:|---:|
| E与F/B混淆 | 36 | 30 |
| F/B方向互反 | 6 | 6 |
| N判错 | 8 | 5 |

0.938相对0.9248净多9行正确：配对行多6、N多3。它仍有30/41行属于E与F/B边界；只修训练数值漂移不一定修复这部分语义错误。

三件事必须分开：

1. **目标欠约束**：Pair损失没有唯一确定每行四类判断。
2. **阶段漂移**：Pair-only更新以后，原先正确的单行判断发生变化。
3. **泛化差距**：健康四分类训练也可能Train很好而Dev明显下降。

例如0.938所选checkpoint的Train独立F1只有0.6365、结构解码后3280/3290正确；不能把它描述为“独立Train已经0.977”。后续Pair+N/非N版本才有Train独立0.9793 / Dev独立0.8246。

### 可以严格证明的部分

令两行分数为z1、z2：

```
S_FB = z1_F + z2_B
S_BF = z1_B + z2_F
S_EE = z1_E + z2_E
```

- 改变任意N分数，三个S都不变。
- 对所有行令F加c、B减c，三个S都不变。
- 对同一对的第一行F/B/E同加a，第二行同减a，三个S也不变。

这些是输出空间的不变方向；它们允许单行argmax改变而Pair损失不变。并非普通softmax共同加常数那种无害不变性。
这并不证明真实网络每一步都恰好沿上述方向更新，也不能把输出空间线性变换直接等同于某个固定参数方向。

最新N/非N辅助损失仍存在欠约束：改变F/B/E内部相对分数，再补偿N以保持P(N)，可以同时保持Pair和二分类损失，却改变单行F/B/E。

### 已有结果约束了下一步选择

| 已做分支 | 同一所选模型independent / pair-aware |
|---|---|
| 普通混合梯度，run13 mixed | 0.832464 / 0.902140 |
| Row优先非对称冲突投影，run13 project | 0.824523 / 0.900597 |
| Row缓存四类KL保持，best Pair | 0.828667 / 0.879227 |
| 完整Row回放，run05 memory版本 | 约0.829652 / 0.895390 |
| N/非N辅助，run24 | 0.824637 / 0.885709 |
| SiFT，run06 | 0.420424 / 0.926958 |

这些选模标准不完全相同，不能拿来做严谨算法总排名；它们说明“再加一个保持项就一定涨分”没有依据。
run13固定checkpoint附近24批诊断：冲突11/24，平均余弦-0.02679，加权Pair/Row梯度范数比均值0.08377；普通混合更新与Row梯度内积均为正。该诊断不是全训练轨迹，不能证明强梯度冲突是唯一根因。

## 2. 相关性与成本口径

- **是**：适配后的约束确实观察并限制我们缺失的四类输出方向；不等于原论文验证过DiCo。
- **部分**：解决主辅干扰、遗忘或稳定性，但不直接消除三组合的不变性。
- **不是**：主要解决其他问题；可作为泛化或结构工程备选。
- 成本按相同样本/批次曝光的普通一次前向、一次反向比较。F/B表示完整编码器前向/反向，不是原训练标签F/B。
- 184,001,800个FP32参数一份约0.736GB，约0.685GiB。显存数字仅是新增张量预算，不是整次训练峰值。

## 3. 主辅任务优化：优先看BLOOP，经典方法作对照

| 方法与论文 | GitHub及已核实入口 | 与零空间问题关系；接入点 | 额外成本 |
|---|---|---|---|
| **BLOOP** — [Careful with that Scalpel: Improving Gradient Surgery with an EMA](https://proceedings.mlr.press/v235/hsieh24a.html)，ICML2024；[arXiv](https://arxiv.org/abs/2402.02998) | 作者[apple-aiml-research/ml-bloop](https://github.com/apple-aiml-research/ml-bloop)：[src/bloop/optim.py](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/optim.py)::bloop_direction；[src/bloop/utils.py](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/utils.py)::tree_proj/update_ema | **部分，梯度类最贴近。** 用Row主梯度EMA定义保护方向，始终投影Pair辅助梯度；改现有row_priority_gradients.py，不是仅调系数。作者代码为JAX/Flax，需要移植。 | 同一联合前向、两次梯度遍历，加主梯度EMA约0.74GB；checkpoint反向重算另计。无独立额外教师前向。 |
| **FAMO** — [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792)，NeurIPS2023 | 作者[Cranial-XIX/FAMO/famo.py](https://github.com/Cranial-XIX/FAMO/blob/main/famo.py)::FAMO.backward/get_weighted_loss/update | **部分。** 根据相对下降速度调Row/Pair权重，不提供Row不可退化保证；接联合损失向量。 | 不存全套逐任务梯度；作者示例在step后另做一次无梯度前向更新loss，不能称完全零额外前向。 |
| **PCGrad** — [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)，NeurIPS2020 | 作者TensorFlow：[PCGrad_tf.py](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)；PyTorch复现：[pcgrad.py](https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py)::pc_backward | **部分。** 处理负内积冲突；本项目已有Row优先非对称版，不能当成从未试过。 | 两套任务梯度；通常同一前向、两次梯度计算，加梯度缓存。 |
| **CAGrad** — [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/abs/2110.14048)，NeurIPS2021 | 作者[Cranial-XIX/CAGrad/toy.py](https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py)；可移植参考[Nash-MTL methods/weight_methods.py](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py)::CAGrad | **部分。** 平衡平均损失与最差任务改进；将Row/Pair同时放进梯度组合，不适用于只有Pair的更新。 | 两套梯度、2×2 Gram矩阵、小型优化求解及CPU同步。 |
| **Aligned-MTL** — [Independent Component Alignment for Multi-Task Learning](https://arxiv.org/abs/2305.19000)，CVPR2023 | 作者SamsungLabs/MTL本轮访问404；核实复现[LibMTL/weighting/Aligned_MTL.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/Aligned_MTL.py) | **部分。** 处理任务梯度病态和支配，未补Pair缺失方向；替换合成梯度规则。 | 两套梯度加小矩阵分解；核实版本仍用torch.symeig，现代PyTorch需改eigh。 |
| **Nash-MTL** — [Multi-Task Learning as a Bargaining Game](https://arxiv.org/abs/2202.01017)，ICML2022 | 作者[AvivNavon/nash-mtl/methods/weight_methods.py](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py)::NashMTL | **部分。** 对称谈判不自动把Row设为不可牺牲的主任务；接任务梯度和谈判权重。 | 官方实现逐任务autograd.grad后还做加权反向，另有CVXPY/ECOS求解；不算轻量首选。 |
| **GradNorm** — [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html)，ICML2018 | PyTorch复现[LucasBoTang/GradNorm/gradnorm.py](https://github.com/LucasBoTang/GradNorm/blob/main/gradnorm.py)，非原作者仓 | **部分，较间接。** 校正量级与训练速率，不管理无监督方向；在共享层计算两项范数学权重。 | 增加选定层梯度计算及权重更新；不必缓存全部参数的每任务梯度。 |
| **Uncertainty weighting** — [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)，CVPR2018 | 作者[yaringal/multi-task-learning-example](https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb)；易读比较实现[Nash-MTL::Uncertainty](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py) | **不是直接同类。** 标量权重不能把某方向上的零梯度变成有监督梯度；加两个可学习log-variance。 | 一次合并反向，加两个标量，计算开销很小。 |

2023–2025不是所有新方法都更简单或已证明更稳：FAMO省逐任务梯度缓存，但有额外无梯度前向；BLOOP机制更贴主辅保护，但仍需双梯度和EMA。
AuxiNash(ICML2023)也区分主辅，但需验证集偏好/隐式微分；若采用必须另划Train内部验证，不能直接用官方Dev作元梯度。此次不列首选。

## 4. 把一致性/保持作为约束

### Primal-dual与DomiKnowS

- 论文：[A Primal-Dual Formulation for Deep Learning with Constraints](https://papers.nips.cc/paper_files/paper/2019/hash/cf708fc1decf0337aded484f8f4519ae-Abstract.html)，NeurIPS2019。
- 作者仓：[dair-iitd/dl-with-constraints](https://github.com/dair-iitd/dl-with-constraints)，NER说明列出dual_training.py/gan_trainer.py；精确嵌套源码本轮未进一步读到，不编造完整文件路径。
- 更现代的明确入口：[DomiKnowS: A Library for Integration of Symbolic Domain Knowledge in Deep Learning](https://aclanthology.org/2021.emnlp-demo.27/)，EMNLP2021 Demo；作者[HLR/DomiKnowS](https://github.com/HLR/DomiKnowS)，[domiknows/program/lossprogram.py](https://github.com/HLR/DomiKnowS/blob/main/domiknows/program/lossprogram.py)::PrimalDualProgram.train_epoch、[model/lossModel.py](https://github.com/HLR/DomiKnowS/blob/main/domiknows/program/model/lossModel.py)::PrimalDualModel。
- 新NLP应用：[Neuro-symbolic Training for Reasoning over Spatial Language](https://aclanthology.org/2025.findings-naacl.128/)，Findings NAACL2025，使用DomiKnowS的primal-dual约束训练。不是DiCo直接验证。
- **关系：是，前提是我们新增的约束观察完整四类输出；只约束原三个Pair分数仍无效。**
- 接入：以Pair为目标时，对四类Row损失、逐行相对margin或全四类输出变化设置预算；或反过来以Row为主目标、交换一致性为约束。用训练约束违反度更新乘子，不用Dev做梯度更新。
- 成本：可共用前向和一次反向，少量对偶标量更新便宜；如果另取保护batch则增加该batch前后向。需要设置预算与对偶步长，平均约束可能掩盖少数类别退化。随机非凸primal-dual并不保证每个Adam步都严格可行。

### Cooper：2025通用实现工具，不冒充NLP模型

- [Cooper: A Library for Constrained Optimization in Deep Learning](https://arxiv.org/abs/2504.01212)，2025。
- 作者[cooper-org/cooper](https://github.com/cooper-org/cooper)：README的ConstrainedMinimizationProblem.compute_cmp_state、ConstraintState、SimultaneousOptimizer.roll完整示例已核实。src/cooper/optim下具体文件本轮未稳定取回，因此不硬写猜测路径。
- **关系：部分，是实现工具；能否处理我们的漂移取决于约束定义。** 可把上述类别级Row预算写入CMPState，复用现有AdamW作primal优化器。
- 成本与上述可微约束相同；不是安装一个库就自动获得点级或Dev保证。

### Dynamic Barrier

- [Automatic and Harmless Regularization with Constrained and Lexicographic Optimization: A Dynamic Barrier Approach](https://papers.nips.cc/paper_files/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html)，NeurIPS2021。
- 未找到可靠原作者源码；BLOOP作者仓[optim.py::dynamic_barrier_direction](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/optim.py)是后续参考实现，不是2021官方代码。
- **关系：部分，直接处理主辅优先级，但局部方向约束不等于所有输出保持。** 接入现有双梯度组合；成本近BLOOP/PCGrad，仍需Row梯度。

## 5. 结构化预测：CRF或hinge为什么未必是新解法

| 方法与论文 | 代码入口 | 关系与接入 | 额外成本 |
|---|---|---|---|
| **BERT + structured hinge** — [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://aclanthology.org/P19-1340/)，ACL2019 | 作者[nikitakit/self-attentive-parser](https://github.com/nikitakit/self-attentive-parser)：[src/benepar/parse_chart.py](https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/parse_chart.py)接AutoModel；[decode_chart.py](https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/decode_chart.py)::SpanClassificationMarginLoss | **不是直接解法。** Pair criterion可改成max(score+cost)-gold_score，但三个原分数的不变方向不变。 | 三候选max操作很小，不增加编码前向。 |
| **CRF / Torch-Struct** — [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/)，ACL2020 Demo | 作者[harvardnlp/pytorch-struct](https://github.com/harvardnlp/pytorch-struct)：[torch_struct/distributions.py](https://github.com/harvardnlp/pytorch-struct/blob/master/torch_struct/distributions.py)::log_prob/marginals/argmax；linearchain.py | **不是，若候选仍为FB/BF/EE。** 两行作两节点，四类分数作unary；仅给原合法组合加固定mask，本质还是当前小结构模型。 | 两节点四标签计算很小，不增加编码前向；新增可学transition则另引入参数和建模假设。 |

两个检查可避免重复实验：

1. 原三个合法组合的CRF-NLL就是当前组合CE的结构化解释，换名不会补N。
2. 若允许全部16组合、分数仍为z1[a]+z2[b]、没有额外非加性势，则分区函数分解，结构NLL严格等于两行Row CE之和。只有加入真正的新结构/势或监督，才不是重写已有目标。

## 6. 损失不变性与锚定

| 方法与论文 | 代码入口 | 关系与接入 | 额外成本 |
|---|---|---|---|
| **RecAdam** — [Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting](https://aclanthology.org/2020.emnlp-main.634/)，EMNLP2020 | 作者[Sanyuan-Chen/RecAdam/RecAdam.py](https://github.com/Sanyuan-Chen/RecAdam/blob/master/RecAdam.py)::RecAdam.step；run_glue_with_RecAdam.py | **部分。** 参数锚定能抑制Pair不观察方向上的位移，但不保证输出不变。可每轮Row后复制参数，Pair期间以它为锚；这是我们的适配，不是原论文的预训练锚点。 | 无额外编码前后向；一份参数约0.74GB及逐参数更新。原coefficient5000不能不加核对地照搬；不同优化器状态安排也会影响对照。 |
| **APO** — [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment](https://aclanthology.org/2025.tacl-1.22/)，TACL2025，预印本2024 | 作者[ContextualAI/CLAIR_and_APO](https://github.com/ContextualAI/CLAIR_and_APO)；实现集成[TRL dpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)的apo_zero/apo_down分支 | **部分，理论类比很贴近。** 相对偏好目标没有指定绝对变化，APO补锚定方向。四分类版需重新推导，不能直接调用DPOTrainer训练DeBERTa；可借“每行概率变化也要受控”的原则。 | 原算法针对生成LM，不能给本项目现成成本；若适配为Train参考概率缓存，每轮一遍无梯度缓存，不必常驻教师。与已有KL保持存在重叠。 |
| **OGD** — [Orthogonal Gradient Descent for Continual Learning](https://arxiv.org/abs/1910.07104)，AISTATS2020；后续[Generalisation Guarantees…](https://arxiv.org/abs/2006.11942)，2020 | 后续论文作者[MehdiAbbanaBennani/continual-learning-ogdplus/ogd/ogd_core_new.py](https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus/blob/master/ogd/ogd_core_new.py)::_get_neural_tangents/optimizer_step；不是原AISTATS作者代码 | **部分，函数保护更直接。** Row后缓存输出Jacobian，Pair实际更新投影到其正交补。保护四类比较应使用三个独立logit差，不能只保金标logit。 | 每个全参数基向量约0.74GB，16个约11.8GB，仅基底就不适合12GB显存；还需逐锚点反向和正交化。 |

OGD与PCGrad的区别：前者保护输出Jacobian张成的空间，后者通常只看某个损失的梯度方向。AdamW的动量、预条件会改变实际位移，因此只投影原梯度不等于实际更新也满足正交性。
没有找到直接针对DiCo这三个组合的精确不变性、同时提供即插即用分类实现的论文；不能把近邻工作说成已验证的专用解法。

SMART论文[ACL2020](https://aclanthology.org/2020.acl-main.197/)同时讨论输入平滑与Bregman proximal/trust-region。但当前MT-DNN model.py核实到SmartPerturbation，未核实完整BPP入口；不能把我们做过的SiFT等同于完整SMART，也不把不完整入口列作即插即用方案。

## 7. 一致性正则

| 方法与论文 | 代码入口 | 关系与接入 | 额外成本 |
|---|---|---|---|
| **R-Drop** — [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)，NeurIPS2021 | 作者[dropreg/R-Drop](https://github.com/dropreg/R-Drop)：[huggingface_transformer_src/src/transformers/models/bert/modeling_bert.py](https://github.com/dropreg/R-Drop/blob/main/huggingface_transformer_src/src/transformers/models/bert/modeling_bert.py)::BertForSequenceClassification.bert_kl；入口bert_rdrop/run_glue.py | **部分。** 两个dropout视图一致，不等于跨阶段锚定；Row前向两遍，CE平均加四类双向KL。两个视图可以一起答错。 | 受影响步骤约两套完整前后向，非只重复head。若只加Row，按编码行预算整轮约1.54倍，非实测时间。 |
| **UDA** — [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)，NeurIPS2020 | 作者[google-research/uda/text/uda.py](https://github.com/google-research/uda/blob/master/text/uda.py)::create_model/kl_for_log_probs；text/augmentation | **部分。** 输入增强一致性；需要Train侧标签保持改写或未标注训练池。原实现为TF1 BERT，不能直接放进现有PyTorch。 | 标注B、未标注U时约B+2U前向、B+U反向，另有生成增强成本。短语中的否定、数词和修饰增删可能改变标签。 |
| **FixMatch** — [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)，NeurIPS2020 | 作者[google-research/fixmatch/fixmatch.py](https://github.com/google-research/fixmatch/blob/master/fixmatch.py)::FixMatch | **不是直接同类。** 原生图像SSL；文本版需弱/强增强、高置信伪标签和独立训练池。不能称现成BERT修复方案。 | 随未标注batch比例变化，另有文本增强成本；当前高置信错误可能被伪标签强化。 |

R-Drop/UDA不能在仅三组合输出上做一致性然后宣称修复N：三组合视图仍看不见N。即便比较四类，两份当前模型也可能共同漂移；它们不是固定Row教师。

## 8. 泛化：成熟入口与成本

| 方法与论文 | 代码入口 | 关系与接入 | 额外成本 |
|---|---|---|---|
| **SAM** — [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)，ICLR2021；NLP证据[Sharpness-Aware Minimization Improves Language Model Generalization](https://aclanthology.org/2022.acl-long.508/)，ACL2022 | 原作者google-research/sam为JAX；PyTorch社区[davda54/sam/sam.py](https://github.com/davda54/sam/blob/main/sam.py)::first_step/second_step | **不是同类。** 改善损失几何与泛化；可先只在train_rows做扰动两次前后向，Pair保留原配方。ACL2022证据为T5/mT5，不是DeBERTa直接验证。 | 受影响步骤约2倍前后向；若只加Row，整轮编码预算约1.54倍。参数备份约0.74GB，实际时间另测。 |
| **LLRD** — [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/abs/2006.05987)，2020预印本 | 作者[asappresearch/revisit-bert-finetuning/run_glue.py](https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py)::get_optimizer_grouped_parameters；sample_commands/llrd.sh | **部分。** 减慢底层表示更新，不补缺失目标。将DeBERTa embedding/12层分层设lr，output及alignment各组件独立入组。 | 基本不增加编码前后向或状态规模，是泛化类最便宜的控制。 |
| **Lookahead** — [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)，NeurIPS2019 | 作者[michaelrzhang/lookahead/lookahead_pytorch.py](https://github.com/michaelrzhang/lookahead/blob/master/lookahead_pytorch.py)::Lookahead.step | **部分。** 减振荡，慢权重也可能沿未监督方向漂移；包装现有AdamW并明确用fast还是slow权重评估。 | 无额外编码前后向；慢参数约0.74GB。旧zero_grad不接set_to_none，state_dict不完整保存slow状态，需修兼容。 |
| **Multi-sample dropout** — [Multi-Sample Dropout for Accelerated Training and Better Generalization](https://arxiv.org/abs/1905.09788)，2019预印本 | 未核实作者BERT实现；社区[lonePatient/multi-sample_dropout_pytorch/nn.py](https://github.com/lonePatient/multi-sample_dropout_pytorch/blob/master/nn.py)仅作思路参考 | **不是同类。** 一次encoder，末端dropout和共享分类层采样K次，平均每次CE；修改CLS/alignment末端特征到分类层，不是对最终logits再dropout。 | 约一次encoder加K次小head，远比R-Drop便宜；不检验encoder dropout。社区代码追加分支似把分类前feature送入CE，不能整段照搬。 |

2023的[SAM-ON](https://arxiv.org/abs/2306.04226)有作者[mueller-mp/SAM-ON/sam_on.py](https://github.com/mueller-mp/SAM-ON/blob/main/sam_on.py)，但只扰动normalization参数仍需两次完整前后向，并非近零算力；原证据主要来自视觉任务。2024的F-SAM同样没有比上述SAM更直接的DeBERTa实证，此次不额外扩充试验清单。

## 9. 推荐排序与唯一首选

1. **BLOOP：首个实际候选。** 与当前“主Row、辅Pair”阶段干扰最贴近，有NLP实验，2024方法，能复用本地双梯度代码；不新增标注、分类头或检索模块。
2. **Row锚点RecAdam式更新。** 成本低、确实对参数漂移施加约束，但锚点适配和系数需要单独验证。
3. **明确预算的primal-dual四类保护。** 对“约束”表述最直接，但不是把Pair损失换名；工程与超参数比BLOOP多。
4. **Row阶段SAM。** 专门检验泛化差距，保留原Pair以便对照；不将它宣传为零空间修复。
5. **LLRD / R-Drop。** 分别是便宜和较成熟的一致性备选。
6. **FAMO / CAGrad / Aligned-MTL / Nash-MTL。** 有价值的MTL优化对照，但当前没有证据证明任务量级平衡就是主因。
7. **仅换hinge/CRF、UDA/FixMatch或全参OGD。** 前两者若保留同候选不会改变欠约束；后几者数据/算力/移植成本不合当前条件。

### 只试一个：BLOOP的具体入口

以0.938的3290行、原模型和原推理为起点；保留完整Row阶段。第二阶段仍需在同一批上计算Row四类主损失与Pair辅助损失，N训练单例正常进入Row目标。
不能把BLOOP当成“只有Pair梯度也能保护Row”的优化器。

```text
g_row = grad(完整四类Row CE)
g_pair = grad(Pair CE)
m = EMA(g_row)
g_pair_projected = g_pair - dot(g_pair,m)/(dot(m,m)+eps) * m
g = g_row + lambda * g_pair_projected
```

接入本地`row_priority_gradients.py`时须改变：

1. 保存跨步Row梯度EMA；它不是模型参数EMA/SWA。
2. 始终按EMA投影，不是仅内积负时投影。
3. `assign_gradients`必须真正减去系数乘EMA张量，不能只换投影系数后仍减当前Row梯度。
4. 当前`separate_gradients`已经给Pair乘0.5。若复用它，不能再乘一次0.5；或改取未加权Pair梯度后只在最终组合乘一次。
5. 保持训练数据、选模和推理口径明确，先与同代码mixed比较，再与历史0.938比较；不把不同epoch成绩拼接。

该方法保护的是局部主梯度方向，不是全部输出Jacobian。AdamW预条件与动量会改变实际位移，所以没有“每步Row loss必降”或“Dev F1必升”的保证。
推荐依据是机制相关性、NLP证据和本地接入成本，不是已经证实能达到0.95。

## 10. 本地证据入口

- 原0.9248：`artifacts/original_error_analysis/summary.csv`、`error_counts.csv`。
- 0.938配置：`experiments/deberta_manual_aug/outputs/run06_relation48_staged/run_config.json`及`error_analysis/REPORT.md`。
- PCGrad对照：`outputs/run13_gradient_mixed/best_metrics.json`、`outputs/run13_gradient_project/best_metrics.json`。
- KL保持：`experiments/deberta_row_anchor/outputs/run01_keep1_t2/final_comparison.csv`。
- 最新Gate：`experiments/deberta_manual_aug/outputs/run24_pair_gate_seed42/PAIR_GATE_ANALYSIS.md`。

这些方法都不能补回输入中缺失的新闻背景，也不能自动解决标注口径冲突。优化稳定、输出可辨识和语义泛化应分别报告。
