# DiCo-NLI多目标训练：方法检索与实施入口

检索日期：2026-10-25。检索arXiv、ACL Anthology、会议官方论文页、GitHub作者代码；Papers with Code部分页面无法访问，不能视为已核验其当前条目。第三方实现和未核实的入口在表中明确标注。本次没有修改训练代码或运行训练。

## 1. 先界定问题

当前Pair分数为S_FB=z_0F+z_1B、S_BF=z_0B+z_1F、S_EE=z_0E+z_1E。它不涉及N；同时，对所有行施加z_F+=c、z_B-=c，三个分数都不变，但单行argmax可以改变。这是从当前代码直接推导出的输出不可识别性，不是某篇论文已在DiCo上验证的结论。

不变性说明Pair目标不能识别这些变化，不说明梯度下降必然朝某个退化方向移动。共享编码器、AdamW动量、预条件和权重衰减会影响实际更新。必须区分：梯度冲突、Pair目标未约束的方向，以及Train→Dev泛化差距。

本地最新证据：参数锚定run06最佳独立/结构F1为0.675109/0.908480；其Train上881条F中465条仍被判N。原Lexical对照为0.468191/0.930569。这不是仅在Dev上出现的过拟合。

下面“是/部分/不是”指与上述机制的匹配程度，不是论文质量。计算倍数是相同输入量下普通一次前向反向更新的粗估，不是本机实测，也不等于整轮训练必然变慢相同比例。

## 2. 最优先：有主次的约束优化

| 方法、论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **Cooper: A Library for Constrained Optimization in Deep Learning**，2025，[论文](https://arxiv.org/abs/2504.01212) | 作者[cooper-org/cooper](https://github.com/cooper-org/cooper)；`src/cooper/optim/constrained_optimizers/alternating_optimizer.py`中的`AlternatingDualPrimalOptimizer.roll`；`src/cooper/constraints/constraint.py`；`src/cooper/formulations/formulations.py`。已核验[官方源码文档](https://cooper.readthedocs.io/en/latest/_modules/cooper/optim/constrained_optimizers/alternating_optimizer.html)。 | **是，需自行定义Row保护约束**。Pair为优化目标，四类Row损失/间隔分别设退化预算，乘子依据预算违反程度更新。Cooper本身不是DiCo算法，也不自动知道保护什么。 | 乘子开销很小；共享logits时接近1×；另加保护Row批次约1.4–2×，再做实际步后检查会更高。 |
| **A Primal-Dual Formulation for Deep Learning with Constraints**，NeurIPS 2019，[论文](https://proceedings.neurips.cc/paper/2019/file/cf708fc1decf0337aded484f8f4519ae-Paper.pdf) | 作者[dair-iitd/dl-with-constraints](https://github.com/dair-iitd/dl-with-constraints)；[`ner/code/gan_trainer_hm.py`](https://github.com/dair-iitd/dl-with-constraints/blob/master/ner/code/gan_trainer_hm.py)，`lambda_update`；[`ner/code/mtl_constraints.py`](https://github.com/dair-iitd/dl-with-constraints/blob/master/ner/code/mtl_constraints.py)；`ner/code/dual_training.py`。README中的旧`gan_trainer.py`不是核实后的文件名。 | **部分**。NLP的NER/SRL/实体类型约束训练先例；原约束不是保护Row分类器。借鉴原始—对偶更新逻辑，工程上优先用现代Cooper，避免搬旧AllenNLP工程。 | 约束计算若复用表示接近1×；独立重放/评估增加相应模型计算。 |
| **Careful with that Scalpel: Improving Gradient Surgery with an EMA**（BLOOP），ICML 2024，[作者论文页](https://machinelearning.apple.com/research/careful-with-that-scalpel) | 作者[apple-aiml-research/ml-bloop](https://github.com/apple-aiml-research/ml-bloop)；[`src/bloop/optim.py`](https://github.com/apple-aiml-research/ml-bloop/blob/main/src/bloop/optim.py)，`bloop_direction`。原版为JAX。 | **部分**。明确主任务Row、辅助Pair，以Row梯度EMA投影辅助梯度。保护的是一阶主损失方向，不是每条样本的分类间隔。**项目已试，不再作为未试首选。** | 两套梯度及EMA，通常1.5–2×，额外梯度存储。 |

### 推荐的约束适配

令theta_0为本轮Row结束后的参考，R_k为仅来自Train的固定保护样本，k为F/B/E/N：

```text
minimize    L_pair(theta)
subject to  g_k(theta) = L_row,k(theta; R_k)
                         - L_row,k(theta_0; R_k) - epsilon_k <= 0

lambda_k <- max(0, lambda_k + dual_lr * g_k)
Lagrangian = L_pair + sum_k(lambda_k * g_k)
```

四类分开保护，避免总Row平均损失掩盖某一类退化。还可用正确类别相对竞争类别的margin损失作为预算；这是本项目的适配建议，不能写成论文原结论。保护参照和预算只由Train确定，Dev仍用于评估与选模。

与固定`Row + 0.5*Pair`的区别是明确退化预算，并由违反程度调整乘子；仍需计算Row监督，不能宣称无需重放。与刚失败的参数锚定区别是约束输出损失，而非全部参数的欧氏距离。

随机原始—对偶优化不保证每个有限AdamW更新都满足预算。若研究目标要求“每个接受的步都满足”，需要对实际候选参数做步后检查/回退并正确恢复优化器状态；保护范围也仅限被测Train样本。预算满足不保证Dev F1提升。

## 3. 梯度冲突与动态权重：2023–2025及经典方法

| 方法、论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks**，2024预印本／ICLR 2025，[论文](https://arxiv.org/abs/2408.11104) | 作者[tum-pbs/ConFIG](https://github.com/tum-pbs/ConFIG)；[`conflictfree/grad_operator.py`](https://github.com/tum-pbs/ConFIG/blob/main/conflictfree/grad_operator.py)，`ConFIG_update_double`。 | **部分**。构造与各损失梯度正向对齐的方向，默认无Row优先级。分别计算Row/Pair梯度，合并后更新。主要实证来自PINN，不能照搬为NLI涨分证据。 | 两次反传、两套梯度，约1.5–2×；两任务专用公式不需大矩阵求逆。 |
| **FAMO: Fast Adaptive Multitask Optimization**，NeurIPS 2023，[论文](https://arxiv.org/abs/2306.03792) | 作者[Cranial-XIX/FAMO](https://github.com/Cranial-XIX/FAMO)；[`famo.py`](https://github.com/Cranial-XIX/FAMO/blob/main/famo.py)，`get_weighted_loss`、`backward`、`update`。 | **部分**。按相对损失改进调权，没有硬Row保护。替换固定权重聚合，并按作者逻辑在模型更新后重新测loss再更新任务权重。 | 一次反传；官方例子另有更新后的无梯度前向，约1.2–1.4×。不是无条件零开销。 |
| **Independent Component Alignment for Multi-Task Learning**（Aligned-MTL），CVPR 2023，[论文](https://arxiv.org/abs/2305.19000) | 作者SamsungLabs/MTL仓库当前无法访问。已核验第三方[LibMTL/weighting/Aligned_MTL.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/Aligned_MTL.py)；[TorchJD说明](https://torchjd.org/stable/docs/aggregation/aligned_mtl/)也说明作者仓库不可用。 | **部分**。对梯度Gram矩阵特征分解后对齐任务分量。接入梯度合并处。任务偏好不等于约束Row不退化；LibMTL所查版本有旧`torch.symeig`调用，需适配。 | 两套梯度，约1.5–2×；两任务只分解2×2矩阵。 |
| **Jacobian Descent for Multi-Objective Optimization**（UPGrad），2024预印本，后续持续更新，[论文](https://arxiv.org/abs/2406.16232) | 作者[SimplexLab/TorchJD](https://github.com/SimplexLab/TorchJD)；[`src/torchjd/aggregation/_upgrad.py`](https://github.com/SimplexLab/TorchJD/blob/main/src/torchjd/aggregation/_upgrad.py)。 | **部分**。投影各梯度到公共对偶锥。**恰好两损失、等权聚合时，与对称PCGrad等价到求和/平均的尺度差，不适合作为本项目新的独立方向。** | 两套梯度和投影，约1.5–2×。 |
| **Gradient Surgery for Multi-Task Learning**（PCGrad），NeurIPS 2020，[论文](https://arxiv.org/abs/2001.06782) | 作者[tianheyu927/PCGrad](https://github.com/tianheyu927/PCGrad)；[`PCGrad_tf.py`](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)，原版TensorFlow。 | **部分**。去除相互冲突梯度分量。改`backward`为逐任务梯度、投影、写回`.grad`。**本项目已做过相关实验，不列为新方案。** | 两套梯度，约1.5–2×。 |
| **Conflict-Averse Gradient Descent for Multi-task Learning**（CAGrad），NeurIPS 2021，[论文](https://arxiv.org/abs/2110.14048) | 作者[Cranial-XIX/CAGrad](https://github.com/Cranial-XIX/CAGrad)；[`toy.py`](https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py)，`cagrad`包含两任务求解。 | **部分**。平均目标与最差局部任务改进之间折中；不明确保护Row。替换梯度聚合，保留两个任务监督。 | 两套梯度＋小型标量优化，约1.5–2×，另有CPU求解开销。 |
| **Multi-Task Learning as a Bargaining Game**（Nash-MTL），ICML 2022，[论文](https://arxiv.org/abs/2202.01017) | 作者[AvivNavon/nash-mtl](https://github.com/AvivNavon/nash-mtl)；[`methods/weight_methods.py`](https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py)，`NashMTL`。 | **部分**。任务公平协商，不等于Row优先；需逐任务梯度并解权重子问题。 | 作者实现权重更新时可有两次任务梯度＋一次加权反传，约2–3×，另有CVXPY/ECOS求解；可降低权重更新频率。 |
| **GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**，ICML 2018，[论文](https://arxiv.org/abs/1711.02257) | 未核实作者仓库；已核验第三方[`LibMTL/weighting/GradNorm.py`](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/GradNorm.py)。 | **不是直接同类**。平衡梯度幅度与相对训练速度，不识别Pair不变性，也不处理所有梯度方向冲突。在共享层测各任务梯度范数，学习任务权重。 | 约1.3–2×，取决于测哪些共享层；还有任务权重更新。 |
| **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**，CVPR 2018，[论文](https://arxiv.org/abs/1705.07115) | 作者[yaringal/multi-task-learning-example](https://github.com/yaringal/multi-task-learning-example)；[`multi-task-learning-example-pytorch.ipynb`](https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb)。 | **不是直接同类**。学习损失标量权重；不保护方向与Row判别。分类任务须按论文对应形式实现，不能直接照搬回归示例的系数。 | 一次反传，约1×。 |

共同边界：`g_Row·d>=0`最多是小步SGD条件下当前批次平均Row损失的一阶结论；不保证逐样本argmax或分类间隔。AdamW动量与预条件会使实际更新不同于`-lr*d`。仅记录投影后cosine不能证明实际参数更新保护了Row。

## 4. 结构化预测

| 论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **Torch-Struct: Deep Structured Prediction Library**，ACL 2020，[论文](https://aclanthology.org/2020.acl-demos.38/) | 作者[harvardnlp/pytorch-struct](https://github.com/harvardnlp/pytorch-struct)；[`torch_struct/linearchain.py`](https://github.com/harvardnlp/pytorch-struct/blob/master/torch_struct/linearchain.py)，`torch_struct/distributions.py`；BERT例子`notebooks/BertTagger.ipynb`。 | **不是，在相同三候选下**。可把DeBERTa分数作为CRF势函数，但FB/BF/EE的归一化概率已经是一个长度2受约束CRF的特例。 | 长度2结构层计算近1×，无额外编码器前向。 |
| **Portuguese Named Entity Recognition using BERT-CRF**，2019预印本，[论文](https://arxiv.org/abs/1909.10649) | 工程参考为独立实现[allanj/pytorch_neural_crf](https://github.com/allanj/pytorch_neural_crf)，不是该论文作者代码；[`src/model/module/linear_crf_inferencer.py`](https://github.com/allanj/pytorch_neural_crf/blob/master/src/model/module/linear_crf_inferencer.py)，`transformers_trainer.py`。 | **不是直接同类**。提供BERT接CRF、转移掩码的成熟范例，但无法补充单行N校准监督。 | 长度2的CRF额外计算很小。 |

### 为什么换CRF或structured hinge仍不够

若仍只用FB/BF/EE，且S(a,b)=z_0a+z_1b+T_ab，F+=c、B-=c仍不改变任何候选分数。CRF负对数似然和`max_y(S_y+Delta_y)-S_gold`形式的structured hinge都保留这个不变性，学习转移参数也不能使N自动参与。这是当前目标的数学推导。

若扩大到所有16组合且没有交互势，联合CE恰好分解成两行Row CE。它可以改变监督，但不能包装为一种全新的联合模型。真正引入输入相关交互势会改变架构，必须单独验证，而不是宣称换个损失名称就解决问题。

## 5. 输出保护、零空间与参数锚定

| 方法、论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **Orthogonal Gradient Descent for Continual Learning**，AISTATS 2020，[论文](https://proceedings.mlr.press/v108/farajtabar20a.html)；代码对应扩展论文**Generalisation Guarantees for Continual Learning with Orthogonal Gradient Descent**，2020，[论文](https://arxiv.org/abs/2006.11942) | 扩展论文作者[MehdiAbbanaBennani/continual-learning-ogdplus](https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus)；[`ogd/ogd_core_new.py`](https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus/blob/master/ogd/ogd_core_new.py)。不冒充原OGD作者仓库。 | **部分，输出保护机制高度相关**。按参考样本输出Jacobian构造保护子空间，Pair更新投影后尽量保持原输出。区别于仅与一个平均Row梯度正交；仍有局部近似与基底陈旧问题。 | 多次输出梯度、基底存储与投影。1.84亿参数的一个FP32向量约0.736GB，多基底全参数版不适合12GB直接照搬。 |
| **Learning without Forgetting**，ECCV 2016，[论文](https://arxiv.org/abs/1606.09282) | 作者[lizhitwo/LearningWithoutForgetting](https://github.com/lizhitwo/LearningWithoutForgetting)是MatConvNet；其README推荐的PyTorch工程[arunmallya/packnet，src/lwf.py](https://github.com/arunmallya/packnet/blob/master/src/lwf.py)。 | **部分**。固定Row参考输出，Pair时蒸馏完整四类分布。项目已有输出KL保持实验，不能称为未试新方案；若参考本身错，也会保留错误。 | 缓存参考可省teacher前向；在线teacher约1.2–1.5×并需保存参考模型。 |
| **Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting**（RecAdam），EMNLP 2020，[论文](https://aclanthology.org/2020.emnlp-main.634/) | 作者[Sanyuan-Chen/RecAdam，RecAdam.py](https://github.com/Sanyuan-Chen/RecAdam/blob/master/RecAdam.py)。 | **部分**。锚定参数并做目标权重退火。刚跑的run06借鉴参数保持，但用Row周期参考和近端回拉，不是原论文复现；不能把该配置失败表述为原RecAdam已被严格否定。 | 前后向近1×，增加参考参数与逐参数运算。 |

没有检索到直接分析“DiCo三组合损失引起F/N单行偏移”的现成论文。最接近的是主辅优化、持续学习输出保护和约束优化；不能把这些文献说成已经证明本任务的具体因果链。

## 6. 一致性正则

| 方法、论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **R-Drop: Regularized Dropout for Neural Networks**，NeurIPS 2021，[论文](https://arxiv.org/abs/2106.14448) | 作者[dropreg/R-Drop，README Usage](https://github.com/dropreg/R-Drop/blob/main/README.md#usage)，`compute_kl_loss`；`huggingface_transformer_src/bert_rdrop/run_glue.py`为BERT入口。不要把未找到KL的`modeling_bert.py`作为关键实现。 | **部分**。两次dropout输出做对称KL，保留监督CE；若两次都稳定答错N，KL仍可接近0。不是固定Row参考保护。 | 约2×前后向。 |
| **Unsupervised Data Augmentation for Consistency Training**，2019预印本／NeurIPS 2020，[论文](https://arxiv.org/abs/1904.12848) | 作者[google-research/uda，text/uda.py](https://github.com/google-research/uda/blob/master/text/uda.py)，`create_model`、`kl_for_log_probs`，有实际BERT调用。 | **部分**。原文本与增强文本分布一致性；当前模型的stop-gradient目标不等于固定Row参照。NLI增强必须保持文本对的关系。 | 无标签原文前向＋增强分支训练；1:1无标签比例通常约2–3×，另有文本增强成本。 |
| **FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence**，NeurIPS 2020，[论文](https://arxiv.org/abs/2001.07685) | 原作者[google-research/fixmatch，fixmatch.py](https://github.com/google-research/fixmatch/blob/master/fixmatch.py)是图像实现；BERT适配参考[Microsoft SemiLearn，semilearn/algorithms/fixmatch/fixmatch.py](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/fixmatch/fixmatch.py)及`semilearn/nets/bert/bert.py`。 | **不是直接同类**。弱增强高置信预测给强增强做伪标签；高置信N偏移可能被强化。不能把置信阈值视为标签正确性。 | 弱分支前向＋强分支训练，取决于无标签比例，通常明显增加；1:1比例粗估2–3×。 |

## 7. 泛化与BERT微调稳定性

| 方法、论文、年份 | 仓库和关键文件 | 是否同类；如何接入 | 额外成本 |
| --- | --- | --- | --- |
| **Sharpness-Aware Minimization for Efficiently Improving Generalization**，ICLR 2021，[论文](https://arxiv.org/abs/2010.01412) | 作者[google-research/sam，sam_jax/training_utils/flax_training.py](https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py)；PyTorch第三方[davda54/sam，sam.py](https://github.com/davda54/sam/blob/main/sam.py)。 | **不是直接同类**。优化局部最坏损失，Pair看不见的方向仍看不见。项目已试Row SAM和Row+Pair SAM，不再作为新首选。 | 约2×前后向。 |
| **Lookahead Optimizer: k steps forward, 1 step back**，NeurIPS 2019，[论文](https://arxiv.org/abs/1907.08610) | 作者[michaelrzhang/lookahead，lookahead_pytorch.py](https://github.com/michaelrzhang/lookahead/blob/master/lookahead_pytorch.py)。 | **不是直接同类**。包装AdamW，slow参数平滑轨迹；不能补充Pair缺失的识别信息。 | 前后向约1×；多一份slow权重及周期插值。 |
| **How to Fine-Tune BERT for Text Classification?**（逐层学习率衰减参考），2019，[论文](https://arxiv.org/abs/1905.05583) | 作者[xuyige/BERT4doc-Classification](https://github.com/xuyige/BERT4doc-Classification) README列出discriminative fine-tuning；已核验现代BERT实现[Microsoft SemiLearn，semilearn/core/utils/build.py](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/core/utils/build.py)，`get_optimizer(layer_decay=...)`及`bert.py`的`group_matcher`。 | **部分**。按层设LR，底层小、上层大，减少表示扰动；仍不能约束分类头的输出不变性。只修改AdamW参数分组。 | 约1×。 |
| **Multi-Sample Dropout for Accelerated Training and Better Generalization**，2019，[论文](https://arxiv.org/abs/1905.09788) | 未核验到可直接推荐的成熟BERT作者实现。查到第三方[lonePatient/multi-sample_dropout_pytorch，nn.py](https://github.com/lonePatient/multi-sample_dropout_pytorch/blob/master/nn.py)，但属于CNN，且所查循环的loss输入有可疑之处，不建议直接复制。 | **不是直接同类**。一次编码，多次dropout分类，平均各次CE。平均logits后算一次CE不等价于平均各次CE。实现简单，但不能称为已有经核验BERT入口。 | 若仅重复轻量head，通常约1.0–1.1×。 |
| **On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines**，ICLR 2021，[论文](https://arxiv.org/abs/2006.04884) | 作者[uds-lsv/bert-stable-fine-tuning，examples/bert_stable_fine_tuning/adamw.py](https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/examples/bert_stable_fine_tuning/adamw.py)。 | **不是直接同类**。微调优化失败、梯度与seed不稳定的诊断文献。检查bias correction、warmup、LR；不把Train高Dev低自动归因于参数过大。 | 每步约1×；增加步数会增加总时间。 |

## 8. 最终排序和首轮实施范围

1. **受约束Pair更新：Cooper框架＋按类Row损失/间隔预算。** 它可直接表达“组合要变好，单行不能任意退化”，是机制与实施入口最贴近的一项。不是保证0.95的已有DiCo配方。
2. **ConFIG。** 如果要继续一个不同的梯度合并对照，双任务实现清晰；仍需检测实际AdamW更新，不能只看梯度夹角。
3. **FAMO。** 更省资源的动态权重对照；适合预算有限时验证固定权重是否失衡，不解决可识别性本身。
4. **R-Drop。** 仅在阶段性漂移已受控制后研究随机稳定性。
5. **LLRD。** 低成本泛化/稳定性对照，不与首轮其他改动同时加入。

首轮若实施第一项，应从**原官方3042行＋Lexical配方重新开始**，与0.930569的匹配对照比较。保留编码器、Token Alignment、词汇分支、类别权重、预测规则；不添加人工数据，不同时叠加SAM/SWA。从run06的0.908模型续训会混入已有训练路径的影响。

同时记录：四类约束违反量、四类单行正确数、非N→N、Row前后独立F1、pair-aware F1。先验证“保护约束是否做到”，再验证“Dev成绩是否上升”。不能用Train预算满足替代泛化证据，也不能只凭独立分上升宣称最终任务改善。

这是相关性和实验价值排序，不是理论涨幅排序。现有文献没有提供本任务达到0.95的保证。

## 9. 后续手算核验修正

进一步对现有分数计算后，按类平均CE保护仍有明确漏洞：22次类别正确数退化中有4次该类平均CE反而下降。因此不能把上面的平均CE预算版本直接当成足够的保护。更贴近本问题的候选是保护Row结束时Train上已判对样本的正分类margin，且不锁住原本错误的行。

详细反例、870批真实梯度日志统计、当前Pair平滑目标的解析反例，以及已跑过的关闭平滑负对照，见[适用性核算](outputs/run06_row_parameter_anchor/METHOD_APPLICABILITY.md)。这些核算用于排除不适合的方法，不提供Dev涨分概率。
