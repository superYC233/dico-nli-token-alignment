# 当前结果与下一步研究选择

核查日期：2026-10-25。本次读取已有实验与作者论文、实现；没有训练，没有修改模型、数据或训练配置。

## 1. 最新原始 base 对照说明什么

以下为同一官方 Train 3042 / Dev 660、Lexical 与原先分阶段 Row/Pair 配方，均在各自结构 F1 最佳的一轮上比较。

| 初始化 | 最佳轮次 | independent W-F1 | pair-aware W-F1 | F / B / E 正确 | N 正确 |
|---|---:|---:|---:|---|---:|
| 旧 NLI DeBERTa-v3-base | 3 | 0.468191 | 0.930569 | 179 / 179 / 158 | 98 |
| 原始 microsoft/deberta-v3-base | 3 | 0.496812 | 0.910326 | 178 / 178 / 144 | 101 |

原始 base 的结构 F1 下降 2.0244 个百分点：N 多对 3 行，E 少对 14 行，F/B 各少对 1 行。其独立 F1 最高的第 5 轮为 0.829140，该轮结构 F1 为 0.861564；不是两项同时提高。

这次实验不支持“MNLI/ANLI 初始化是当前低分的主要原因”。它也不能证明原始 base 在重新调参后一定更差：当前使用的是已有 NLI 模型的配方，且只有一个种子。明确成立的是：去掉 NLI 初始化没有解除现有分阶段训练的退化，反而降低了当前配方的 E/FBE 表现。

数值来源：

- experiments/deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/best_summary.csv
- experiments/deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/epoch_summary.csv
- experiments/deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/best_summary.csv

## 2. 当前至少有三个不同问题

### 2.1 训练目标留有自由方向

三组合 Pair CE 的候选只含 FB、BF、EE。给所有行的 F/B/E 同时加同一个常数，而保持 N 不动，三个组合分数整体平移，Pair CE 不变；给 F 加 c、B 减 c，也不改变任何候选组合分数。

这说明该目标无法约束部分绝对行分数。它不证明梯度必然沿某个方向漂移；实际方向还受共享参数、优化器、正则化和训练顺序影响。加入一个新编码器或比较层，也不会自动移除这项损失的数学性质。

因此，结构分高和独立分低可以同时出现；不能把这一现象全部归因于模型不认识词，或全部归因于训练数据太少。

### 2.2 学到人工模板，未充分迁移到真实片段

新增 family768 后，保留家族的 192 行人工探针 independent 从旧模型的 0.720455 提高到 0.979209；但同预算 copy 对照的 Dev pair-aware 为 0.910746，manual 为 0.908421。

这支持模型学到了这批模板分布中的区别，没有证明这些区别能提高自然 Dev 的结构成绩。不能把“再加 200 行”按固定斜率外推成“再涨 1 分”。

当前破坏的 E 例子包含 by judge / issued by a judge、for granting asylum / for asylum、of grassy field / in a field of green grass。这些涉及省略、事件角色、修饰和词义解释；不是单纯再增加规则名词短语就能覆盖。

详细已有分析：experiments/deberta_family_aug/outputs/run01_family768_manual/analysis_vs_0p938.md。

### 2.3 局部比较后缺少显式组合层

当前 experiments/deberta_lexical/model.py 的结构是：

    DeBERTa 上下文编码
        → 128 维投影
        → token soft alignment
        → [x, aligned, x-aligned, x*aligned] 比较
        → 每个文本直接 mean/max pooling
        → 分类；加上 CLS 与 Lexical 分支

DeBERTa 本身已经编码上下文和顺序，因此不能说整套模型完全忽略词序。具体可检验的缺口是：比较结果生成后，头部没有再显式组合相邻的差异证据。

例如同一个修饰词限定哪个对象、两个并列项各自有什么限制，需要一起解释多个位置。现有直接汇总可能不够理想，这是研究假设，不是已经证明的根因。

## 3. 架构候选：按接入优先级排序

### 第一：ESIM 式比较后组合，使用 RE2 的轻量 CNN 实现思路

- 论文：Enhanced LSTM for Natural Language Inference，ACL 2017。
- 论文链接：https://aclanthology.org/P17-1152/
- 作者代码：https://github.com/lukecq1231/nli
- 相关轻量实现论文：Simple and Effective Text Matching with Richer Alignment Features，ACL 2019。
- RE2 论文：https://aclanthology.org/P19-1465/
- 作者 PyTorch 实现：https://github.com/alibaba-edu/simple-effective-text-matching-pytorch
- 关键文件：src/network.py、src/modules/encoder.py、src/modules/fusion.py、src/modules/connection.py。

可直接借鉴的改法：把现有 compare_layer 的输出分别按 A/B mask 隔开，经过共享的两层 128→128、kernel=3 的 Conv1d，再做现有 mean/max pooling。每层都清除无效位置，避免 padding 激活进入下一层。CLS 与 Lexical 分支保留。

两层带 bias 卷积共 98,560 参数，无额外 DeBERTa 前向，无新增标注。额外成本主要是短序列上的小卷积；本次未实测训练速度或显存。

它与完整 RE2 不同：完整 RE2 在块间进行 CNN 编码→对齐→融合→再次编码/对齐，并保留原始及历史特征。本方案借 ESIM 的“比较后组合”位置和 RE2 的轻量编码方式，不应宣称完整复现 RE2。

优先原因：对当前代码有明确接入位置，检验的是相邻比较证据的组合，区别于已试过的词汇全局特征、NULL 匹配、WordNet/KIM 和最终四分类联合头。局限是两层 k=3 只提供局部感受野，不能自动解决长距离推理、缺语境和 Pair 目标的自由方向。

### 第二：DIIN 的二维交互空间

- 论文：Natural Language Inference over Interaction Space，ICLR 2018。
- 论文：https://openreview.net/forum?id=r1dHXnH6-
- 作者代码：https://github.com/YichenGong/Densely-Interactive-Inference-Network
- 关键文件：python/models/DIIN.py，bi_attention_mx、dense_net、dense_net_block。

区别在于保留 A 中每个 token 与 B 中每个 token 的交互网格，再对二维邻域做卷积，而不是先把对面所有 token 压缩成一个对齐向量。它有机会组合连续短语之间的匹配与差异。

接入可复用现有 DeBERTa 与 128 维投影，用小型二维卷积分支替换 alignment 的汇总路径。原作者实现是旧 TensorFlow，需要移植；简化版本必须称为 DIIN-inspired，而不能套用论文完整模型成绩。

按 B=16、两侧各 20 token、128 通道估算，交互张量本身 FP32 约 3.28 MB；完整训练还包含卷积激活、梯度和主干，不能据此承诺整机显存。一次主干前向即可，工程改动比第一项大。

### 第三：Neural Natural Logic 的局部关系投影与组合

- 论文：Exploring End-to-End Differentiable Natural Logic Modeling，COLING 2020。
- 论文：https://aclanthology.org/2020.coling-main.101/
- 作者代码：https://github.com/feng-yufei/Neural-Natural-Logic
- 关键入口：train_aligned.py。
- 关键模型：model/nnl/nnl_aligned_modular_final2.py；关系模块 model/nnl/relnn_aligned_modular_final2.py。

该路线预测局部自然逻辑关系，结合 polarity/projectivity 投影，再组合为整体判断，研究目标最接近“哪处差异改变蕴含”。但七种自然逻辑关系不等于 DiCo 四类，需要适配预处理、投影和读出。仅 3042 行训练局部隐关系可能产生看似合理的伪解释，暂不作为第一轮改造。

## 4. 预训练模型：有针对性的候选是 PEARL，但不优先于结构检验

PEARL：Learning High-Quality and General-Purpose Phrase Representations，Findings of EACL 2024。

- 论文：https://aclanthology.org/2024.findings-eacl.66/
- 作者代码：https://github.com/tigerchen52/PEARL
- 关键文件：source/model.py、source/augmentation.py、source/loss.py。
- 可下载权重：https://huggingface.co/Lihuchen/pearl_base

其训练针对无上下文短语，加入短语类型、形态信息和不同粒度增强；作者公开模型卡的 PPDB-filtered 为 72.2，E5-base 为 65.6。原论文表格报 72.7±0.4，不能把这两个版本的数混为同一下载权重结果。

这些是短语相似、复述和实体匹配证据，不是 F/B/E/N 方向分类证据。若尝试，应保留有序双文本输入与四分类头，用 PEARL 作为初始化；单个余弦相似度无法区分 F/B，因为余弦对交换顺序对称。

另一个重要边界：PEARL 基于 E5，其监督训练包含 NLI 等数据；Phrase-BERT 官方训练命令也用 bert-base-nli-stsb-mean-tokens 初始化。它们都不适合充当“完全没有 NLI 训练历史”的对照。

证据：

- PEARL 作者模型卡：https://huggingface.co/Lihuchen/pearl_base
- E5 论文：https://arxiv.org/abs/2212.03533
- Phrase-BERT 论文：https://aclanthology.org/2021.emnlp-main.846/
- Phrase-BERT 作者实现：https://github.com/sf-wa-326/phrase-bert-topic-model

SpanBERT、ModernBERT、Ettin 都有公开模型和实现，但本轮没有找到它们在孤立短语方向关系上优于现有 NLI DeBERTa 的直接证据。SpanBERT 在 Phrase-BERT 论文的短语评测中也未展示稳定优势。新架构、长上下文或更大的通用预训练成绩，不足以推出本任务更好。

## 5. 训练/诊断数据：两类更贴近当前错误

### PLANE：修饰语与上下位组合

- 论文：Testing Large Language Models on Compositionality and Inference with Phrase-Level Adjective-Noun Entailment，COLING 2022。
- 论文：https://aclanthology.org/2022.coling-1.359/
- 作者代码：https://github.com/lorenzoscottb/PLANE
- 关键目录：supervised_learning/One-IT/、supervised_learning/OoD/。
- 作者公开数据：https://huggingface.co/datasets/lorenzoscottb/PLANE-ood

它专门检验形容词—名词短语的必要蕴含、上下位替换及组合泛化，比通用整句 NLI 更接近修饰范围和 F/E 边界。作者数据来自自然短语与 WordNet 规则，不应称所有行都经人工金标。

优先用作独立诊断，再决定是否少量迁移。它仍有不同的必要蕴含口径：例如 small mouse→small animal 不被视为必然成立，因为形容词的比较标准可能改变。与 DiCo 的具体标注约定必须逐类核验。

其 1/0 仅标一个方向是否蕴含。正例不能一律标 F，负例不能一律标 N。若只知道 T1 蕴含 T2，四类允许集合是 {F,E}；若只知道该方向不蕴含，允许集合是 {B,N}。可考虑部分标签辅助目标：

    正向蕴含：     -log[p(F) + p(E)]
    正向不蕴含：   -log[p(B) + p(N)]

这只监督已知方向，保留未知反向；是否对 DiCo 有效仍需实验。交换文本也不会自动补出未知方向的真值。

### SherLIiC：动作、事件与论元类型

- 论文：SherLIiC: A Typed Event-Focused Lexical Inference Benchmark for Evaluating Natural Language Inference，ACL 2019。
- 论文：https://aclanthology.org/P19-1086/
- 作者数据/代码：https://github.com/mnschmit/SherLIiC
- 后续预训练模型实现：Language Models for Lexical Inference in Context，EACL 2021。
- 论文：https://aclanthology.org/2021.eacl-main.108/
- 实现：https://github.com/mnschmit/lm-lexical-inference
- 数据路径：data/sherliic/train.csv、dev.csv、test.csv；训练入口 src/train/nli_model.py。

原论文包含 3985 条人工标注推理规则候选，关注具有论元类型的事件关系。它更适合检验“事件相关/先后发生，不等于必然蕴含”，也强调同一个动词在不同论元类型下推理不同。

原始数据是带类型的依存关系路径，不能直接当作普通短语 CSV；后续作者有文本化和训练实现。仍属单向蕴含监督，必须保留论元类型，且不能把所有非蕴含负例硬改为 DiCo 的 N。当前尚未对该数据执行项目级重合审计，因此还不是可直接混入的训练池。

## 6. 已实际审计一个看似直接可用的资源：Pavlick 自然逻辑关系标注

作者资源：https://cs.brown.edu/people/epavlick/resources/natlog-labeled-rte-pairs.gz

这不是整库 PPDB 的预测标签文件。作者现存文件有 23,179 行，包含 t、h、majority_label、num_labels、distribution，版本规模与早期论文数字不同。排除 NA/none 后为 22,976 条；其中 3005 行有最高票并列，不能全部当高置信金标。

本次只读、内存审计结果：

| 检查 | 结果 |
|---|---:|
| 与官方 Train 相同的无序文本对 | 21 |
| 与官方 Dev 相同的无序文本对 | 5 |
| 与 Train 共有的完整短语 | 237 |
| 与 Dev 共有的完整短语 | 74 |
| 按新增词形及 Porter 词干双零 Dev 交集筛选后 | 5918 |
| 上述筛选后，两端均为多词短语 | 85 |
| 再要求最高票比例≥0.8、无并列 | 2619 |
| 高一致度子集，两端均为多词短语 | 38 |

5 个 Dev 重复对包括 a child/a man、a man/a person、a man/a young man、are playing/play、couple/two people。

严格筛选后的 5918 行初步关系映射计数为 F449/B581/E321/N4567，以单词和 N 为主；高一致度 2619 行也有 N2247。它不能直接补成数千条同域自然片段。

来源需要分清：已确认的任务关系是 DiCo←PhrasIS←iSTS，不能说 PPDB 是 PhrasIS 的直接祖先。Pavlick 2015 的人标来自 PPDB∩SICK，后续博士论文还包含 PPDB∩RTE2，并涉及这些来源的多个原始分区。现存合并文件没有 source/split 字段，排除精确重复后也不能直接声称来源完全独立。

- 原论文：https://aclanthology.org/P15-1146/
- 作者数据页：https://cs.brown.edu/people/epavlick/data.html
- 来源说明见博士论文第 47–50 页：https://cs.brown.edu/people/epavlick/papers/ellie-thesis.pdf

所有外部训练样本继续遵守用户要求的新增词形、词干与 Dev 零交集，包括功能词与数字。本轮未放宽。普通词汇重合与样本/标签泄漏是不同检查；这一额外隔离要求会大量过滤自然语言短语，因此资源总规模不等于实际可用规模。

## 7. 现在具体做什么

建议先做一个明确结构改造：**NLI DeBERTa + 原 Lexical + 比较后共享 CNN composition**。

起点使用官方 3042 行的 0.930569 Lexical 配方，与当前原始 base 的初始化对照区分开。保留原分阶段训练、类别权重和选模规则，首次只替换 compare→pool 之间的计算。这一轮的目的，是检验比较证据组合是否改善自然片段；不是又一次声称修改辅助权重就能修复所有退化。

每个已有评估点导出同一权重的：独立 F1、结构 F1、F/B/E/N 正确数，以及排除 N 后的 F/B/E 排序诊断。分别保留 Row 后、Pair 后结果，才能区分结构能力增加和绝对分数偏移。排除 N 的结果仅作诊断，不用于给单例指定答案。

如果自然 Dev 的 E/FBE 改善，再接回旧 3290 行增强配方，和对应旧模型比较。若只在人工模板上提高、自然片段不改善，就不要把它解释为实际任务突破。

数据方面先整理 PLANE 的必要蕴含切片和 SherLIiC 的事件切片，审查标签、方向及重合；不要再次直接混入整库 MNLI/PPDB，也不再重复生成同一套规则模板。

如果明确要再换一个预训练模型，优先有短语实证的 PEARL-base，作为另一条独立实验；不与首次 composition 或新外部数据同时更换。

这个选择基于代码接入位置、现有错误类型和作者实现的适配成本。相关论文没有验证当前 DiCo 配方，不能据此给出必到 0.95 的概率。当前最有价值的检验是：增加真实片段上的关系判别信息，能否在已有分数偏移之外带来净修复。
