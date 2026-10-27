# 从Le Duc Minh公开方案借鉴到本项目：方法适配判断

核对日期：2026-10-26。结合本地阅读报告、实际模型代码和作者公开网页。没有训练模型，也没有修改已有训练配置。

## 1. 决策

不能从对方0.961推导出“换成他的模型就能得到更好的独立预测”。该数字是DEV选择的集成系统，并使用结构先验。模型页无结构先验的英文分层系统为0.856，但仍含模型组合与解码，不能称纯单模型independent。

对当前提高independent的目标，最值得新增的一项具体检查是：**恢复base NLI checkpoint训练过的ContextPooler，保留现有Alignment和Lexical，检验原生NLI读出能否比裸CLS读出迁移得更好。** 这不是已证实的收益，也不是对方高分的已知原因，而是对照模型接口后得到的新假设。

分层N门、反向一致性和N反向增强都有相近旧实验，不能重新命名成三个新突破。保留训练中持续四类监督的底座，不指望恢复pooler消除Pair-only损失的不充分监督。

## 2. 成绩与证据边界

作者模型页：[LakoreAI/dico-nli-checkpoints](https://huggingface.co/LakoreAI/dico-nli-checkpoints)。公开成绩并未在本机复现。

- Clean-7、Prior-14、按赛道选择的系统不是同一个单模型，只能比较其公开设置，不能把全部差值归给某一个组件。
- 对方结构先验将无反向记录的单例直接判N。我们历史pair-aware只排除真实配对行的N，单例仍由模型预测，设置不同。
- 对方无先验成绩也可能有正反联合或多语言source合并，不自动等于独立预测。
- 作者源文件的具体训练实现本轮未成功取回；README、模型页与实验说明可读。因此本文明确区分公开声明、已核实本地代码及我们的适配推断。

作者入口：[README](https://github.com/MinLee0210/dico-nli-challenge)、[实验记录](https://github.com/MinLee0210/dico-nli-challenge/blob/main/docs/EXPERIMENTS.md)。

## 3. 最具体的模型差别：我们丢弃了NLI pooler

本地当前模型：`experiments/deberta_lexical/model.py`。其CLS路径是：

```text
DeBERTa最后层CLS → dropout → 新四分类Linear
```

模型通过AutoModel加载，只取编码器。本地固定revision的safetensors头信息确认，原NLI权重还有：

```text
pooler.dense.weight  [768,768]
pooler.dense.bias    [768]
classifier.weight   [3,768]
classifier.bias     [3]
```

checkpoint：`D:/BaiduNetdiskDownload/PretrainedModel/models/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/snapshots/6f5cf0a2b59cabb106aca4c287eed12e357e90eb/model.safetensors`。

对应[模型config](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli/blob/main/config.json)声明DebertaV2ForSequenceClassification、pooler_hidden_act=gelu。原生分类器使用ContextPooler变换CLS，再接分类层。

作者模型页声明使用AutoModelForSequenceClassification；通常将三类分类层改成四类时，可以保留相容的pooler参数。但未读取其具体加载函数，不能断言作者所有实验都做了这一保留。

建议测试的本地路径：

```text
CLS → 原NLI ContextPooler → 新四分类Linear
                                  ↓
               + Token Alignment logits
               + Lexical logits
```

不增加主干前向；base pooler新增约59.1万参数。不能把原NLI的entailment/neutral/contradiction直接按位置复制为F/B/E/N，它们不是一一对应标签。只保留已训练的pooler，新四类输出层仍需学习。

已有`experiments/baselines/deberta_large.py`使用原生SequenceClassification，但这是large、不同训练配方，不能排除base在相同Lexical底座上保留pooler的收益。当前base Lexical及双向模型没有这项相同配置比较。

限制：大部分NLI知识本来就在编码器中，并没有因丢pooler而全部丢失。恢复pooler只是一个读出假设；若以后需证明收益来自NLI初始化而非增加非线性容量，可再比较相同结构的随机pooler。

## 4. 分层概率可以借鉴，但不是一条全新损失

作者分层组合为：

```text
p(N)=g
p(F/B/E)=(1-g) × q(F/B/E)
```

普通四分类分布也可以如此分解，非N行的NLL恰好是：

```text
-log(1-g) -log(q_gold)
```

仅把输出写成二分类门加三分类头，不自动增加信息，也不保证两个任务的共享编码器不再相互影响。真正隔离梯度需要明确哪些参数接收哪个任务的梯度；只分开最后两个Linear仍共享编码器。

我们已有：

- `experiments/auxiliary/deberta_nli_base_hierarchical.py`：relation/direction分层头。
- `experiments/deberta_frozen_n_head/model.py`：固定原F/B/E、新学N头，概率公式正是上式。
- `experiments/deberta_manual_aug/train_relation_pair_gate.py`：Pair阶段补N/非N损失。

对方的分层组合使用不同训练来源的模型专家，与“单编码器加两个头”也不相同。可借其职责分解解释本项目，但不应声称我们从未试过N门，更不能直接以对方分层集成成绩预估单模型收益。

## 5. 反向KL与现有Pair CE不同，但仍不提供正确标签

对方README声明reversal-KL。我们旧`experiments/pair_training/deberta_pair_consistency.py`做过可逆行正反概率MSE，不是严格同一个KL；R-Drop则是同输入的两次dropout，也不是swap-KL。

一个足够说明限制的例子：

```text
p(A,B)=[0.01,0.01,0.01,0.97]
p(B,A)=[0.01,0.01,0.01,0.97]
```

即便两行金标为F/B，交换对齐后的四类分布相同，KL为0，却两行都错。所以swap-KL可约束不一致，不能单独解决稳定答N的退化。四类KL有时能感知两方向N概率不匹配，不应笼统声称它对所有F/B/E平移严格不变。

本地官方`evaluation_functions/labels.py`明确只有F↔B、E↔E映射，N没有deterministic reverse。这不证明每个N在语义上都不对称，但官方没有为全部N提供可直接复用的反向金标。不能再把所有N反转称为官方保证正确的监督。

若研究swap-KL，应先限于有真实反向金标的F/B/E组，完整四类CE持续存在；不能单靠一致性项训练，也不能从原N单例硬推反向N标签。

## 6. 正反log-prob合并不是新的解码收益来源

若双方都限定FB/BF/EE候选：

```text
log p0(a)+log p1(b)
= z0(a)+z1(b)-logZ0-logZ1
```

最后两项对三个候选相同，因此与我们的logit加和选出相同argmax；平均而非求和也不改变结果。区别来自是否允许N、是否强制单例N、是否合并多语言视图，而非把logits换成log-prob这一操作。

## 7. 值得学习的训练管理和多语言方向

作者实验记录指出细粒度验证、较小学习率和早期峰值；这提示我们观察epoch内部的独立分数变化。更频繁评估只改变观察与选模，不能当成模型学到更多语义；频繁复用DEV也增加选择偏差。若采用，预先固定频率，并优先在Train按源pair划分的内部验证上决定训练长度。

多语言同源Train视图比任意新造模板更贴近任务标注体系，是后续可研究的数据方向。但同一个pair的多语言行并非新增独立语义标签。应只用Train视图，按源pair分组，避免同一源对的不同语言落入训练/内部验证两侧。英文模型不能原封不动混入其他语言；需要多语言编码器或教师。测试时跨赛道合并输入还涉及赛道规则，不能默认采用。

作者自己的对照中N反向增强、部分外部数据和WordNet并未稳定胜出，不支持立即大规模添加这些数据。

## 8. 下一步顺序

1. 若要继续提高独立能力，先用现有稳定Lexical混合训练做参照，单独恢复base NLI pooler；相同数据、种子、损失和评估协议，不同时添加新样本、KL或Child-Tuning。
2. 观察同一checkpoint的independent、N误报/漏报、E↔F/B和pair-aware。改善必须体现在真实错误减少，而非只提高N召回。
3. 仍保留原两阶段模型作为高结构分对照，但pooler不会在数学上修复三组合Pair-only目标的盲区。
4. 若pooler无效，再按上一份研究报告评估Child-Tuning-D；不要因看到0.961就重开集成、N反转、同类分层头的旧实验。

这是基于公开方法接口与本地代码差异的适配建议，不是对pooler、Child-Tuning或0.95的效果保证。本轮没有启动训练，现有运行文件保持不变。
