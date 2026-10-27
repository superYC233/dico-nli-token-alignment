# 借鉴版：原生四分类 + 真实反向KL

直接运行`train_deberta_reversal.py`，训练交给用户执行：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_reversal_kl/train_deberta_reversal.py
```

## 本次完整流程

```text
原生DeBERTa NLI编码器 + 原NLI pooler + 新四分类头
                         ↓
官方Train3042行按真实pair_id组批，8组最多16行
                         ↓
一遍前向得到本批所有行的F/B/E/N logits
                         ↓
所有原行：完整四类CE（等权，label smoothing 0.05）
真实正反组：交换反向F/B，计算完整四维对称KL
                         ↓
loss = CE + 0.5 * KL
一次反向、梯度裁剪1.0、AdamW更新
lr 1.5e-5，10% warmup后线性衰减
```

没有Alignment、Lexical、Child-D、三组合Pair CE或独立Pair-only阶段。不是在旧框架继续叠加一项损失。原NLI最后的三分类头因尺寸不同重新初始化为四类，pooler和编码器保留。

数据只有官方Train3042和Dev660；不使用人工增强、外部新NLI数据或人为反向N。完整一轮1760组、220次更新；3042行各出现一次，1282真实反向组各约束一次。

## KL到底比较什么

本目录标签顺序为F/B/E/N。对一组真实F/B或E/E行：

```python
p = softmax(logits_forward)
q = softmax(logits_reverse[[1, 0, 2, 3]])
KL = (KL(p || q) + KL(q || p)) / 2
```

两边都反传，不detach。保留N概率分量以比较两边的错误概率，不把三类重新归一化。这里N分量固定位置只是对已知可逆组的概率约束，不是为官方N单例添加确定反向金标。N单例只参与自己的四类CE。

两行一起错误答N时，KL仍可为0；完整CE会惩罚这个错误。因此一致性项不单独使用，也不保证模型从此所有关系都判断正确。

## 选模与评估

默认每1/4轮（55步）评估一次，按independent W-F1保存最佳；连续8次评估不提高则早停，最多10轮。耐心值单位是评估次数，不是epoch。只在刷新最佳时覆盖一份完整`best_model.pth`。

同时报告同一checkpoint的三种读出：

| method | 含义 |
| --- | --- |
| independent | 原始四类argmax，主指标 |
| twin | 真实两行组允许FB/BF/EE/NN竞争；保留N，因此SoftCons可能不到1 |
| pair_aware | 沿用我们旧版FB/BF/EE三组合，仅用于对比旧结果 |

全部模式的单例都由原模型预测，不使用“单例直接判N”。更细评估会增加Dev选择次数；这是开发集选模结果，不是独立测试成绩。`twin`里NN仅是解码候选，不产生N配对训练标签。

## 和作者方案的关系

来源：[MinLee0210/dico-nli-challenge](https://github.com/MinLee0210/dico-nli-challenge)，尤其README的模型、损失、训练调度说明和[实验记录](https://github.com/MinLee0210/dico-nli-challenge/blob/main/docs/EXPERIMENTS.md)。

借鉴其原生cross-encoder、CE加reversal-KL、warmup线性调度和细粒度验证。作者具体loss/config源码本轮未能取回，因此这里给出明确的本地数学实现，不宣称逐行复现。英文base NLI初始化、仅3042行、等权四类CE、0.25轮评估、耐心8次及梯度裁剪1.0均明确写入本目录配置。

作者有多语言模型、不同选模指标和集成/结构先验系统。这是英文单模型适配版，不是Clean-7、Prior-14或0.961成绩的复现；本版本同时改变模型读出、目标与训练流程，不作单组件因果归因。

## 文件与路径

- `config.py`：RUN_NAME和全部参数；后续新实验只改RUN_NAME即可保存到新位置。
- `data.py`：原始行和真实反向组，无词汇特征。
- `train_deberta_reversal.py`：原生模型加载、四类CE/反向KL、训练调度和选模。
- `evaluation.py`：三种读出、官方评分、混淆矩阵。
- `evaluate_best.py`：重载保存的最佳完整权重并核对逐条分数。
- `check_setup.py`：CPU机制检查；没有optimizer.step。

权重：`D:/model/SEAMEVAL/deberta_reversal_kl/run01_native_ce_kl_seed42/best_model.pth`。

结果：`experiments/deberta_reversal_kl/outputs/run01_native_ce_kl_seed42/`。

查看`best_summary.csv`和`best_metrics.json`，不要把不同step的独立与结构最高分拼在一起。`eval_summary.csv`记录所有验证step，`step_<数字>/predictions_with_scores.csv`保留逐条预测。重载评估无需再次下载原模型；使用保存的模型配置、tokenizer和完整权重。
