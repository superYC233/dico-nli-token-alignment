# Pair阶段补齐N/非N监督

直接右键运行`train_relation_pair_gate.py`。本次代码放在原`deberta_manual_aug`目录，旧训练代码和模型不变。

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments\deberta_manual_aug\train_relation_pair_gate.py
```

默认：

```python
RUN_NAME = 'run24_pair_gate_seed42'
GATE_WEIGHT = 1.0
```

## 为什么做这个

最新run20的最佳第4轮，Row后独立F1为0.839388，Pair后掉到0.464267。非N→N从51变成355，其中283行原本答对。
355条误报中310条在F/B/E内部仍是金标最高，只是N超过了它。问题不只是缺少语义样本，也包括四类读出在Pair阶段发生漂移。

普通三组合Pair CE没有N项，不能直接惩罚这个错误。此前完整Row回放又把单行F/B/E分类全部加回来，尚未超过0.938。
本次检验一个更窄的变化：第二阶段只保留Row目标里的“N与非N”部分，让Pair目标负责F/B/E组合。
这是损失拆分试验，仍属于混合目标训练；不是新的分类头、模型集成或推理规则。

## 每轮做什么

```text
第一阶段：原3290行整体shuffle
    四分类Row CE，206次更新
    评估Dev，记录独立/结构F1与非N→N

第二阶段：1380个真实Pair，配入已有530条N
    loss = 0.5 × Pair CE + 1.0 × N/非N CE
    87次更新，每批最多39行
    评估Dev，再比较阶段前后的变化

如果Pair后pair-aware F1刷新最高值，覆盖唯一best_model.pth
```

第二阶段每个真实Pair出现一次；N单例也各出现一次，没有N过采样或反向N。
因此每条训练行每轮共编码两次：Row一次、第二阶段一次；不是只把N多训练一遍。
不读取Dev标签构造训练样本或训练目标。

## 二分类损失怎么写

仍使用模型原来的四类logits：

```python
gate_logits = torch.stack([
    torch.logsumexp(logits[:, :3], dim=-1),  # F/B/E合在一起
    logits[:, 3]                           # N
], dim=-1)

gate_targets = labels.eq(3).long()
gate_loss = gate_criterion(gate_logits, gate_targets)
loss = 0.5 * pair_loss + 1.0 * gate_loss
```

二分类softmax恰好为：

```text
[P(F)+P(B)+P(E), P(N)]
```

- 真F/B/E行的目标是非N，会直接惩罚过高的N概率。
- 真N行的目标是N，保持有效N监督。
- 不在此处要求F、B、E中具体哪一类正确；这个目标交给原Pair CE。

要用`logsumexp(F,B,E)`，不能换成`max(F,B,E)`，后者不再是准确的N/非N概率。
也不对logits或概率detach，梯度会更新原编码器、原分类头及原Token Alignment。

## 和旧完整四类回放有什么区别

对非N行，四类交叉熵可以准确拆成：

```text
四类CE = -log(1-P(N)) + 条件F/B/E交叉熵
```

对N行，则只有`-log(P(N))`。

旧回放：

```text
0.5 × Pair CE + N/非N CE + 非N行的条件F/B/E CE
```

本次：

```text
0.5 × Pair CE + N/非N CE
```

区别是去掉第二阶段单行F/B/E条件分类项，保留缺失的N/非N监督。第一阶段仍然完整训练四分类。
这给出了一个明确、可比较的目标变化，不把已经做过的完整Row回放改名重跑。

代码中的加权处理保留相同分母：Row权重[1,1,1,1.5]，二分类对应[1,1.5]。没有额外提高N的类别系数。
与历史原版相比，N在第二阶段开始参与损失；与此前完整回放相比，样本曝光和分组预算相同。

## 数据、模型与参数

- 原0.938训练快照3290行：官方3042 + 旧scope200 + 旧relation48。
- **不加入最新失败的64行，不使用PAWS或MNLI新数据。**
- 同一NLI初始化：`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`，固定原revision。
- 原CLS四分类 + Token Alignment残差，全参数184001800；不加新头或词汇分支、不使用LoRA。
- 从NLI初始化重训，不加载按Dev选出的0.938权重继续训练。
- lr2e-5，10epochs，Row batch16，Pair batch16组，float32。
- Row权重[1,1,1,1.5]，Pair权重[1,1,1.25]，Pair smoothing0.05、系数0.5。
- 动态padding + 非重入梯度检查点，降低显存占用。

原两阶段每轮编码6050行；本次为6580行。相比原版增加530行第二阶段N曝光，相比完整Row回放保持相同曝光预算。
动态padding会改变dropout随机轨迹，不声称逐位复现历史0.938。

## 推理与保存

预测仍对原四类logits取argmax；pair-aware仍在真实两行组里计算FB/BF/EE。
**二分类仅为训练损失。预测不增加N阈值，不把单例统一判N。**

模型目录：`D:/model/SEAMEVAL/deberta_manual_aug/run24_pair_gate_seed42/`。
结果目录：本目录`outputs/run24_pair_gate_seed42/`。
只在Pair后Dev pair-aware F1刷新最高值时覆盖`best_model.pth`，不保存十份完整模型。
训练结束自动重载最佳模型，核对Dev预测，再评估Train。
需要单独重载评估时运行`evaluate_relation_pair_gate.py`。

主要记录：

- `stage_summary.csv`：每轮Row/Pair结束的两种F1和完整分类指标。
- `stage_changes.csv`：每轮阶段变化、非N→N、各类正确数变化。
- `pair_gate_batches.csv`：第二阶段损失和训练行索引，核对所有行各一次。
- `best_metrics.json`：同一个最佳checkpoint的独立/结构F1、非N→N与对应Row阶段独立F1。
- `best/dev/`、`best/train/`：重载后各自的逐条结果，避免把Train成绩当Dev。

## 这次怎样判断是否有效

先看Pair后是否仍把大量正确非N变成N，再看F/B/E排序和最终结构F1有没有提高。
所有成绩必须来自同一个checkpoint，不能拼接不同轮次的最好数字。

二分类损失不在数学上保证每个非N行的四类argmax都正确：非N总概率较高时，F/B/E分散仍可能让N单类最大。
它也会通过共享编码器间接改变F/B/E排序，不是硬冻结或无损保护。
所以这是针对当前漂移的一次可验证试验，没有超过0.938或达到0.95的保证。

普通完整回放、KL保持、梯度投影此前已有实现；本次区别在于第二阶段监督的具体子目标，不把这些旧实验当成未试的新方案。

## 检查与可选对照

只核对数据和编码，不加载模型权重或训练：

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments\deberta_manual_aug\train_relation_pair_gate.py --prepare-only
```

机制检查，不执行优化器更新：

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments\deberta_manual_aug\check_pair_gate.py --real
```

已通过：交叉熵拆分、二分类概率、梯度符号、Gate系数零时退化为原Pair目标、3290行无重复遗漏、真实模型38行单批反向与参数不更新。
真实GPU检查峰值约2.04GiB，不含Adam状态，不能当作完整训练峰值承诺。

若以后需要同代码损失对照，换RUN_NAME并设`GATE_WEIGHT=0.0`；默认不自动跑多组。
该对照仍保留相同第二阶段batch和N输入，因此不是历史原版的逐位复现。
