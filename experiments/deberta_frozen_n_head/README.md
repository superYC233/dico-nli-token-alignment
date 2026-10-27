# 冻结0.9381，重新训练N判断头

本次目标：**同一个checkpoint、同一轮、同一套四类分数**，独立W-F1至少0.87、pair-aware W-F1至少0.94。这是实验目标，不是已达到的成绩。

## 怎么运行

在PyCharm直接运行`train_deberta_n_head.py`。下一次实验只改`config.py`的RUN_NAME。这个脚本训练交给用户运行。

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_frozen_n_head/train_deberta_n_head.py
```

只检查原0.9381是否加载正确，不训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_frozen_n_head/train_deberta_n_head.py --check-only
```

训练完成后运行`evaluate_deberta_n_head.py`，从保存的完整模型重新前向，复核同轮两项成绩。

`check_n_head.py`是CPU小模型机制检查，不加载或训练真实DeBERTa。

## 使用什么数据、训练什么

- 精确复用0.9381那次的3290行：官方3042、旧scope200、新48。读取原训练快照并沿用哈希核对。
- 加载`D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged/best_model.pth`。
- 冻结184001800个原参数，并让原模型始终保持eval模式，关闭dropout。
- 新头输入是单行的CLS与原alignment池化特征。不输入pair_id、组大小、Dev标签或另一行的预测。
- 新头为LayerNorm → Linear(1280,64) → ReLU → Linear(64,1)，共84609个参数。
- 只用Train的N/非N标签训练BCE；正类权重1.0。没有第二阶段Pair CE，也没有新的人工样本。
- 默认30轮小头训练，batch64、lr1e-3、AdamW。编码器不反传，Train/Dev特征各提取一次。缓存含特征与原logits，不含标签。

## 四类输出怎么算

令原分数为uF、uB、uE，新N头分数为g。最终返回：

```text
[uF, uB, uE, logsumexp(uF,uB,uE) + g]
```

因此，归一化后的P(N)=sigmoid(g)，P(F/B/E)=(1-P(N))*softmax(uF,uB,uE)。新N头的BCE等于四类NLL中可训练的部分；F/B/E条件分类项已被冻结。

独立预测对这四类取最大值。**不能单独用sigmoid(g)>0.5代替四类argmax**，二者决策条件不同。

pair-aware仍从同一套分数中计算FB、BF、EE，单例保持独立答案。原F/B/E数值直接保留，配对行预测逐位不变，原Dev配对正确数应维持518/554。总pair-aware F1仍会随单例N表现变化，并不保证维持0.9381。

这是一个编码器加一个N头的完整模型，不是两个模型集成，也不是把结构答案伪装成独立预测。

## 怎样保存最佳轮

每轮计算：

```text
joint_score = min(independent_F1/0.87, pair_aware_F1/0.94)
```

选择joint_score最高的同一轮，完全相同则保留较早轮。两项都过线才会标记targets_met=true；没有过线就明确记录未达标。

每次改善只覆盖很小的`best_n_head.pth`。训练结束载回该头，再把原模型加新头一起保存为一个`best_model.pth`，完整模型只写一次，不保存30份编码器。

## 文件位置

- 模型：`D:/model/SEAMEVAL/deberta_frozen_n_head/run01_n_head/`
- 结果：当前目录`outputs/run01_n_head/`
- `joint_selection.csv`：一行一个epoch，独立和结构两项并列。
- `epoch_summary.csv`：逐轮两种方法的完整指标。
- `best_summary.csv`：同一个最佳模型的Train和Dev指标，split列区分。
- `best/dev/`：最佳Dev逐行预测、混淆矩阵和解码修好/破坏数量。
- `reference_check/`：先复现原0.9381，历史预测必须零差异。
- `feature_cache/`：一次提取的Train和Dev特征，只有本次小头训练需要。

独立评估读取本次保存的backbone_config、tokenizer和完整权重，不再依赖旧best_model.pth。Dev仍用于选模，且冻结的参考模型此前也是按Dev选出的；论文应如实披露两阶段开发过程。

## 后面是否做混合 + SWA

先完成本次实验，当前不自动接着跑混合+SWA。

这次先问：**固定当前成对关系排序后，已有隐藏特征能否支持更好的N区分？** 单看N召回不够，还要同时看独立非N→N与两项F1。

冻结配对预测时，N目前101/106，若另外修好2条而不破坏原正确单例，结构F1约可到0.941；这只是条件计算，不是增益预测。新N头也可能降低N召回。

- 如果同轮两项达标，先补一个种子检查新头训练的可重复性；原编码器仍来自同一个seed42，只能称新头种子复验，不能称完整训练多种子。
- 如果独立改善但N召回下降，SWA没有保证把高召回找回来。先看新头的Train/Dev差距和N误报。
- 如果候选在连续几轮已接近两个目标、主要剩小幅抖动，可以在这套固定结构上另做预先指定窗口的小头平均，对照同轮单模型。只有N头变化时，无需平均整个编码器。
- 混合训练会重新更新共享编码器，因此会失去本次固定F/B/E排序的性质。应作为另一种训练配方单独比较；不能从本次结果推定它加SWA就能同时提升。

现有混合训练和两阶段权重平均均未达到两个目标，因此“混合+SWA”保留为后续候选，不把它当作确定能解决问题的一步。
