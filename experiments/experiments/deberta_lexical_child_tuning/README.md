# 原Lexical混合版 + Child-Tuning-D

直接右键运行`train_deberta_child_tuning.py`。训练交给用户执行。

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_child_tuning/train_deberta_child_tuning.py
```

## 每次运行做什么

```text
1. 从原NLI checkpoint初始化DeBERTa + Alignment + Lexical，没有pooler。
2. 仅遍历官方Train一次，计算四分类Row CE梯度，不更新权重。
3. 在encoder层内按平均平方梯度的重要度，固定保留约20%的坐标。
4. 正式混合训练10轮：
   logits = model(batch)
   loss = Row CE + 0.5 * Pair CE
   loss.backward()
   遮掉未选encoder坐标的梯度
   原AdamW.step()
5. 按independent F1选最佳，同时报告同一权重的pair-aware。
```

Fisher准备不是额外训练一轮：有前后向，但没有optimizer.step。准备结束清空梯度、恢复CPU/CUDA随机状态。正式训练loader的shuffle生成器从未用于Fisher，不提前消耗正式训练批次。

Fisher只读Train，不用Dev、历史错误或Pair CE。使用与正式Row相同的类别权重[1,1,1,1.5]。它是作者实现中的“逐参数裁剪后，批梯度平方平均”的近似，不是精确逐样本Fisher。

## 沿用的设置

- 官方Train3042、Dev660，无增强、反向N、PAWS或新的外部训练集。
- 固定原base NLI revision：`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`。
- 原Token Alignment、20维Lexical分支和四分类头直接复用。
- 每轮1760组shuffle，8组最多16行；3042行各训练一次，1282对各算Pair一次，220步。
- 主干学习率2e-5，Lexical学习率1e-3，AdamW weight_decay0.01，float32，长度128。
- Pair权重[1,1,1.25]、label smoothing0.05、系数0.5。
- 不做末尾独立Pair-only阶段，不加入pooler、SAM、SWA或新损失。

## “20%”准确指什么

仅筛选名称以`deberta_model.encoder.layer.`开头的参数坐标，所有encoder层共同取全局分位数，掩码整个训练固定。相同阈值的并列坐标都保留，因此以fisher_summary.json中的实际比例为准。

embedding、encoder共享相对位置embedding、CLS分类头、Alignment和Lexical不遮罩，继续正常训练。**不是整个模型只训练20%，不是LoRA，也不是省去80%反向计算。**

遮罩针对数据梯度；原AdamW的weight decay仍作用于未选坐标，因此不能称这些坐标严格冻结。掩码在第一个Adam状态产生前固定，不继承先前未遮罩训练的动量。

保留现有PyTorch AdamW及两组学习率，不复制作者2021版优化器的eps、调度器等其他配方。这里只适配Child-D的重要度选择和梯度遮罩逻辑，避免把更换优化器实现混成方法收益。

## 额外成本

开始会多一次Train梯度准备，共191批；之后每步仍只有一次主干前向。正式训练额外保存bool掩码，准备时还需要浮点Fisher累计及CPU分位数工作空间。不能保证训练更快或显存减少。

## 保存和查看

- 权重：`D:/model/SEAMEVAL/deberta_lexical_child_tuning/run01_child_d_seed42/best_model.pth`。
- 结果：本目录`outputs/run01_child_d_seed42/`。
- 只在independent刷新最佳时覆盖一份完整模型，掩码只保存一次。
- `epoch_summary.csv`、`best_summary.csv`、`best_metrics.json`记录同一轮两种成绩。
- `fisher_summary.json`记录筛选范围、实际比例、Train类别计数。
- `fisher_parameters.csv`记录每个参数的保留数量。
- 完成训练后运行`evaluate_best.py`。完整模型可直接推理，无需加载梯度掩码，也不依赖原checkpoint再次初始化。

下一次只修改本目录`config.py`中的RUN_NAME。若要匹配基线，改新RUN_NAME并设`USE_CHILD_TUNING=False`；不会运行Fisher或遮罩，但仍按independent选模。

先比较相同数据/同轮模型的independent，以及N误报漏报、E与F/B混淆；不能只看N正确数或把其他epoch的结构分拼进来。此方法是小样本泛化假设，不保证超过0.852或达到0.95。

## 方法来源

- [Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.749/)。
- [作者calculate_fisher](https://github.com/RunxinXu/ChildTuning/blob/main/ChildTuningD.py)。
- [作者梯度掩码优化器](https://github.com/RunxinXu/ChildTuning/blob/main/ChildTuningOptimizer.py)。

`check_setup.py`用于CPU检查数据覆盖、Fisher不改权重、随机状态恢复、掩码范围、梯度以及小模型保存重载。真实模型仅检查两行，不执行完整Train准备或任何optimizer.step。
