# Lexical混合训练 + NLI pooler

直接运行：[train_deberta_pooler.py](train_deberta_pooler.py)。训练交给用户运行。

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_pooler/train_deberta_pooler.py
```

## 本次改什么

基于历史`deberta_lexical_mixed_pair/outputs/run01_mixed_pair/run_config.json`的训练设置，恢复固定base NLI checkpoint中已经训练的`pooler.dense`。不使用后来共享config里改成的Lexical学习率1e-4。

```text
旧CLS分支：CLS → dropout → 新四分类Linear
新CLS分支：CLS → 原NLI pooler（dense + GELU）→ dropout → 新四分类Linear

最终logits = CLS分数 + 原Token Alignment分数 + 原Lexical分数
```

pooler新增590592个可训练参数，使用主干学习率2e-5，不增加编码器前向次数。三分类NLI输出层不复用，F/B/E/N仍由新四分类头学习。恢复pooler不等于解决Pair目标的全部盲区，也不保证涨分。

## 数据和每轮流程

- 官方Train3042、Dev660；没有scope、relation、PAWS或新MNLI样本。
- 每轮1760个source组shuffle，8组最多16行，220次更新。
- 四类所有原行各参加一次Row CE，1282个真实Pair各参加一次Pair CE。
- 同一次前向计算`loss = row_loss + 0.5 * pair_loss`，然后一次backward/step。
- 没有先Row一整轮再Pair一整轮的独立阶段。N单例只参加Row监督，不造反向N。
- 主干及pooler lr2e-5，Lexical lr1e-3；10epochs，float32，固定长度128。
- Row权重[1,1,1,1.5]；Pair权重[1,1,1.25]、smoothing0.05。

## 选模与对照

本次目标是独立能力，因此按**independent W-F1**选最佳，同时报告同一checkpoint的pair-aware。这个选模设置与历史run01按pair-aware选模不同，但不影响训练更新。

默认只覆盖一份`best_model.pth`，完整保存编码器、pooler、Alignment和Lexical，不逐轮保存大权重；每轮预测和指标仍保留。

若要跑匹配基线，在本目录`config.py`中改：

```python
RUN_NAME = 'run02_without_pooler_seed42'
USE_NLI_POOLER = False
```

其余设置保持不变，包括按independent选模。已有run01的ep2只有预测，不能把另一个epoch保存的权重说成ep2的模型。先比较本次和已有同轮预测，不拼接不同epoch的最好分数。

本实验检验的是pooler读出（结构和预训练初始化一起改变）；不能仅凭正结果断言收益全部来自NLI知识。要进一步分辨初始化作用，再比较相同容量的随机pooler。

## 路径

只需修改本目录`config.py`中的RUN_NAME即可做下一次运行，旧实验目录不变。

- 权重：`D:/model/SEAMEVAL/deberta_lexical_pooler/run01_nli_pooler_seed42/best_model.pth`
- 结果：`experiments/deberta_lexical_pooler/outputs/run01_nli_pooler_seed42/`
- `epoch_summary.csv`：每轮同一checkpoint的独立/结构指标。
- `best_metrics.json`、`best_summary.csv`：最佳独立模型对应的两种分数。
- `run_config.json`：模型配置、固定revision、数据与代码哈希。

仅检查加载、真实CPU前向和小模型梯度，可运行`check_setup.py`；不会更新参数。

训练完成后运行`evaluate_best.py`，从完整最佳权重重新评估，不需要再次读取原NLI pooler权重。它核对重载分数与保存轮次一致，不依赖后来可能改变的pooler开关或隐藏层配置。
