# 官方3042 + Lexical + Train逐样本margin保护

直接运行`train_deberta_lexical_margin.py`。默认先做1轮机制验证：

```python
RUN_NAME = 'run07_row_margin_probe'
EPOCHS = 1
```

模型保存：`D:/model/SEAMEVAL/deberta_lexical/run07_row_margin_probe/best_model.pth`。
结果保存：本目录`outputs/run07_row_margin_probe/`。

不从0.9306或0.908权重续训。从相同NLI初始化开始，官方Train3042行、Dev660行、1282组真实Pair；原DeBERTa、Token Alignment、20维Lexical、两个学习率、Row/Pair权重、Pair平滑、FP32、固定padding128与原Lexical一致。旧脚本不修改。

## 每轮流程

1. 原Row CE：191次更新。
2. 完整Train的eval前向：只记录Row判对、且正确标签分数严格高于竞争类别的行。
3. 原Pair CE：81次更新；每次另外处理一批Train保护样本。
4. Pair结束后完整Train及Dev评估，统计单行损坏及结构分。
5. 只覆盖保存目前Dev pair-aware F1最高的一个完整模型，并报告同轮独立F1。

保护样本是原3042行，无新增数据。每次约38行、最后一批2行，每轮遍历一次，包括478条N。保护前后向拆成16行的microbatch，与Pair大图先后反传，再做一次共享AdamW更新，不把全部计算图叠在显存里。

## 保护公式

```text
margin_i = 正确标签分数 - 最高错误标签分数
floor_i  = min(Row结束时margin_i, 1)
damage_i = max(0, floor_i - 当前margin_i)
```

仅对Row已判对且margin>0的Train行计算damage。原来判错的行不锁定。每条damage非负，另一条变得更自信不能抵消损坏。按F/B/E/N分别取保护行damage均值，得到4个零预算约束g_c<=0。

```text
模型更新：AdamW(0.5*Pair CE + sum(lambda_t[c] * g_c))
乘子更新：lambda_next = max(0, lambda_t + 0.1 * g_detached)
```

四个乘子各从0.25开始；每轮Row参照刷新后重新初始化。模型更新使用旧乘子，乘子随后基于同一旧参数上的损坏更新，属于同时原始—对偶。microbatch共享同一组旧乘子，分母来自完整保护批次，不能分别平均。

这是参考[Cooper同时更新公式](https://cooper.readthedocs.io/en/latest/_modules/cooper/optim/constrained_optimizers/simultaneous_optimizer.html)的自包含任务适配实现，不依赖`cooper`安装包，也不是原论文DiCo实验复现。

## 重点看输出

- `margin_audit.csv`：每轮、每类Train中原先正确的保护行，Pair后损坏多少（`protected_broken`）；剩余margin违反量与最终乘子。
- `stage_summary.csv`：Train/Dev，Row/Pair前后，两种解码的F1及类别正确数。
- `stage_changes.csv`：Pair前后修好/破坏数量。
- `constraint_batches.csv`：每步类别损坏、旧/新乘子、Pair损失。
- `epoch_1/after_row/train_margin_reference.csv`：Train参照行、保护标记与margin下限。

这不是硬步长投影或逐步回退算法。随机保护批次与有限乘子不保证每一步或整轮严格满足所有约束，必须看完整Train审计。守住Train不等于Dev一定提高。Dev没有进入保护目标、乘子或反向传播。

预测保持原四类argmax及FB/BF/EE解码，单例继续由模型判断。评估脚本为`evaluate_lexical_margin.py`，无需加载参照或乘子。

## 检查及后续运行

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical/check_lexical_margin.py'
```

`--gpu`仅增加真实模型前后向与显存检查，不更新真实模型参数。先前后向检查使用构造的参考以覆盖非零保护梯度，不伪称是已训练Row模型。

本次已通过：CPU公式与梯度、microbatch等价、原版关闭保护路径及随机序列、Train覆盖和原词汇特征对照。真实GPU检查覆盖32条Pair输入、38条保护输入（16条分块）与等量Adam矩缓冲，峰值allocated约8.728GiB；没有执行真实模型优化器更新。实际AdamW临时分配及其他程序占用不包含在此数字中。

训练脚本复用`train_deberta_lexical_anchor.py`中已验证的数据校验、模型构造和加载函数；没有调用它的参数锚定训练、没有参数回拉。

若后续决定完整10轮，改同一训练脚本中的：

```python
RUN_NAME = 'run08_row_margin_full'
EPOCHS = 10
```

也支持`--run-name`、`--epochs`参数。已有RUN_NAME不会被覆盖。`--no-protection`提供同代码原Pair梯度路径对照，不会自动启动第二个实验；它也保留完整Train/Dev诊断。
