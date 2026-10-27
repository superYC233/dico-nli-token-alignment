# 原0.938 + Row/Pair两阶段SAM

直接运行同一个 `train_relation_sam.py`。默认配置：

```python
RUN_NAME = 'run27_row_pair_sam'
SAM_RHO = 0.05
PAIR_SAM_RHO = 0.05
```

本次相对run26只增加Pair阶段SAM，Row训练函数、模型、padding和损失设置保持一致。run26的结果/模型不覆盖；修改前训练源码已按run26保存的SHA256核对，并保存在 `outputs/run26_row_sam/train_relation_sam_source.txt`，只供查看，不作为运行入口。训练仍由你运行。

## 保留与改动

数据固定读取原 `outputs/run06_relation48_staged/training_data.csv`：官方3042 + scope200 + relation48，共3290行，Dev660行。

模型仍是NLI DeBERTa-base + CLS四分类 + Token Alignment残差，全参数微调，从相同NLI初始化重新训练。没有词汇分支、LoRA、BLOOP或新增样本。

保留学习率2e-5、10轮、Row batch16、Pair batch16、AdamW、原类别权重、Pair系数0.5、Pair标签平滑0.05。

每轮仍是完整Row阶段206次更新，然后Pair-only阶段87次更新。Pair仍对两个方向分别前向，使用FB/BF/EE三组合，不回放Row、不构造NN；新改动是Pair也在临时扰动参数下重算双向前向和梯度。

为节省计算和显存，继续使用动态padding和非reentrant梯度检查点。设置 `PAIR_SAM_RHO=0` 恢复上一轮仅Row-SAM的Pair路径；`SAM_RHO=0`关闭Row-SAM。两者都为0时就是普通两阶段CE路径。动态padding会改变相对于历史固定128的dropout轨迹，不承诺逐字节复现0.938。

## SAM怎么更新

```python
# 每个阶段各自使用目标：Row用Row CE，Pair用0.5*Pair CE。
g = grad(stage_loss(w))
epsilon = stage_rho * g / (global_L2_norm(g) + 1e-12)
临时参数 = w + epsilon
g_sam = grad(stage_loss(临时参数))
恢复原参数w
AdamW使用g_sam更新一次
```

这不是将两遍CE相加，也不是新增λ乘一项损失。第一遍只找扰动，不能把第一遍梯度累加到第二遍。两遍使用相同输入、标签和dropout随机状态；Pair重放的是完整的两个方向，不能给两个方向错误地复用同一个dropout起点。第二遍结束后恢复随机状态，不额外推进后续shuffle/dropout序列。

Pair的0.5在每遍反向中都只乘一次：第一遍用于确定扰动，第二遍才产生最终更新梯度；不能再对第二遍梯度额外乘0.5。

扰动覆盖全部可训练参数，使用全模型单一L2范数，采用普通SAM，不是ASAM。两个阶段的临时参数都按备份精确恢复，然后才调用AdamW；保存和推理不带扰动。两个阶段顺序执行，复用相同逻辑，不同时持有两份参数备份；额外备份约0.74GB，用完即释放，没有新增永久模型参数。

本次在run26上单独检验Pair更新的扰动稳健性。它不消除原Pair三组合的欠约束，也不保证independent或pair-aware到某个分数。

## 成本与输出

Row和Pair各自都使用两遍梯度、一次优化器更新；Pair每遍包含两个方向的前向。每轮总更新仍293次。编码行数由run26的9340增加到12100，约增加29.6%；相对原0.938的6050行约翻倍，实际耗时依机器而定。

- 模型：`D:/model/SEAMEVAL/deberta_manual_aug/run27_row_pair_sam/best_model.pth`
- 结果：本目录 `outputs/run27_row_pair_sam/`
- 只按**Pair阶段结束后的Dev pair-aware W-F1**保留一份最佳完整权重；同时报告同一轮independent。
- `stage_summary.csv`、`stage_changes.csv`比较两个阶段。
- `sam_batches.csv`记录干净/扰动Row loss，新增`pair_sam_batches.csv`记录干净/扰动Pair loss、0.5加权后的loss、扰动半径及梯度范数。
- 最后重载最佳权重，评估Dev和Train，并检查Dev逐条预测与选中轮一致。
- 模型不保存Adam状态，不用于精确恢复中断训练。

单独预测用 `evaluate_relation_sam.py`。它不需要SAM辅助对象，不添加扰动；单例仍由模型四分类。

`check_relation_sam.py`检查公式、数据覆盖、Row及Pair各自rho0/原版梯度一致性、Pair完整双向RNG重放、0.5权重、SAM恢复和检查点兼容性；加 `--gpu` 可检查真实DeBERTa的两个SAM阶段，检查中的step不执行真实更新。

本次已通过语法、prepare-only、CPU梯度和真实GPU前后向检查；也已按源码语法树核对Row训练函数、模型构造和数据padding函数与run26保持一致。GPU使用184001800参数模型，Row为16×26、Pair两个方向各16×26；两个阶段交给模拟优化器前，全部参数都与原始备份精确一致，梯度有限。峰值allocated约2.45GiB，不含Adam状态，不能当作完整训练峰值。正式训练没有启动，也没有保存新模型权重。

## 来源

- [SAM原论文，ICLR2021](https://arxiv.org/abs/2010.01412)。
- [NLP实验：Sharpness-Aware Minimization Improves Language Model Generalization，ACL2022](https://aclanthology.org/2022.acl-long.508/)。该文不能当作DiCo/DeBERTa上的涨幅保证。
- [Google官方SAM实现](https://github.com/google-research/sam)。
- [PyTorch社区参考：davda54/sam.py](https://github.com/davda54/sam/blob/main/sam.py)。本任务使用独立的小型扰动辅助类，不安装此仓库，也不包装第二套优化器。
