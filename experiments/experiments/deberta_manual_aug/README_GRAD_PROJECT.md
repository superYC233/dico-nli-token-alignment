**Row优先的梯度投影：先诊断，再比较两个训练臂**

代码继续放在experiments/deberta_manual_aug，没有改动原训练代码、数据、权重。真实训练由你启动。

先右键运行train_relation_grad_project.py，默认：

```python
MODE = 'diagnose'
RUN_NAME = 'run13_gradient'
```

它只加载原0.938权重，在原3290行Train里按固定种子抽取前24个混合批次。一次前向分别计算Row与Pair梯度；没有优化器、没有参数更新，不读取Dev样本、不保存模型。诊断时开启训练模式dropout，参数版本保持不变会被检查。

结果在outputs/run13_gradient_diagnose。先看gradient_summary.csv的all行：conflict_rate是两种非零梯度方向冲突的批次比例，mean_cosine为平均夹角余弦，mean_pair_to_row_norm_ratio为加权Pair梯度与Row梯度范数比。分层行用于定位冲突发生在哪些层。

这里只检查固定checkpoint附近的局部方向，不是重建原训练轨迹；冲突多不自动证明投影能提高Dev，冲突少也不排除其他阶段有冲突。程序不会依据一个任意阈值自动开训。

此次已实际运行24批无更新诊断，结果在outputs/run13_gradient_check_diagnose：11/24批冲突，平均余弦-0.026789，加权Pair/Row范数比的均值0.083772，投影系数绝对值的均值0.007708。原普通混合梯度在这24批上与Row梯度的内积均为正。因此有局部冲突，但不是强烈反向，当前结果不支持“普通混合更新整体都在破坏Row”的说法，也不能由此预测明显提分。真实DeBERTa两次反向和参数不变检查均已通过，没有训练或运行优化器。

**两个训练臂怎么运行**

先改MODE并右键运行：

```python
MODE = 'mixed'
```

完成后改为：

```python
MODE = 'project'
```

保留相同RUN_NAME。输出和模型路径自动加模式后缀，彼此分开：

- outputs/run13_gradient_mixed、outputs/run13_gradient_project。
- D:/model/SEAMEVAL/deberta_manual_aug/run13_gradient_mixed/best_model.pth。
- D:/model/SEAMEVAL/deberta_manual_aug/run13_gradient_project/best_model.pth。

两臂都从同一个NLI预训练初始化重新训练，不从0.938续训。保持3290行、8组/批、239步/轮、10轮、float32、lr=2e-5、AdamW、原类别权重、Pair系数0.5和平滑0.05。不新增NN，不冻结，不加词汇分支，不加KL或SWA。

配对组的两行不拆散，N单例混在同一组加载器中。每行每轮出现一次。N只算Row CE，配对行同时算Row CE和三组合Pair CE；随机批次不保证每批都有N，gradient_batches.csv记录每类行数。

两臂都采用一次前向、两次autograd.grad，所以同一次前向里的dropout完全相同，梯度测量不会额外消耗前向随机数；批次顺序也相同。计算成本高于原先一次合并反向，两臂保持同一诊断开销。数学上mixed的合成梯度等价于普通loss相加；浮点运算顺序不同，不承诺与旧run08逐字节复现。

**投影到底怎样做**

    gr = grad(Row CE)
    gp = grad(0.5 * Pair CE)

    两者内积 >= 0：g = gr + gp
    两者内积 < 0：
        c = dot(gr, gp) / dot(gr, gr)
        gp_projected = gp - c * gr
        g = gr + gp_projected

所有可训练参数共同构成一个梯度向量，只计算一个全局c；不把每层分别投影，也不分别归一化两个梯度。Row不被投影，Pair已经加权，组合时不会再乘一次0.5。某个参数没有Pair梯度时，也必须计入全局投影的-c*gr项。

使用的是借鉴PCGrad的非对称版本，优先保留Row目标，不是原论文的对称随机投影算法。论文：https://arxiv.org/abs/2001.06782 。

投影只保证局部梯度内积的数学性质；AdamW还有预条件、动量和权重衰减，因此不保证每步Row损失必降，更不保证Dev两个F1同时提高。gradient_batches.csv中的row_dot_assigned_gradient指合成梯度，不是实际AdamW参数位移。

**选模与保存**

两臂都用预先固定的同轮指标：

```python
selection_score = min(independent_f1 / 0.87, pair_aware_f1 / 0.94)
```

选择这个值最高的epoch，相同取更早轮。0.87和0.94只用作双目标标尺，不参与训练损失，不是保证达到的分数。避免只按pair-aware挑到独立退化的轮次。每次刷新才覆盖唯一best_model.pth；不会保存每轮模型。

- epoch_summary.csv：每轮同一模型的independent与pair-aware结果。
- selection_history.csv：每轮两个F1和固定的选模值。
- best_metrics.json、best_summary.csv：所选同一轮的指标。
- best/dev、best/train：重新加载最佳权重后分别评估，不能混淆Train/Dev。
- gradient_batches.csv：全模型每批梯度与F/B/E/N曝光、Train样本ID。
- gradient_layers.csv：分层每批梯度。
- gradient_summary.csv：按epoch与层汇总。
- run_config.json：数据哈希、代码哈希、设置、诊断权重来源与实验边界。

gradient_summary.csv汇总的是投影前的两个原始梯度。project模式下conflict_rate不会自动变成0；是否投影看gradient_batches.csv的projection_applied，冲突批投影后的理论内积看row_dot_pair_if_projected。它也记录在mixed/diagnose中，表示假设启用投影时的数学结果，不表示这些模式实际做了投影。

旧0.938是按结构F1选出的模型，是历史结构成绩参考；本次检验应先比较mixed与project在相同规则下的增量，再看是否同时改善独立与最终成绩。不把不同epoch的峰值拼成一个模型。

**检查命令**

CPU自检不下载、不加载DeBERTa、不更新参数：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/check_grad_project.py'
```

只检查原数据与路径，不创建训练记录：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/train_relation_grad_project.py' --prepare-only
```

MODE也可在命令行设置，不改文件：--mode diagnose、--mode mixed、--mode project。重跑同一模式时修改RUN_NAME。默认不自动串行运行两组训练。
