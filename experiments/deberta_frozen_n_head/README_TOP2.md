# 顶部两层微调 + 四分类监督 + F/B/E条件保持

## 运行

直接右键运行`train_deberta_top2.py`。新的模型与结果均放在`run02_top2_keep`，不覆盖旧run01。

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_frozen_n_head/train_deberta_top2.py
```

只核对起点与缓存、不开启真实训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_frozen_n_head/train_deberta_top2.py --check-only
```

训练后运行`evaluate_deberta_top2.py`，从保存的一个完整checkpoint复核同轮独立和pair-aware指标。

配置只改`config_top2.py`；后续同配方实验换RUN_NAME即可。`check_top2.py`是CPU小模型机制检查，不训练真实DeBERTa。

## 与上次的区别

| 项目 | run01冻结N头 | 本次run02 |
|---|---|---|
| 起点 | 原0.9381 + 新初始化N头 | run01第4轮完整最佳模型 |
| 训练参数 | N头 | 顶部两层编码器 + N头 |
| 学生特征 | 提取一次后固定 | 每批重新计算，允许反传 |
| 损失 | N/非N BCE | 四类Row CE + 0.5×条件KL |
| 配对行预测 | 严格不变 | 可能变化，用保持损失约束 |

上次最佳同轮是独立0.825260、pair-aware 0.906702。本次首先复现该起点；不是从原NLI模型重新初始化，也不是把0.9381和0.8253拼成一个成绩。

## 每轮流程

1. 打乱3290行原训练快照。仍为官方3042 + scope200 + 原48，不新增样本。
2. 每批一次学生前向；每行计算四分类Row CE，N样本也参与。
3. 对教师在Train上条件分类正确的非N行，额外计算F/B/E条件KL。
4. 两项相加后一次反传，同时更新顶部两层与N头。
5. 每轮结束重新逐行预测Dev，用同一套分数计算独立和pair-aware F1。
6. 按两个目标的最小完成比例选最佳轮；起点epoch0也参与比较。

这是Row CE与条件保持的联合微调，没有Pair-only阶段，不计算Pair CE，不添加NN。随机批次不保证每批恰好含N，但所有N每轮都按四类金标参与一次监督。

## 保持损失的来源与意义

直接复用run01的`feature_cache/train.pt`中的`base_logits`。它是原0.9381在3290行Train上的分数，按训练快照和行序核对。

只取F/B/E三列、在三类内归一化：

```text
teacher = softmax(original_train_logits[F,B,E] / T)
student = softmax(current_logits[F,B,E] / T)
loss = CE4(current_logits, gold) + 0.5 * T² * KL(teacher || student)
T = 2
```

仅非N且教师条件argmax与Train金标相同的行参与KL；教师在Train已经判错的关系不强制保留。所有行仍参与Row CE。

保持项完全不读取教师N分数。Dev只用于评价和选模，不生成训练目标。该约束用来减少已学关系的漂移，不引入新的语义知识，不能保证分数提升。

顶部两层更新后，N所使用的特征和F/B/E分数都可能改变，不能再沿用上版“518条配对预测严格不变”的保证。

## 参数

- 编码器最后2层学习率2e-6；N头1e-4；AdamW，weight_decay=0.01。
- float32，batch16，5轮，梯度范数裁剪1.0。
- embedding、底部10层、原CLS分类器、Token Alignment参数冻结。
- 固定模块保持eval，顶部两层的dropout按训练/评估模式切换。固定对齐与分类运算仍可把梯度传回顶部编码器。
- 学生不能复用旧特征缓存，因为顶部层正在更新。教师缓存保持固定，不额外运行第二个编码器。

## 保存与评估

```text
joint_score = min(independent_F1 / 0.87, pair_aware_F1 / 0.94)
```

两项均来自同一轮同一个checkpoint，同一套四类分数。只有两项均达标才标记targets_met=true。

起点为epoch0。如果所有微调轮次更差，最佳模型仍保留起点并明确打印“未超过起点”。只在joint_score刷新时覆盖一个完整best_model.pth，不按每轮保存多个模型。

- 模型：`D:/model/SEAMEVAL/deberta_frozen_n_head/run02_top2_keep/`
- 结果：当前目录`outputs/run02_top2_keep/`
- `joint_selection.csv`：每个epoch同轮的两个F1、选模分数和训练损失。
- `epoch_summary.csv`：每轮两种预测的完整指标。
- `parameter_groups.csv`：所有参数的冻结/训练范围。
- `best_summary.csv`：同一个最佳checkpoint的Train/Dev结果。
- `best/dev/`：逐行预测、混淆矩阵、解码修好/破坏数量。

初始模型和其参考模型此前均按Dev选过，论文需要记录该开发过程；本次不是完全独立的新测试。

## 混合 + SWA

本次不开启SWA，也不自动接着运行另一套混合Pair训练。先检验“少量特征更新 + 单行监督 + 条件保持”能否改善同轮两项成绩，再根据曲线决定是否值得研究固定窗口平均。平均不能预先保证达到0.87/0.94。
