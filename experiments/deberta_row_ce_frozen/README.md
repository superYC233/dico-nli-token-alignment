# 冻结参数 Row CE 实验

本目录只保留冻结参数相关的训练和结果：冻结 embedding 与底部 6 层，使用纯 Row CE。
原版全参数增强实验已移到 `experiments/deberta_original_aug`，请在那里运行 `train_original_aug.py`。

## 冻结 Row CE 的范围增强 / 等量复制

**训练由用户启动；此次只准备了数据和代码，没有运行训练。**

下面是已完成的冻结版实验说明，使用 `train_deberta_row_ce.py` 和 `config.py`。范围增强时配置为：

```python
RUN_NAME = 'run02_scope'
AUGMENT_MODE = 'scope'
```

这次从原 NLI 初始化重新训练，保留冻结底部 6 层、学习率 2e-5、N 权重 1.5、10 轮和纯 Row CE。新增 200 条人工模板对照，Train 3042 → 3242；每轮 235 步。

新增标签按“字面名词短语、并列项不自动共享修饰词”解释构造；这不是官方新增金标，也没有独立标注者复核。可先查看 `outputs/run02_scope/extra_train_rows.csv` 及 `augmentation_audit.csv`。

训练对照包含：简单限定 F/B、保义改写 E、对象变化 N、并列限定 F/B、属性归属变化 N、并列换序 E。不能仅凭是否出现 and 或词数变化决定标签。

完成后，以同一脚本跑复制对照，只改两项：

```python
RUN_NAME = 'run03_copy'
AUGMENT_MODE = 'copy'
```

复制来源排除与 Dev/保留诊断题已有重合的组。两臂新增的 F/B/E/N 数量分别为 40/40/80/40，组数、每一步标签和行数、总更新次数相同。与原始 3042 行版本相比两者都增加了步数，因此 scope 相对 copy 的增量才隔离了新增内容的作用。

每轮仍只按独立 Dev F1 保存唯一最佳权重。训练结束恢复该最佳权重，自动评估保留的 200 条诊断题；它们不参与训练或选 epoch。未审核的建议标签匹配率与审核后成绩分开报告。

模型保存到 `D:/model/SEAMEVAL/deberta_row_ce_frozen/<RUN_NAME>/best_model.pth`，结果保存在本目录 `outputs/<RUN_NAME>/`。同名模型/训练记录已存在时会停止，避免覆盖。

此次仅运行了数据/分组预算检查：两臂 3242 行、1880 组、每轮 235 步，同 seed 每批标签和张量大小一致；没有构建或更新训练模型。旧审计脚本已固定指向 `run01_freeze6`，不会因新 RUN_NAME 改变而误加载。

## 历史基线：run01_freeze6

直接运行 `train_deberta_row_ce.py`。

对照对象是刚完成的 `experiments/deberta_joint_pair_frozen/outputs/run01_freeze6`：
其独立 F1 最佳为 0.843989，同一轮 pair-aware F1 为 0.896533。

本次只改：

```python
PAIR_WEIGHT = 0.0
loss = row_loss
```

Pair CE 在权重为零时完全跳过，不计算也不反向传播。
保留 embedding、底部 6 层、底部卷积及相对位置表冻结；顶部层与分类头、Token Alignment 继续训练。
原 NLI 初始化、数据、seed=42、学习率 2e-5、类别权重、10 轮、float32、220 步/轮均不变。
不从上一版最佳 checkpoint 续训，从同一 NLI 初始化重新训练。

## 为什么还保留分组？

配对的两行继续放在同一 batch，仅为保持批次组成和之前一致，隔离 Pair CE 的影响。
每行模型前向仍只输入自己的两个短语；不会读取另一行的标签或分数。
训练目标只根据每行四类 logits 和它自己的金标计算。
每轮 Train 3042 行各出现一次，N 单例也参与；没有新增数据。

## 评估与保存

仍按独立 Dev F1 保存唯一最佳模型，每轮同时报告 independent / pair-aware。
预测阶段的 pair-aware 保留，用来比较同一个模型两种解码的表现；不参与训练。

模型目录：`D:/model/SEAMEVAL/deberta_row_ce_frozen/run01_freeze6`。
结果目录：本目录 `outputs/run01_freeze6`。
训练完运行 `evaluate_best.py`，重新加载最佳权重。
再次训练先改 RUN_NAME，防止覆盖旧结果。

如果独立成绩明显改善，支持当前配方的 Pair CE 对独立分类不利；
如果仍在约 0.84，说明只移除 Pair CE 没有解决泛化瓶颈。
单种子结果不能证明 Pair CE 在所有配置下一定有害。

`test_cpu.py` 只检查零权重损失及梯度、配方一致性，不下载模型，不使用 GPU。
