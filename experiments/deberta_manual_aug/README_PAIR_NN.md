# 先Row CE，再四组合Pair CE

直接运行同目录的 `train_relation_pair_nn.py`。原训练文件、0.9381权重和run09代码不修改。本脚本不依赖run09的训练结果。

训练数据直接读取 `outputs/run06_relation48_staged/training_data.csv`，仍从同一个NLI预训练模型初始化。

## 训练顺序

1. Row阶段：原来的3290行做F/B/E/N四分类CE，每轮206步。
2. Pair阶段：真实FB/BF/EE组，加上从Train N单例构造的NN组，整体打乱，训练四组合CE。

四个候选分数是：

```python
[F1 + B2, B1 + F2, E1 + E2, N1 + N2]
```

NN进入同一个四分类softmax。非N组也会与NN竞争；不是仅对N组单独算一个容易的损失。

Pair权重0.5；四组合类别权重 `[1, 1, 1.25, 1]`；标签平滑0。学习率2e-5、10轮、float32、batch16。没有平均或保持损失。

## 实际数据

- Row仍是3290行，N为530行。
- 原1380个F/B/E组不变。
- 529条N可以生成反向N，形成529个NN组。
- `South Africans / South Africa` 在Train正反方向存在非N冲突标签，只跳过它的NN增强；原始Row监督保留。
- Pair共1909组，每轮120步；比原87步增加。每轮总更新从293步增加到326步。
- `pair_input_rows.csv`有3819行，是Pair编码所用的原行和新增反向行；Row训练没有变成3819行。

筛选只查看Train标签，不根据Dev金标接受或拒绝NN样本。N反转是本实验构造的监督，不冒充官方提供的反向金标。

## 三种评估

| method | 规则 |
|---|---|
| independent | 每行直接四分类 |
| pair_aware | 沿用原FB/BF/EE解码，单例不动，方便和0.9381同口径比较 |
| pair_four | 所有组都允许FB/BF/EE/NN；给单例额外预测交换文本后的输入 |

四组合解码不通过组大小预先排除N，也不会读取金标决定候选。已有两行就复用反向分数，没有反向行就交换文本预测；额外输入的label列仅为数据类要求而填固定占位，从未传给模型或用于解码。

Dev仍只评分原始660行，Train仍只评分原始3290行。生成的反向单例不计入样本数。原始行固定batch16，额外反向输入单独分批。

每轮评估Row和Pair结束后的Dev；第2轮额外评估两个阶段的Train。评估恢复模型模式和PyTorch随机数。

## 保存

默认 `SELECT_BY = 'pair_aware'`，继续按Pair结束后的原三组合Dev F1选模，只保存一个 `best_model.pth`。最终三个读出的分数来自这一份模型，不能理解成各自独立选出的最佳模型。

如果以后专门研究四组合选模，可另开RUN_NAME，设置 `SELECT_BY = 'pair_four'`；这是另一个选模协议。

模型：`D:/model/SEAMEVAL/deberta_manual_aug/run10_pair_nn/best_model.pth`

结果：本目录 `outputs/run10_pair_nn/`。

- `final_comparison.csv`：原0.9381、本次最佳模型三种读出，以及固定第2轮结果。
- `epoch_summary.csv`：所有阶段/读出的F1、各类正确数和六种错误计数。
- `phase_changes.csv`：Pair相对Row修好/破坏多少，各类正确数如何变化。
- `nn_reverse_audit.csv`：全部N的接受/跳过记录。
- `pair_groups.csv`：每个训练组的两行ID和目标组合。
- `epoch_*/after_*/dev/predictions_with_scores.csv`：原分数、反向分数和三种预测。

`--prepare-only`只准备训练快照和NN配对审计，不加载模型或训练。

比较边界：相对run09，此版新增NN候选、NN训练组和额外Pair更新；相对0.9381，还关闭了标签平滑。结果不能全部归因于单独一个因素，也不保证四组合会提高F1。
