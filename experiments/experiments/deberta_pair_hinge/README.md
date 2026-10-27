# 原3290行：混合CE3与16组合hinge

默认右键运行本目录 `train_deberta_pair_hinge.py`，先试hinge16。训练由你手动启动。

## 本次保持什么

- 数据直接读取 `experiments/deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv`，即0.938145那次的3290行快照：官方3042 + scope200 + 原48。不生成新样本。
- 标签数量：F933、B933、E894、N530；1380组真实Pair、530条N单例，共1910组。
- 模型仍是NLI初始化的DeBERTa + CLS分类 + Token Alignment，全参数训练。0.938版没有启用Lexical分支，这里也不启用；不加载旧最佳权重续训。
- lr=2e-5、AdamW、weight_decay=0.01、FP32、10轮、最大长度128。
- 每批8组，最多16行，每轮239步。整组随机打乱，配对的两行不拆开；每条Train行每轮出现一次。
- 每步只前向一次，再做 `Row CE + 0.5 × Pair loss`，反向一次、更新一次。所有N单例参与Row CE。随机组批可能有个别批次不含N，日志记录其数量。
- 两臂使用同一组批、初始化种子、类别权重、Row CE与更新次数。

这不是0.938两阶段配方的逐步复现。0.938是每轮先Row后Pair；这里两臂都用混合训练，只有Pair目标不同。因此应先比较本目录的ce3和hinge16，再对照历史0.938结果。

## 损失的区别

`ce3`：沿用三组合FB/BF/EE的加权CE与0.05 label smoothing。

`hinge16`：计算FF到NN的全部16个加和分数，按两行真实标签取金标；排除金标自身，选其余15个组合的最高分：

```python
loss_per_pair = max(0, 1.0 + highest_negative_score - gold_score)
```

沿用金标组FB/BF/EE权重 `[1, 1, 1.25]`，使用加权均值；hinge没有label smoothing。两臂比较的是完整Pair目标替换，包含候选集合、CE/hinge形式及其平滑方式的差别，不是单独的候选集合消融。

训练中含N的组合可以作为配对行的负例。N单例仍有Row CE正监督，不人工构造NN正样本。

`losses.py` 是新损失的完整实现。此实现没有增加两行交互网络；组合分数仍然是两行logit相加，hinge会重点约束较弱的单行分类间隔。

## 运行

默认实验：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_pair_hinge/train_deberta_pair_hinge.py'
```

相同流程的CE3对照：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_pair_hinge/train_deberta_pair_hinge.py' --pair-loss ce3
```

两个实验分别写入 `run01_mixed_hinge16` 和 `run01_mixed_ce3`，不会自动连续训练。再次实验改 `config.py` 的 `RUN_NAME`，或加 `--run-name run02`。右键想跑CE3，也可以把 `PAIR_LOSS` 改为 `ce3`。

只准备快照、不训练：在训练命令后加 `--prepare-only`。

公式、数据与旧CE梯度检查：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_pair_hinge/check_pair_hinge.py'
```

在检查命令后加 `--gpu` 可做真实模型单批前后向；不执行optimizer.step，不保存模型。

## 保存与查看

模型：`D:/model/SEAMEVAL/deberta_pair_hinge/run01_mixed_hinge16/best_model.pth`。

结果：本目录 `outputs/run01_mixed_hinge16/`。CE3对应目录结尾为 `ce3`。

- 每轮Dev同时报告independent与pair-aware，保留逐行分数、混淆矩阵、N正确数、非N→N数量和解码修好/破坏数量。
- 每轮是否同时达到independent≥0.87、pair-aware≥0.94写在 `epoch_summary.csv`。两项来自同一轮，不能拼接不同轮的最佳分。
- 沿用pair-aware W-F1选模；仅刷新纪录时替换一份 `best_model.pth`，不保存每轮权重。`best_summary.csv` 和 `best_metrics.json` 同时报告该权重的两项分数。
- `comparison_with_0p938/` 比较旧0.938最佳轮次。两臂都跑完后，后跑的一臂自动输出相对另一臂的修好/破坏表。
- 预测仍只在实际配对行中选FB/BF/EE；单例保持模型自己的四分类argmax。训练的16组合负例不改变推理候选规则。

重载最佳模型：右键 `evaluate_deberta_pair_hinge.py`；对CE3模型添加 `--pair-loss ce3`。重新预测固定batch=16。

## 上一版margin保护运行记录

`deberta_lexical/outputs/run08_row_margin_3epochs` 实际运行了10轮，名字不代表训练轮数。它使用官方3042行 + Lexical，和本次3290行非Lexical配置不同。

- 按pair-aware选出的最佳权重：第8轮，independent=0.833252，pair-aware=0.902628，N正确84，结构F/B/E正确512。
- 第10轮Pair后：Train independent=0.998684，Dev independent=0.835186；保护Train上的正确判断没有保证Dev同步改善。
- 第10轮Row后的Dev independent=0.846868，和第8轮最佳结构权重不是同一个模型。
- 第8轮Pair后Train保护样本没有被破坏，但Dev仍有9条原本正确的独立预测被破坏。不能把Train保护达成等同于泛化改善。

因此本次暂不叠加margin保护、SWA、KL或新增强；先观察完整四类监督与含N负例的Pair margin能否同时改善两项Dev分数。
