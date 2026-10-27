# 词汇版 DeBERTa：整轮混合训练

前一版最后连续 81 步只训练 F/B/E 配对行，添加四分类 CE 后，N 正例在这一阶段
缺席，10 轮都出现 N 正确数下降。本版取消这两个连续阶段。

## 训练安排

1. 使用原始 Train 3042 行，不增强、不重采样。
2. 配对的两行作为一组，N 单例也作为一组，共 1760 组。
3. 全部组一起打乱；每批 8 组，最多 16 行。每轮 220 步。
4. 每批只前向一次：所有行计算四分类 CE，其中真实配对组再计算 Pair CE。

```python
loss = row_loss + 0.5 * pair_loss
```

每轮每条原始行恰好出现一次，每个真实配对组恰好出现一次。
N 在整轮中穿插，不保证每个批次都有 N；全 N 批次只计算四分类 CE。
训练日志记录各类行数，应为 F=881、B=881、E=802、N=478。

模型、Token Alignment、20 维词汇特征及评分直接复用
`experiments/deberta_lexical_stage_probe` 的实现，没有新增模型分支或其他正则项。
学习率、类别权重、Pair 权重和 smoothing 沿用前版。
从同一个 NLI 主干重新训练，不继续加载已经退化的权重。

这是训练日程的对照：相对两阶段版，步数、批次组成和样本曝光次数都改变了，
不能称为“只改一个损失系数”。它也不同于 `deberta_lexical_joint` 的
“第二阶段额外抽取辅助 Row 批次”：本版没有第二阶段，所有行只走一遍。
之前无词汇分支的 `deberta_joint_pair` 已试过类似混合训练；本版是在原词汇模型上
验证这一安排，不宣称是全新方法，也不保证能超过旧的 0.9306。

## 运行

直接在 PyCharm 运行本目录的 `train_deberta_mixed_pair.py`。

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_mixed_pair/train_deberta_mixed_pair.py'
```

配置在 `config.py`：10 轮、seed=42、float32、主干学习率 2e-5、词汇分支 1e-3。
模型默认保存在：

```text
D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run01_mixed_pair/best_model.pth
```

每轮分别报告 independent / pair-aware。仍按 pair-aware W-F1 保存一个最佳模型，
与上一版选模口径相同；`best_metrics.json` 和 `best_summary.csv` 同时保留这个权重
对应的两种成绩。`epoch_summary.csv` 保存逐轮结果及三项损失，不保存每轮模型。
训练结束时内存是最后一轮，重新评估最佳权重运行 `evaluate_best.py`。

## 同脚本的必要对照

当前默认 `RUN_NAME='run01_mixed_pair'`、`PAIR_WEIGHT=0.5`。
如要确认 Pair CE 的增量，把配置改成下面两项再独立运行：

```python
RUN_NAME = 'run02_mixed_row_only'
PAIR_WEIGHT = 0.0
```

其他配置不动。两次使用同样的模型初始化、混合批次、样本次数和优化步骤，
都只有一次前向；差别才是 Pair CE 是否参与更新。
不能把两阶段版的 after_row 当成这个纯 Row 对照，它已经受此前 Pair 更新影响。

主要观察独立 F1、N 正确数、非 N→N 错误及 E/F/B 混淆，同时保留最终结构成绩。
如果只提高一个分数，不能宣称所有目标都改善；Dev 仍是反复使用的开发集。

`test_cpu.py` 只核对完整数据覆盖、配对索引、方向特征、损失和 Dev 随机数隔离，
不加载预训练权重、不调用 backward 或 optimizer.step。

## run04：降低主干学习率，检查泛化

直接运行 `train_deberta_slow_backbone.py`，使用单独的 `config_slow_backbone.py`。
原 `config.py` 和训练入口不用修改，也不用手动恢复 run03 的词汇学习率。

以 run01 为对照：主干学习率 2e-5 降到 5e-6，CLS 和 Token Alignment 保持 2e-5，
词汇分支保持 run01 的 1e-3；没有冻结参数。模型、样本、混合批次、Pair 损失、
seed 和 10 轮预算保持一致。这检验主干是否更新过快，不保证提高分数。

本次按 independent F1 选模，只保存一个最佳权重，同时报告对应的 pair-aware 分数。
旧 run01 要按相同规则比较：独立基准为第2轮 0.850565，而不是其结构最佳轮的 0.846092。
这项选模规则调整单独记录在 run_config.json 中，不能称为整个评估协议完全未变。

权重：`D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run04_slow_backbone/best_model.pth`。
结果：`outputs/run04_slow_backbone/epoch_summary.csv`。
训练后可运行 `evaluate_slow_backbone.py` 重新加载最佳权重复测。
`test_slow_backbone.py` 只检查参数分组及配置，不训练。

## run05：Base混合训练 + R-Drop

**右键运行 `train_deberta_rdrop.py`。** 配置单独放在 `config_rdrop.py`，原 `config.py`
和旧训练入口保持原样，不需要手动把run03改回run01。

本次恢复run01配方：原始3042行、base NLI主干、Token Alignment和Lexical，
float32、主干学习率2e-5、词汇分支1e-3、8组/批、每轮220次更新、10轮、seed42。
从NLI预训练初始化开始，不加载Large或旧的任务checkpoint续训。

同一批输入在不同Dropout下前向两次：

```text
loss = 两次Row CE的均值
     + 0.5 * 两次Pair CE的均值
     + 1.0 * 四类分布的对称KL
```

两次前向都接受金标监督，KL包含F/B/E/N；对类别求和、对样本平均，不detach。
每步只合计反向一次、更新一次。新增一次前向会增加训练时间和显存，推理仍只前向一次。
若将 `RDROP_WEIGHT=0` 并换一个RUN_NAME，会恢复单次前向的原始联合损失路径；
这是整个R-Drop方案与原训练的对照，不是“保留双CE、只去掉KL”的分量消融。

**按独立W-F1选模，只保留一个 `best_model.pth`**，每轮同时保存独立/结构预测和分数。
Row CE、Pair CE、R-Drop KL分开记录。推理仍固定batch16、float32。

- 权重：`D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run05_mixed_rdrop/best_model.pth`
- 成绩：`outputs/run05_mixed_rdrop/epoch_summary.csv`
- 最佳权重的两种成绩：同结果目录下 `best_summary.csv`
- 训练后可用 `evaluate_rdrop.py` 单独加载最佳权重评估；不用运行旧的 `evaluate_best.py`。
- `test_rdrop.py` 是CPU损失/配置检查，不加载预训练模型、不更新模型参数。

方法出处：[R-Drop论文](https://arxiv.org/abs/2106.14448)、
[作者代码](https://github.com/dropreg/R-Drop)。本次没有同时改变数据、模块或其他正则项。

## run06：Train内部三折难例评分 + 最终单模型训练

**首次直接右键运行 `run_oof_all.py`。** 先完成3折评分，再自动训练最终一个模型。
本次恢复原run01的混合训练，不叠加R-Drop。所有新配置在 `config_oof.py`，旧代码不改。

### 自动运行什么

1. 读取原始3042行Train，规范化Unicode、大小写和空白。相同无序文本对作为同一个
   划折组，可逆的两行、不同pair_id下重复的文本不会落入不同折。
2. 三折分层分组。每折从相同NLI主干重新初始化，用约2/3的Train固定训练3轮，再
   预测剩余1/3。不会读取官方Dev，不会按留出分数选择epoch，不加载已有DiCo权重。
3. 保存每条训练样本的折外概率：每条样本只由没训练过它的模型评分一次。
4. 从FB/E/N三种分组类型各自的OOF边界错误中，按低金标概率选难例，最多选该类型
   25%的组。仅F↔B判反的组不因方向错误被选入。整对的两行一起调整权重。
5. 难例原始倍率为2，其余为1；每类内部归一化，使各类样本权重总和等于原样本数。
   原Row类别权重 `[1,1,1,1.5]` 保留。只改变Row CE的样本权重，Pair CE不变。
6. 从NLI初始化重训最终一个base模型：完整Train、10轮、float32、8组/批、220步/轮。
   按Dev独立F1保存一份最佳权重，同时报告对应pair-aware成绩。

三折模型不保存权重，评分完成就释放。最终预测不使用模型集成。
这次预算是 **3次短训练 + 1次最终训练**，不是只训练一次。

### 入口和路径

- `run_oof_all.py`：一键运行完整流程。
- `score_train_oof.py`：只做三折评分，想先看难例清单时运行这个。
- `train_deberta_oof_hard.py`：已有三折评分后，只训练最终模型。
- `evaluate_oof_hard.py`：训练后加载最终最佳模型重新评估。
- `oof_data.py`：分组、权重和批次整理。
- `test_oof.py`：CPU划折/权重/损失检查，不训练模型。

评分输出：`outputs/oof01_train_only/`。

- `fold_assignments.csv`：所有行的折号、规范化分组和冲突标记。
- `fold_1_scores.csv`等：每折评分；中断重跑时复用已完成折。
- `train_oof_scores.csv`：按原始Train顺序合并的3042行折外分数。
- `hard_examples_for_review.csv`：候选难例、原金标、OOF预测、金标概率和建议权重。
- `label_conflicts_for_review.csv`：已识别的标签冲突关联行。

最终权重：`D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run06_oof_hard/best_model.pth`。
最终成绩：`outputs/run06_oof_hard/epoch_summary.csv`。
权重审计：同目录的 `train_sample_weights.csv`、`group_weights.csv`、`class_weight_totals.csv`。

### 标签与复核

程序保留官方金标，不生成新标签。规范化文本冲突关联组不额外加权，原始行仍参与
训练；OOF错判本身不证明标签已经审核正确。难例清单供复核，默认流程不会等待人工
确认。要排除某些可疑难例的额外加权，在 `EXCLUDE_PAIR_IDS` 填入对应pair_id。
自动生成的评分和清单不要手改；修改排除名单后仅重跑最终训练即可，另换RUN_NAME。

### 同预算随机加权对照

若要检验收益是否来自“挑对难例”，在 `config_oof.py` 改：

```python
RUN_NAME = 'run07_oof_random'
WEIGHT_MODE = 'random'
```

保持OOF_RUN_NAME不变，直接运行 `train_deberta_oof_hard.py`。随机模式在FB/E/N组内
打乱同一套权重，保持相同的加权数量和倍率；不需要再训练三个折。
`WEIGHT_MODE='uniform'` 是样本等权对照，损失及梯度恢复原run01的Row CE+Pair CE。

OOF阶段只使用Train；官方Dev仍用于最终选模与报告，不是新的未使用测试集。
三折固定3轮模型的难例评分属于内部训练诊断，不应与全量训练后的Dev分数混报。
