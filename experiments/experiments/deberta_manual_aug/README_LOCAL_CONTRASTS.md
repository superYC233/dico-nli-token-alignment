# 局部对照：先诊断，再训练

代码继续放在deberta_manual_aug。没有修改旧训练脚本、旧数据或旧模型。

## 先运行哪一个

先右键运行 `evaluate_local_contrasts.py`。

它先核对0.938模型和run03模型能否逐条复现已有Dev预测，再用两个模型分别预测同一批64行新诊断。这里只预测，没有训练。

主看输出的independent及四个子类型错误。pair-aware仍使用FB/BF/EE及N单例，是额外的结构结果，不能代替独立诊断。

汇总在 `outputs/run01_local_diagnostic/comparison.csv`，逐条改动在同目录的 `row_comparison.csv`，各模型的错误表在其子目录的 `independent_errors.csv`。

## 这次新增什么

| 类型 | 内容 | 新Train | 留出诊断 |
|---|---|---:|---:|
| name_boundary | 虚构品牌的名称边界、属性限制和品牌描述改写 | 32 | 16 |
| purpose_place | 物品用途限定、地点和时间是不同条件 | 32 | 16 |
| path_actor | 环境中的动作不等于主体的动作 | 32 | 16 |
| role_restriction | 明确职级限制、活动改写和独立职业活动 | 32 | 16 |

新Train共128行，F/B/E/N各32；诊断共64行，各16。每类4个训练词族、2个诊断词族。内容词干在两个新分区间隔离，构造模板和语法算子可以共用；不声称所有结构都未见过。旧3290行中已有的词汇也不声称全部未见过。

N使用不同的单例pair_id，不创建NN配对。F/B和E正常生成反向行。一条职级或动作样本依照正向字面读法标注，不把“参与某事件”等同于“事件的另一个阶段”。数量、疑似SIMI/EQUI冲突和需源句才能解释的题本次都未扩写。

运行准备脚本后，全部120个基础对照及每条理由在 `outputs/local_contrasts_prepared/examples_source.csv`，展开后分别为 `extra_train_rows.csv` 与 `diagnostic_rows.csv`。这是助手人工设计的标签，尚无独立人工仲裁；没有冒称为官方数据。

构造只用Dev文本作为拒绝黑名单，不加载Dev标签；新Train和新诊断均检查词形、Porter词干、整短语及无序文本对与Dev零交集。还排除与旧3290行重复的整短语/文本对。审核结果在 `audit.json`。

## 之后训练

诊断看完后，右键运行 `train_local_contrasts.py`。顶部默认：

```python
RUN_NAME = 'run11_local128_manual'
AUGMENT_MODE = 'manual'
```

这次是旧3290+新128=3418行，从原NLI预训练初始化重新训练，不接着0.938权重续训。架构、float32、随机种子42、学习率2e-5、10轮、batch16、类别权重、Pair权重0.5和平滑0.05全部沿用0.938配置。训练函数直接复用旧代码，仍每轮先完整Row CE，再完整Pair CE。

选模同样沿用Dev pair-aware F1，只保存一份最佳权重；同时逐轮记录同一模型、同一四类logits的independent与pair-aware。不会拿不同轮次拼成绩。64行留出诊断只在最终选模结束后运行一次，不参加训练或选模。

模型保存到 `D:/model/SEAMEVAL/deberta_manual_aug/run11_local128_manual/best_model.pth`。结果保存到代码目录的 `outputs/run11_local128_manual`。

## 等量复制对照

需要区分“新内容”与“增加行数/训练步数”时，改两行再运行同一个脚本：

```python
RUN_NAME = 'run12_local128_copy'
AUGMENT_MODE = 'copy'
```

copy随机复制16个FB组、16个BF组、16个EE组和32个N单例，也是新增128行、四类各32，组内方向构成也与manual一致。manual和copy均为3418行、1428个可逆组，每轮214个Row步+90个Pair步。

`baseline`模式使用旧3290行，可用新RUN_NAME检查旧配方；不自动串行跑多个训练。复制的旧Train不符合“新造文本零Dev词汇”的承诺，审计中与新造文本分开说明。

## 只检查，不训练

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/prepare_local_contrasts.py'
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/evaluate_local_contrasts.py' --check-only
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/train_local_contrasts.py' --prepare-only
```

第二条只严格加载两个现有checkpoint并各做CPU两行前向。第三条只准备数据并核对旧超参数。正式训练仍由你运行。
