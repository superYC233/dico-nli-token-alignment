# 家族增强768行 + 保留诊断192行

数据已在 `prepared/` 生成。训练仍由你启动，原Train、旧增强、旧模型和hinge目录均不改。

## 新数据

| 家族类型 | 训练家族 | 新训练行 | 保留家族 | 诊断行 |
|---|---:|---:|---:|---:|
| 属性范围与独立属性 | 8 | 128 | 2 | 32 |
| 空间参照与修饰 | 8 | 128 | 2 | 32 |
| 施事/受事属性与动作 | 8 | 128 | 2 | 32 |
| 动作对象、位置、时间限制 | 8 | 128 | 2 | 32 |
| 事件活动区别 | 8 | 128 | 2 | 32 |
| 明确数量与描述 | 8 | 128 | 2 | 32 |
| 合计 | 48 | 768 | 12 | 192 |

每个家族包含4组限制关系、2组等价关系、4条N：展开后F/B/E/N各4行，共16行。F/B/E保留完整互换Pair，N仍是单例。限制组的FB/BF起始方向各一半。

合并训练集：原0.938版本3290 + 本批768 = **4058行**。总标签F1125、B1125、E1086、N722。

这批没有把任意中心词替换都当N，也没有把数字变小机械判F：

- 属性N是两个独立属性的对照，不要求两条互相矛盾。
- 空间E仅使用静态under/underneath等明确读法，不机械改任意介词。
- 参与者家族区分人员属性与物品属性；复合修饰的E按泛指物品类别读，不额外断言精确基数。
- 不从“检查/运输、计划/完成、起草/续约”的常见流程推导必然蕴含。
- 数量显式写exactly；不同精确基数是N。没有把“恰好17个带属性对象”标成蕴含“总共恰好17个对象”。

这是助手设计、按普通字面义标注的人工模板草稿，未经独立标注者仲裁。逐条理由和读法可在源表中审核；不是新官方样本，也不声称程序证明了全部自然语言标签。

## 直接查看

- `prepared/extra_train_rows.csv`：本批768行，格式与项目Train一致。
- `prepared/training_data_manual.csv`：可直接训练的4058行合并表。
- `prepared/examples_source.csv`：600个源对照，包含家族、用途、源标签与理由；其中480个用于训练、120个保留诊断。
- `prepared/manual_label_notes.csv`：训练行的逐条理由、源对照与正反方向。
- `prepared/diagnostic_rows.csv`：192行保留家族。
- `prepared/family_manifest.csv`：60个家族的固定划分，不按Dev分数筛选。
- `prepared/word_provenance.csv`：词项在官方Train中的原文出处，或标为词干相关/新选常用英文。并非所有新词都在Train出现过。
- `family_examples.py`：可读的家族词项与生成公式；改动后重跑准备脚本。

## 隔离与审计

准备脚本只读取Dev的text1/text2做拒绝黑名单，不读取其标签。

本批新增训练与保留诊断，分别与Dev的 **完整词形、Porter词干、完整短语、无序文本对** 交集均为0；功能词和数字也计算。两分区之间没有完整短语或无序文本对重合，诊断家族的指定中心词词干也不出现在新增训练文本中。

本批还检查原3290行、已知旧人工增强和原保留诊断的文本对；保留诊断的完整短语不出现在这些旧材料中。单个词可能在原Train出现，不能把它说成全部词汇未见过的测试集；共享关系模板是有意设计。

新增样本不增加同一输入的标签冲突。官方原有的South Africans/South Africa冲突原样保留，写到 `existing_train_label_conflicts.csv`；复制池不额外复制冲突组。

`prepared/overlap_report.json` 记录检查结果、参与检查的旧文件及哈希。严格词汇隔离适用于本批新增/保留诊断，不适用于旧3290行和复制对照。隐藏Test不可核验，不宣称与Test零重合。家族研究方向受之前Dev分析影响，应在论文中披露。

## 运行

重新准备数据，不训练：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_family_aug/prepare_family_data.py'
```

默认训练：右键 `train_deberta_family_aug.py`。

默认 `RUN_NAME='run01_family768'`、`AUGMENT_MODE='manual'`。原0.938模型和两阶段循环保持相同：全参数、无Lexical分支、10轮、FP32、lr=2e-5、Row batch16、Pair batch16、Row权重[1,1,1,1.5]，三组合Pair CE权重0.5及原平滑0.05。每轮Row254步 + Pair105步，不加入hinge或其他新目标。

复制对照：只把 `AUGMENT_MODE` 改成 `copy`，再次运行同一文件。自动保存到不同目录；复制768行的类别数、配对数、组内FB/BF顺序与新人工数据完全一致。

也可用命令：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_family_aug/train_deberta_family_aug.py' --augment-mode copy
```

`baseline`模式只用原3290行。`--prepare-only`仅准备数据。新一轮实验改RUN_NAME，不修改已有运行。

只保存按Dev pair-aware F1选出的唯一最佳权重，同时报告该轮independent成绩。每轮预测/混淆矩阵保存，但不保存每轮模型。训练结束才评估保留家族，保留家族不用于选轮次。

模型目录：`D:/model/SEAMEVAL/deberta_family_aug/run01_family768_manual/`。

结果目录：本目录 `outputs/run01_family768_manual/`。复制对照结尾是 `_copy`。

## 看是否学到可迁移规则

右键 `evaluate_family_probes.py` 默认先评估原0.938权重：

`D:/model/SEAMEVAL/deberta_manual_aug/run06_relation48_staged/best_model.pth`

若旧权重已移动，用 `--model-dir` 指向保留旧权重的实际目录。新模型训练完成后自动跑同一批192条；也可以手动指定新模型目录：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_family_aug/evaluate_family_probes.py' --model-dir 'D:/model/SEAMEVAL/deberta_family_aug/run01_family768_manual'
```

主要比较同一批保留家族上的 **independent**，同时查看Dev。不要用pair-aware诊断成绩代替模型独立能力，也不要根据这192条反复调参后继续称其为未见测试。

先让已在运行的hinge实验完成；本入口用于单独验证新数据，避免同时更换损失和训练集。

## 检查

运行 `check_family_data.py`：检查属性/析取的集合关系、基数读法、整家族拒绝机制、互换标签、预算匹配、诊断隔离和实际分词长度；不加载DeBERTa模型、不训练。
