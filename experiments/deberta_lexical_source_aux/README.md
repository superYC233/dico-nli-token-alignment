# distance＋PhrasIS原始关系辅助监督

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

直接右键运行 [train_deberta_source_aux.py](train_deberta_source_aux.py)。这是当前稳定distance版本的新增实验，沿用DeBERTa＋Token Alignment＋Lexical、官方3042行，不是原0.9306全参数Pair配方。原目录不修改，正式训练由你运行。

```powershell
python 'experiments/deberta_lexical_source_aux/train_deberta_source_aux.py'
```

## 只增加什么

主分类仍预测F/B/E/N；新增线性头，从同一次前向的CLS表示预测F/B/E/SIMI/REL/OPPO。辅助头与主CLS头共用已经产生的dropout表示，不增加额外主干前向或dropout采样。它把辅助梯度传回共享编码器，不把六类概率合并成四类，也不参与预测。

```text
一轮：
  Row：distance四分类损失＋0.1×六类辅助CE，全部参数更新（191步）
  评估：只用主四分类logits，记录independent和pair-aware
  Pair：原0.5×FB/BF/EE损失，只更新Alignment（81步）
  再评估：仍只用主四分类logits
```

辅助头只有hidden_size×6＋6个参数，base模型为4614个。主干2e-5、Lexical 1e-3、batch16、FP32、10轮、seed42、distance margin=0.5均不变。辅助头用主干组的2e-5；六类权重1/1/1/1.5/1.5/1.5，与主任务粗类别权重对应，不额外放大稀少的OPPO。

## 辅助标签来自哪里

**文本与官方主标签不变。** F/B/E的辅助标签直接沿用官方Train；只有N回溯项目已有的两个PhrasIS源文件。匹配按NFKC、大小写、空白规范化，不删除标点、不做模糊匹配。

| 官方Train原行 | 辅助标签 | 数量 |
|---|---|---:|
| F | F | 881 |
| B | B | 881 |
| E | E | 802 |
| N，来源明确 | SIMI | 370 |
| N，来源明确 | REL | 97 |
| N，来源明确 | OPPO | 8 |
| N，来源有歧义 | 不算辅助损失，仍算主损失 | 3 |

共3042行，其中3039行有辅助标签。不会删除歧义行或据此改官方标签。N必须有同方向的直接来源匹配；反方向源记录只用于排除同文本对的关系歧义，不生成反向N。

使用的是 `data/phrasis_sources/PhrasIS.train.all.txt` 和 `PhrasIS.test.all.txt`。其中`test`指历史PhrasIS语料分区，**不是DiCo隐藏测试集**。程序只给传入的Train行做匹配，不把其他源文本加入训练，不读取旧的含Dev分析CSV来生成标签。源关系也不等于经过本次人工复核的语义真值。

这属于“无新增文本、增加来源标注”的实验。论文需披露原始PhrasIS额外标注和源文件指纹；不能写成完全没有额外标注。主实验仍使用官方Dev选模，因此不是新的独立测试集。Train本身已有的Train/Dev重合不会因这个脚本自动消失。

只准备标签、不加载模型或训练：

```powershell
python 'experiments/deberta_lexical_source_aux/train_deberta_source_aux.py' --prepare-only
```

结果在 `outputs/prepared_official_source/`，其中`training_aux_labels.csv`列出每行辅助标签、跳过原因、源文件行号；`source_audit.json`保存源文件SHA-256。只用Train内部验证可传`--validation internal`，复用原来源/重复文本对分组划分，只给内部训练部分准备辅助标签，不给留出部分准备训练目标。

## 保存、评估和比较

默认权重：`model/deberta_lexical_source_aux/run01_source_a0p1_official/seed42/`。

默认结果：`outputs/run01_source_a0p1_official/seed42/`。

- 只保留`best_independent.pth`与`best_pair_aware.pth`两份最佳权重，不保存每轮模型。独立最佳在两个阶段选择，结构最佳只在Pair后选择；每份JSON同时给出同一检查点的两种F1。
- `epoch_summary.csv`分开记录主distance、普通CE、margin新增项和辅助损失。损失日志按批平均，不能直接当整轮按样本加权均值。
- `training_data.csv`是原始训练快照；`training_aux_labels.csv`单独保存额外标签，不能把它写成新增训练样本。
- `stage_parameters.csv`列出source_aux分支，Pair阶段应为False；`pair_changes.csv`及分支logits保持原记录方式。
- 跑完在数据、初始化及主要训练条件匹配时，自动生成`comparison_with_distance.csv`；读已有相同seed的distance结果，不重训基线。按相同selected_by比较，不拼接不同轮次的最高分。
- [evaluate_best.py](evaluate_best.py)重评独立最佳；`--kind pair_aware`重评结构最佳。重载只需要保存的模型目录和验证快照，不需要源标签；预测不调用辅助头。

重复实验改[config.py](config.py)的RUN_NAME，不覆盖旧结果。暂时只跑默认seed42。

## 可选对照，默认不自动运行

`--aux-weight 0`完全跳过辅助头计算和损失，恢复distance训练路径；保存到a0的独立目录。CPU测试已核对两轮小模型主参数与distance逐项完全一致。

如果正式实验有提升，再考虑`--aux-mode shuffled`：仅在Train中打乱已知N子类，保留各类数量与缺失掩码。它用于检查收益是否来自真实细分类信息，而不是新增头或额外CE。默认不启动这个对照，不需要一次跑三份模型。

## 检查与适用范围

[check_setup.py](check_setup.py)只在CPU用随机小DeBERTa检查数据、梯度、单次前向、零权重对照、全未知辅助批次、Pair冻结和保存重载，不加载真实预训练权重，不执行正式训练。

该实验检验原始关系监督是否改善独立泛化。它不能自动修复源标签疑点、缺失语境或保证超过0.9306；也不同于以前Qwen直接改成六类后合并输出的实验。预测仍由一个模型的四分类头完成。
