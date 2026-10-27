# 现有模型多模型预测比较：16组方案的完整说明

整理日期：2026-10-26。这里把历史目录简称展开为实际模型、训练数据和训练方式。成绩沿用已核验的冻结预测，不重新训练或重新挑选模型。

**本表是历史预测CSV的离线融合成绩，不是重新训练后的成绩，也不能据此确认旧权重仍在。** S01、S11、S12、S14、S16的新入口已改为“重训各成员→各存最佳权重→加载新权重预测→融合”，代码位于项目的 `experiments/ensemble_selected/`。新权重保存到 `D:/model/SEAMEVAL/ensemble_selected/run01_retrain/`；不同方案共用的成员在同一RUN_NAME内只训练一次。新的成绩以这些入口实际运行后的结果为准。

已对本报告全部16组固定组合完成“原选模规则 vs 全部按pair-aware选模”的历史逐轮CSV复算：10组不变、1组提高、5组下降。[完整选模比较](../RESULTS.md)。S16历史成绩保持0.954759，其训练入口已按用户要求统一为pair-aware选模，并改用新的 `run02_pair_select`，避免与原混合选模的重训记录混用；其他入口保留原选模规则。

## 先说明“一个模型”和“一组方案”

一个成员是某次训练选定的一份权重，例如“DeBERTa＋对齐＋词汇、seed7”。一组方案让若干成员各自预测同一个Dev，再合并分数。成员可以共用同一个预训练骨干，只是数据、训练方式或随机种子不同。
**各成员的训练数据是分别使用的，不是把下表所有行数加起来，再共同训练一个新模型。**
本报告的97份分数与同一官方660行Dev按instance_id对齐并去重；16组涉及35个不同成员。S16来自严格限定官方Train或复制样本的补充搜索，实际六个成员均只用官方3042行。所有方案均不强制单例判N，不按每行金标选择模型。
独立F1表示不使用文件分组解码；它不保证每个成员内部只做一次编码，例如双向联合头内部会编码两个方向。

## 16组结果总表

| 编号 | 融合方案 | 实际模型数 | independent F1 | pair-aware F1 | N正确/106 | FBE正确/554 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [S01](#s01) | 普通平均：最高结构分 | 6 | 0.773545 | 0.959401 | 99 | 534 |
| [S02](#s02) | 普通平均：最高独立分 | 4 | 0.888548 | 0.927560 | 81 | 532 |
| [S03](#s03) | 普通平均：独立≥0.87且结构≥0.94 | 5 | 0.870753 | 0.943725 | 89 | 534 |
| [S04](#s04) | 普通平均：最佳双模型结构分 | 2 | 0.507521 | 0.947468 | 99 | 526 |
| [S05](#s05) | 普通平均：最佳三模型结构分 | 3 | 0.482678 | 0.954969 | 102 | 528 |
| [S06](#s06) | 普通平均：官方3042行最高结构分 | 4 | 0.599192 | 0.948720 | 102 | 524 |
| [S07](#s07) | 普通平均：官方3042行最高独立分 | 5 | 0.875189 | 0.914436 | 82 | 522 |
| [S08](#s08) | 普通平均：不用large的最高独立分 | 3 | 0.880337 | 0.923292 | 82 | 528 |
| [S09](#s09) | 普通平均：包含Qwen的最高独立分 | 6 | 0.886378 | 0.924826 | 83 | 528 |
| [S10](#s10) | 普通平均：包含ModernBERT的最高独立分 | 5 | 0.883736 | 0.930859 | 83 | 532 |
| [S11](#s11) | 几何平均：最高结构分 | 5 | 0.548800 | 0.960847 | 100 | 534 |
| [S12](#s12) | 几何平均：两项目标同时达到 | 6 | 0.870621 | 0.941924 | 90 | 532 |
| [S13](#s13) | 几何平均：官方3042行最高结构分 | 4 | 0.717115 | 0.954786 | 96 | 534 |
| [S14](#s14) | 分层融合：0.938作为N门 | 6 | 0.545861 | 0.965418 | 101 | 536 |
| [S15](#s15) | 分层融合：0.9306作为N门 | 6 | 0.486196 | 0.962259 | 99 | 536 |
| [S16](#s16) | 仅官方3042行：分层融合，最高结构分 | 6 | 0.481308 | 0.954759 | 98 | 532 |

## 三种融合方法

普通平均：每个成员先对F/B/E/N四类做softmax，再等权平均，p=Σp_m/K。几何平均：对四类log-prob等权平均，再归一化，等价于归一化的概率几何均值。
分层融合：指定一个模型提供g=p(N)；方向成员各自仅在F/B/E内归一化，等权平均得到q。最终p(N)=g，p(F/B/E)=(1-g)q。N门也计入实际模型数；若N门同时作为方向成员，只算一个实际模型。
所有方法都先得到一份固定融合分数：independent取四类最大值；pair-aware对真实两行组比较FB/BF/EE，单例保留该融合模型的四类判断。因此每行表格的两项F1来自同一方案，不是拼两个不同方案的最高分。

## 数据包对照：数字分别是什么意思

下表是**训练文本行数**，不是独立源句数量，也不是所有成员数据的并集大小。类别顺序为F/B/E/N。

| 数据编号 | 数据包 | 行数构成 | 实际总行数 | 类别数量F/B/E/N |
| --- | --- | --- | ---: | --- |
| `official` | 官方Train | 3042 | 3042 | 881 / 881 / 802 / 478 |
| `scope` | 官方Train＋scope200 | 3042＋200＝3242 | 3242 | 921 / 921 / 882 / 518 |
| `relation` | 官方Train＋scope200＋关系48 | 3042＋200＋48＝3290 | 3290 | 933 / 933 / 894 / 530 |
| `closure` | 官方Train＋E改写88 | 3042＋88＝3130 | 3130 | 881 / 881 / 890 / 478 |
| `manual80` | scope底座＋早期人工80 | 3042＋200＋80＝3322 | 3322 | 941 / 941 / 902 / 538 |
| `event200` | scope底座＋动作事件200 | 3042＋200＋200＝3442 | 3442 | 961 / 961 / 962 / 558 |
| `local_manual` | 3290底座＋人工局部对照128 | 3042＋200＋48＋128＝3418 | 3418 | 965 / 965 / 926 / 562 |
| `local_copy` | 3290底座＋复制官方Train128 | 3042＋200＋48＋复制128＝3418 | 3418 | 965 / 965 / 926 / 562 |
| `semantic64` | 3290底座＋语义对照64 | 3042＋200＋48＋64＝3354 | 3354 | 949 / 949 / 910 / 546 |
| `qwenC` | Qwen C标签数据＋解释辅助任务 | 3042＋筛选后的反向N475＝3517 | 3517 | 881 / 881 / 802 / 953 |

### 数据包 `official`：官方Train

仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。

实际训练CSV：[dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)

### 数据包 `scope`：官方Train＋scope200

200行范围/修饰与结构对照：属性限制、同义结构改写、中心对象替换、并列范围和顺序等。是历史人工模板草案，不是官方额外金标。

实际训练CSV：[training_data.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/training_data.csv)

### 数据包 `relation`：官方Train＋scope200＋关系48

在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)

### 数据包 `closure`：官方Train＋E改写88

直接核对CSV：原3042行保留，新增88行全部标E，组成44个双向组。例如文件将play/plays、is holding/holding标E。closure只是旧文件名；当前未找到完整生成脚本，不能仅凭名字断言88行全由经过核验的Train传递闭包推导。

实际训练CSV：[closure88_train.csv](../../../../../deberta_closure/closure88_train.csv)

### 数据包 `manual80`：scope底座＋早期人工80

80行早期人工对照，涉及修饰限制、上下位、词汇/拼写等价、属性独立、中心词替换、事件状态等；不是后来的relation48。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run01_original_manual/training_data.csv)

### 数据包 `event200`：scope底座＋动作事件200

额外200行为20个动作/事件家族，每家族10行。包括动作具体化、同义改写、事件阶段/状态差异；这200行F40、B40、E80、N40。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run03_event_200/training_data.csv)

### 数据包 `local_manual`：3290底座＋人工局部对照128

额外128行F/B/E/N各32。16个训练家族，类型为名称/品牌边界、用途与地点、路径与行为者、角色限制。另有64行诊断数据不进入训练或选模。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run11_local128_manual/training_data.csv)

### 数据包 `local_copy`：3290底座＋复制官方Train128

复制128行已有官方Train样本，F/B/E/N各32；不是把人工local128加进去，也不代表新增128条独立语义知识。copy_control_sources.csv给出了原始ID和新副本ID。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run12_local128_copy/training_data.csv)

### 数据包 `semantic64`：3290底座＋语义对照64

额外64行F/B/E/N各16，分food_activity、aircraft_scope、modifier_scope、event_roles四个家族，各16行；官方Train词汇重组成新对照，涉及论元、上下位、修饰范围、事件角色及状态。

实际训练CSV：[training_data.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv)

### 数据包 `qwenC`：Qwen C标签数据＋解释辅助任务

3517行都训练标签；其中2040行训练解释，包括1756行官方来源和284行反向N来源。解释是额外目标，不是再追加2040行标签文本，也没有把Dev理由输入模型。

实际训练CSV：[train_C_with_reasons.csv](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/prepared/train_C_with_reasons.csv)

### 容易混淆的几个名字

- `closure88`：实际是官方3042＋88行E标签改写，不应只看文件夹名就解释成已证实的传递闭包算法。新增88行与Dev的归一化无序文本对精确交集为0，但这不是来源独立性证明。
- `local128_manual`：增加128行新造的局部对照；`local128_copy`：增加128行官方Train副本，两者都从3290行底座出发。
- `event200`：从3242行scope底座加200行动作/事件例子，总计3442，不包含relation48。
- `semantic64`：从3290行底座加64行语义对照，总计3354；64行来自四个家族，仍采用先Row后Pair训练。
- “词汇版”指手工20维Lexical特征分支，不是WordNet词表，也不是额外NLI预训练。
- DeBERTa的reason/evidence版训练token证据位置；Qwen的reason版训练生成解释，两者不是同一种辅助任务。

## 逐组解释与完整成员

<a id="s01"></a>
### S01：普通平均：最高结构分

本组共6个实际模型。独立F1=0.773545，pair-aware F1=0.959401。N对99/106，配对F/B/E对534/554。

操作：下面6个成员各预测一次，做四类概率普通平均；每个成员权重均为1/6。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M014 | DeBERTa-base＋词汇＋E改写88 | 四类共同投票 | 3042＋88＝3130 | 3130 | 42 | 3 | 0.316835 / 0.922322 |
| M065 | 0.938数据＋复制Train128 | 四类共同投票 | 3042＋200＋48＋复制128＝3418 | 3418 | 42 | 3 | 0.522136 / 0.920146 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M075 | 0.938数据＋语义对照64，两阶段 | 四类共同投票 | 3042＋200＋48＋64＝3354 | 3354 | 42 | 4 | 0.464267 / 0.928647 |
| M080 | 0.938数据＋仅Row阶段SAM | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 7 | 0.782118 / 0.929029 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |

#### M014：DeBERTa-base＋词汇＋E改写88

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原CLS四分类头＋Token Alignment＋Lexical；closure88是历史文件名。
- **训练文本：** 3042＋88＝3130。直接核对CSV：原3042行保留，新增88行全部标E，组成44个双向组。例如文件将play/plays、is holding/holding标E。closure只是旧文件名；当前未找到完整生成脚本，不能仅凭名字断言88行全由经过核验的Train传递闭包推导。
- **训练方式：** 全参数两阶段；注意文件的data_augmentation=false字段与实际3130行不一致，实际增加了88行，本文按CSV记录。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.316835、结构F1=0.922322。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_closure/outputs/run01_closure88/lexical_seed_42/run_config.json)。
- **训练数据文件：** [closure88_train.csv](../../../../../deberta_closure/closure88_train.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_closure/outputs/run01_closure88/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增88行全为E；旧data_augmentation=false不能据此解释为无增强。新88行与Dev归一化无序整对精确交集为0；这不证明来源独立或与Dev词汇完全隔离。

#### M065：0.938数据＋复制Train128

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与M064相同的CLS＋Alignment，没有Lexical。
- **训练文本：** 3042＋200＋48＋复制128＝3418。复制128行已有官方Train样本，F/B/E/N各32；不是把人工local128加进去，也不代表新增128条独立语义知识。copy_control_sources.csv给出了原始ID和新副本ID。
- **训练方式：** 全参数两阶段；这是M064的复制对照，实际加入的是官方Train副本，目的在匹配新增行数、类别和Pair预算。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.522136、结构F1=0.920146。
- **历史选模规则：** `best Dev pair-aware weighted F1; independent recorded at the same epoch`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run12_local128_copy/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run12_local128_copy/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run12_local128_copy/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 这是复制对照，不是人工128版。副本来自Train，不能把人工样本的Dev词汇隔离结论套到这些副本上。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M075：0.938数据＋语义对照64，两阶段

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原DeBERTa CLS＋Alignment；没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＋64＝3354。额外64行F/B/E/N各16，分food_activity、aircraft_scope、modifier_scope、event_roles四个家族，各16行；官方Train词汇重组成新对照，涉及论元、上下位、修饰范围、事件角色及状态。
- **训练方式：** 全参数两阶段；开启动态padding和梯度检查点，因而训练轨迹也与历史固定padding的0.938有差别。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第4轮；单模型独立F1=0.464267、结构F1=0.928647。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M080：0.938数据＋仅Row阶段SAM

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原DeBERTa CLS＋Alignment，没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 先全部Row-SAM，rho=0.05，再普通0.5 Pair CE；只有Row阶段做参数扰动并二次前后向。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.782118、结构F1=0.929029。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run26_row_sam/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run26_row_sam/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好18行、破坏4行。
本组融合分数和逐行预测：[S01_predictions.csv](../../S01_predictions.csv)。

<a id="s02"></a>
### S02：普通平均：最高独立分

本组共4个实际模型。独立F1=0.888548，pair-aware F1=0.927560。N对81/106，配对F/B/E对532/554。

操作：下面4个成员各预测一次，做四类概率普通平均；每个成员权重均为1/4。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M008 | DeBERTa-base双向联合头LoRA，seed7 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 7 | 9 | 0.850378 / 0.898961 |
| M046 | 词汇版，Row CE＋反向KL | 四类共同投票 | 3042 | 3042 | 42 | 2 | 0.860056 / 0.886018 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |

#### M008：DeBERTa-base双向联合头LoRA，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 共享DeBERTa分别编码两个输入方向，再用联合四分类头；本版没有原Token Alignment分支，也没有Lexical。编码器用LoRA适配。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 每条原始行计算四分类CE，没有单独Pair CE；LoRA r16/alpha32/dropout0.1，作用于query_proj/value_proj。
- **主要参数：** seed=7；主学习率=0.0001；计划20轮；max_length=128；文本对单元batch=8；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]。
- **本次采用版本：** 第9轮；单模型独立F1=0.850378、结构F1=0.898961。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M046：词汇版，Row CE＋反向KL

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，不使用NLI pooler。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合组批；保留四类Row CE，用0.5×四类正反对称KL替代Pair CE。只在真实可逆组对齐F/B，没有另造N反向监督。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；反向KL权重=0.5。
- **本次采用版本：** 第2轮；单模型独立F1=0.860056、结构F1=0.886018。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好14行、破坏20行。
本组融合分数和逐行预测：[S02_predictions.csv](../../S02_predictions.csv)。

<a id="s03"></a>
### S03：普通平均：独立≥0.87且结构≥0.94

本组共5个实际模型。独立F1=0.870753，pair-aware F1=0.943725。N对89/106，配对F/B/E对534/554。

操作：下面5个成员各预测一次，做四类概率普通平均；每个成员权重均为1/5。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M056 | scope200＋动作事件200 | 四类共同投票 | 3042＋200＋200＝3442 | 3442 | 42 | 4 | 0.664316 / 0.924949 |
| M064 | 0.938数据＋人工局部对照128 | 四类共同投票 | 3042＋200＋48＋128＝3418 | 3418 | 42 | 6 | 0.825607 / 0.902142 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M072 | DeBERTa LoRA r16/dropout0，seed7 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 7 | 5 | 0.839874 / 0.904756 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |

#### M056：scope200＋动作事件200

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＋200＝3442。额外200行为20个动作/事件家族，每家族10行。包括动作具体化、同义改写、事件阶段/状态差异；这200行F40、B40、E80、N40。
- **训练方式：** 全参数两阶段；事件/动作200是额外训练数据，不是模型分支。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第4轮；单模型独立F1=0.664316、结构F1=0.924949。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run03_event_200/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run03_event_200/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run03_event_200/epoch_4/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M064：0.938数据＋人工局部对照128

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS＋Alignment；没有Lexical。
- **训练文本：** 3042＋200＋48＋128＝3418。额外128行F/B/E/N各32。16个训练家族，类型为名称/品牌边界、用途与地点、路径与行为者、角色限制。另有64行诊断数据不进入训练或选模。
- **训练方式：** 全参数两阶段；从NLI预训练底座重新开始，不是继续训练已选中的0.938权重。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第6轮；单模型独立F1=0.825607、结构F1=0.902142。
- **历史选模规则：** `best Dev pair-aware weighted F1; independent recorded at the same epoch`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run11_local128_manual/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run11_local128_manual/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run11_local128_manual/epoch_6/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M072：DeBERTa LoRA r16/dropout0，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与M071同类的DeBERTa LoRA＋CLS＋Alignment，没有Lexical。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；r16/alpha32/dropout0，seed7。保存配置的comparison_note沿用了旧文案，本文以lora_dropout实际数值0为准。
- **主要参数：** seed=7；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第5轮；单模型独立F1=0.839874、结构F1=0.904756。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run18_lora_r16_seed7_dropout0/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run18_lora_r16_seed7_dropout0/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 保存配置中comparison_note有沿用旧实验的文字；实际lora.lora_dropout=0.0且seed=7。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好16行、破坏12行。
本组融合分数和逐行预测：[S03_predictions.csv](../../S03_predictions.csv)。

<a id="s04"></a>
### S04：普通平均：最佳双模型结构分

本组共2个实际模型。独立F1=0.507521，pair-aware F1=0.947468。N对99/106，配对F/B/E对526/554。

操作：下面2个成员各预测一次，做四类概率普通平均；每个成员权重均为1/2。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M081 | 0.938数据＋Row/Pair双阶段SAM | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 7 | 0.699295 / 0.928084 |
| M084 | 原0.9355版：仅加scope200 | 四类共同投票 | 3042＋200＝3242 | 3242 | 42 | 3 | 0.480876 / 0.935493 |

#### M081：0.938数据＋Row/Pair双阶段SAM

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与M080相同的原CLS＋Alignment，没有Lexical。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** Row和Pair阶段都使用SAM，二者rho均0.05；仍是先Row再Pair，不是混合训练。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.699295、结构F1=0.928084。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run27_row_pair_sam/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run27_row_pair_sam/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M084：原0.9355版：仅加scope200

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＝3242。200行范围/修饰与结构对照：属性限制、同义结构改写、中心对象替换、并列范围和顺序等。是历史人工模板草案，不是官方额外金标。
- **训练方式：** 全参数两阶段；与官方3042行原版相比，训练文本新增scope200。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.480876、结构F1=0.935493。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_original_aug/outputs/run01_original_scope/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

本组结构预测相对原0.938：修好15行、破坏9行。
本组融合分数和逐行预测：[S04_predictions.csv](../../S04_predictions.csv)。

<a id="s05"></a>
### S05：普通平均：最佳三模型结构分

本组共3个实际模型。独立F1=0.482678，pair-aware F1=0.954969。N对102/106，配对F/B/E对528/554。

操作：下面3个成员各预测一次，做四类概率普通平均；每个成员权重均为1/3。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M014 | DeBERTa-base＋词汇＋E改写88 | 四类共同投票 | 3042＋88＝3130 | 3130 | 42 | 3 | 0.316835 / 0.922322 |
| M075 | 0.938数据＋语义对照64，两阶段 | 四类共同投票 | 3042＋200＋48＋64＝3354 | 3354 | 42 | 4 | 0.464267 / 0.928647 |
| M080 | 0.938数据＋仅Row阶段SAM | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 7 | 0.782118 / 0.929029 |

#### M014：DeBERTa-base＋词汇＋E改写88

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原CLS四分类头＋Token Alignment＋Lexical；closure88是历史文件名。
- **训练文本：** 3042＋88＝3130。直接核对CSV：原3042行保留，新增88行全部标E，组成44个双向组。例如文件将play/plays、is holding/holding标E。closure只是旧文件名；当前未找到完整生成脚本，不能仅凭名字断言88行全由经过核验的Train传递闭包推导。
- **训练方式：** 全参数两阶段；注意文件的data_augmentation=false字段与实际3130行不一致，实际增加了88行，本文按CSV记录。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.316835、结构F1=0.922322。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_closure/outputs/run01_closure88/lexical_seed_42/run_config.json)。
- **训练数据文件：** [closure88_train.csv](../../../../../deberta_closure/closure88_train.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_closure/outputs/run01_closure88/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增88行全为E；旧data_augmentation=false不能据此解释为无增强。新88行与Dev归一化无序整对精确交集为0；这不证明来源独立或与Dev词汇完全隔离。

#### M075：0.938数据＋语义对照64，两阶段

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原DeBERTa CLS＋Alignment；没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＋64＝3354。额外64行F/B/E/N各16，分food_activity、aircraft_scope、modifier_scope、event_roles四个家族，各16行；官方Train词汇重组成新对照，涉及论元、上下位、修饰范围、事件角色及状态。
- **训练方式：** 全参数两阶段；开启动态padding和梯度检查点，因而训练轨迹也与历史固定padding的0.938有差别。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第4轮；单模型独立F1=0.464267、结构F1=0.928647。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M080：0.938数据＋仅Row阶段SAM

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原DeBERTa CLS＋Alignment，没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 先全部Row-SAM，rho=0.05，再普通0.5 Pair CE；只有Row阶段做参数扰动并二次前后向。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.782118、结构F1=0.929029。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run26_row_sam/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run26_row_sam/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

本组结构预测相对原0.938：修好12行、破坏1行。
本组融合分数和逐行预测：[S05_predictions.csv](../../S05_predictions.csv)。

<a id="s06"></a>
### S06：普通平均：官方3042行最高结构分

本组共4个实际模型。独立F1=0.599192，pair-aware F1=0.948720。N对102/106，配对F/B/E对524/554。

操作：下面4个成员各预测一次，做四类概率普通平均；每个成员权重均为1/4。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M007 | 原始DeBERTa-base＋对齐＋词汇 | 四类共同投票 | 3042 | 3042 | 42 | 3 | 0.496812 / 0.910326 |
| M028 | 词汇版，类别全部等权 | 四类共同投票 | 3042 | 3042 | 42 | 3 | 0.287174 / 0.912551 |
| M048 | 词汇版＋全Train证据监督 | 四类共同投票 | 3042 | 3042 | 42 | 7 | 0.857297 / 0.898119 |
| M050 | 无Lexical的原结构对照，seed7 | 四类共同投票 | 3042 | 3042 | 7 | 6 | 0.424967 / 0.912365 |

#### M007：原始DeBERTa-base＋对齐＋词汇

- **基础模型：** `microsoft/deberta-v3-base`。
- **结构：** microsoft原始预训练底座，没有MNLI/ANLI任务微调；自建CLS四分类头＋Token Alignment＋20维Lexical修正分支。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数、两阶段：先Row CE，再0.5×三组合Pair CE。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.496812、结构F1=0.910326。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M028：词汇版，类别全部等权

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与原词汇版相同的CLS＋Alignment＋Lexical结构。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Row权重[1,1,1,1]、Pair权重[1,1,1]，与0.9306版的权重设置不同。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.0]；Pair权重FB/BF/EE=[1.0, 1.0, 1.0]。
- **本次采用版本：** 第3轮；单模型独立F1=0.287174、结构F1=0.912551。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run03/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run03/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** [best_model.pth](../../../../../../model/deberta_lexical/run03/lexical_seed_42/best_model.pth)。

#### M048：词汇版＋全Train证据监督

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与M047相同的主分类结构和训练期证据头。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合Row CE＋0.5 Pair CE＋0.1 token CE；3042行都有理由审阅标注，1756行supported用于证据监督，其他行仍保留原四分类监督。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.857297、结构F1=0.898119。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run09_reason_all_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run09_reason_all_seed42/epoch_7/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** reason在这里指Train理由标注转换成token证据监督；与Qwen生成完整解释的SFT不同。标注由助手制作，未独立人工复核。

#### M050：无Lexical的原结构对照，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；用于换seed检验的无词汇对照，不是seed42的0.9248权重。
- **主要参数：** seed=7；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第6轮；单模型独立F1=0.424967、结构F1=0.912365。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/epoch_6/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好17行、破坏10行。
本组融合分数和逐行预测：[S06_predictions.csv](../../S06_predictions.csv)。

<a id="s07"></a>
### S07：普通平均：官方3042行最高独立分

本组共5个实际模型。独立F1=0.875189，pair-aware F1=0.914436。N对82/106，配对F/B/E对522/554。

操作：下面5个成员各预测一次，做四类概率普通平均；每个成员权重均为1/5。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M030 | 词汇版，N行权重1.8 | 四类共同投票 | 3042 | 3042 | 42 | 9 | 0.704990 / 0.919490 |
| M045 | 词汇混合训练＋OOF难例加权 | 四类共同投票 | 3042 | 3042 | 42 | 7 | 0.852621 / 0.892869 |
| M046 | 词汇版，Row CE＋反向KL | 四类共同投票 | 3042 | 3042 | 42 | 2 | 0.860056 / 0.886018 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M083 | DeBERTa-xsmall＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 7 | 0.832263 / 0.865908 |

#### M030：词汇版，N行权重1.8

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与原词汇版相同的CLS＋Alignment＋Lexical结构。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Row的N权重改为1.8，Pair仍为[1,1,1.25]。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.8]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第9轮；单模型独立F1=0.704990、结构F1=0.919490。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run04_weight/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run04_weight/lexical_seed_42/epoch_9/predictions_with_scores.csv)。
- **已有权重：** [best_model.pth](../../../../../../model/deberta_lexical/run04_weight/lexical_seed_42/best_model.pth)。

#### M045：词汇混合训练＋OOF难例加权

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical；不增加文本行。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 用Train内部OOF预测选每类难例，hard_fraction=0.25、倍率2；按类归一化权重，只修改Row CE样本权重，Pair CE不改。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.852621、结构F1=0.892869。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run06_oof_hard/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run06_oof_hard/epoch_7/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M046：词汇版，Row CE＋反向KL

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，不使用NLI pooler。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合组批；保留四类Row CE，用0.5×四类正反对称KL替代Pair CE。只在真实可逆组对齐F/B，没有另造N反向监督。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；反向KL权重=0.5。
- **本次采用版本：** 第2轮；单模型独立F1=0.860056、结构F1=0.886018。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M083：DeBERTa-xsmall＋对齐＋词汇混合训练

- **基础模型：** `cross-encoder/nli-deberta-v3-xsmall`。
- **结构：** cross-encoder/nli-deberta-v3-xsmall，hidden_size384、12层；CLS＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合Row CE＋0.5 Pair CE，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.832263、结构F1=0.865908。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_xsmall/outputs/run01_xsmall_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_xsmall/outputs/run01_xsmall_mixed_pair/epoch_7/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好12行、破坏27行。
本组融合分数和逐行预测：[S07_predictions.csv](../../S07_predictions.csv)。

<a id="s08"></a>
### S08：普通平均：不用large的最高独立分

本组共3个实际模型。独立F1=0.880337，pair-aware F1=0.923292。N对82/106，配对F/B/E对528/554。

操作：下面3个成员各预测一次，做四类概率普通平均；每个成员权重均为1/3。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M008 | DeBERTa-base双向联合头LoRA，seed7 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 7 | 9 | 0.850378 / 0.898961 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M083 | DeBERTa-xsmall＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 7 | 0.832263 / 0.865908 |

#### M008：DeBERTa-base双向联合头LoRA，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 共享DeBERTa分别编码两个输入方向，再用联合四分类头；本版没有原Token Alignment分支，也没有Lexical。编码器用LoRA适配。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 每条原始行计算四分类CE，没有单独Pair CE；LoRA r16/alpha32/dropout0.1，作用于query_proj/value_proj。
- **主要参数：** seed=7；主学习率=0.0001；计划20轮；max_length=128；文本对单元batch=8；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]。
- **本次采用版本：** 第9轮；单模型独立F1=0.850378、结构F1=0.898961。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M083：DeBERTa-xsmall＋对齐＋词汇混合训练

- **基础模型：** `cross-encoder/nli-deberta-v3-xsmall`。
- **结构：** cross-encoder/nli-deberta-v3-xsmall，hidden_size384、12层；CLS＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合Row CE＋0.5 Pair CE，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.832263、结构F1=0.865908。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_xsmall/outputs/run01_xsmall_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_xsmall/outputs/run01_xsmall_mixed_pair/epoch_7/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好12行、破坏21行。
本组融合分数和逐行预测：[S08_predictions.csv](../../S08_predictions.csv)。

<a id="s09"></a>
### S09：普通平均：包含Qwen的最高独立分

本组共6个实际模型。独立F1=0.886378，pair-aware F1=0.924826。N对83/106，配对F/B/E对528/554。

操作：下面6个成员各预测一次，做四类概率普通平均；每个成员权重均为1/6。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M008 | DeBERTa-base双向联合头LoRA，seed7 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 7 | 9 | 0.850378 / 0.898961 |
| M051 | 词汇版换seed7 | 四类共同投票 | 3042 | 3042 | 7 | 1 | 0.790963 / 0.887611 |
| M060 | 0.938同数据，Row/Pair混合训练 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 2 | 0.823173 / 0.871319 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M094 | Qwen2.5-3B-Instruct，C＋reason | 四类共同投票 | 3042＋筛选后的反向N475＝3517 | 3517 | 42 | 4 | 0.838121 / 0.902142 |

#### M008：DeBERTa-base双向联合头LoRA，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 共享DeBERTa分别编码两个输入方向，再用联合四分类头；本版没有原Token Alignment分支，也没有Lexical。编码器用LoRA适配。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 每条原始行计算四分类CE，没有单独Pair CE；LoRA r16/alpha32/dropout0.1，作用于query_proj/value_proj。
- **主要参数：** seed=7；主学习率=0.0001；计划20轮；max_length=128；文本对单元batch=8；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]。
- **本次采用版本：** 第9轮；单模型独立F1=0.850378、结构F1=0.898961。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M051：词汇版换seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段，seed7；属于可靠性实验，不是新的预训练模型。
- **主要参数：** seed=7；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第1轮；单模型独立F1=0.790963、结构F1=0.887611。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_reliability/outputs/run01/lexical_seed_7/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_reliability/outputs/run01/lexical_seed_7/epoch_1/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M060：0.938同数据，Row/Pair混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与M059相同的CLS＋Alignment，没有Lexical。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 每批按真实配对组和N单例混合，单次前向中算Row CE＋0.5 Pair CE；改变了组批、更新步数和前向暴露，不只是改shuffle顺序。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.823173、结构F1=0.871319。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run07_relation48_mixed/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run07_relation48_mixed/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M094：Qwen2.5-3B-Instruct，C＋reason

- **基础模型：** `D:/LLM/model/Qwen2.5-3B-Instruct`。
- **结构：** 生成式Qwen，LoRA r16/alpha32/dropout0.05、all-linear；没有DeBERTa Token Alignment和Lexical。
- **训练文本：** 3042＋筛选后的反向N475＝3517。3517行都训练标签；其中2040行训练解释，包括1756行官方来源和284行反向N来源。解释是额外目标，不是再追加2040行标签文本，也没有把Dev理由输入模型。
- **训练方式：** 标签SFT＋0.2解释SFT，两个独立聊天Prompt、分别按监督token归一化；不使用Pair CE，预测只问标签，不提供Dev解释。
- **主要参数：** seed=42；lr=0.0001；微批次4×累积4；计划5轮；bf16；r=16、alpha=32、dropout=0.05；reason权重=0.2。
- **本次采用版本：** 第4轮；单模型独立F1=0.838121、结构F1=0.902142。
- **历史选模规则：** `pair_aware`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/run_config.json)。
- **训练数据文件：** [train_C_with_reasons.csv](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/prepared/train_C_with_reasons.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/best/predictions_with_scores.csv)。
- **已有权重：** adapter_model.safetensors（本机文件，不随仓库提供）。
- **已有权重：** adapter_model.safetensors（本机文件，不随仓库提供）。
- **需保留的说明：** 融合选用训练结束回载best_adapter后的缓存：独立0.838121、结构0.902142；没有使用训练期epoch4较高的0.843581/0.905397。此前回调batch16与结束重评batch8不同，代码后来已统一。

本组结构预测相对原0.938：修好12行、破坏20行。
本组融合分数和逐行预测：[S09_predictions.csv](../../S09_predictions.csv)。

<a id="s10"></a>
### S10：普通平均：包含ModernBERT的最高独立分

本组共5个实际模型。独立F1=0.883736，pair-aware F1=0.930859。N对83/106，配对F/B/E对532/554。

操作：下面5个成员各预测一次，做四类概率普通平均；每个成员权重均为1/5。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M008 | DeBERTa-base双向联合头LoRA，seed7 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 7 | 9 | 0.850378 / 0.898961 |
| M046 | 词汇版，Row CE＋反向KL | 四类共同投票 | 3042 | 3042 | 42 | 2 | 0.860056 / 0.886018 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M093 | ModernBERT-base＋原对齐配方 | 四类共同投票 | 3042 | 3042 | 42 | 2 | 0.642604 / 0.866803 |

#### M008：DeBERTa-base双向联合头LoRA，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 共享DeBERTa分别编码两个输入方向，再用联合四分类头；本版没有原Token Alignment分支，也没有Lexical。编码器用LoRA适配。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 每条原始行计算四分类CE，没有单独Pair CE；LoRA r16/alpha32/dropout0.1，作用于query_proj/value_proj。
- **主要参数：** seed=7；主学习率=0.0001；计划20轮；max_length=128；文本对单元batch=8；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]。
- **本次采用版本：** 第9轮；单模型独立F1=0.850378、结构F1=0.898961。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_bidirectional/outputs/run01_bidirectional_seed7/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M046：词汇版，Row CE＋反向KL

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，不使用NLI pooler。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合组批；保留四类Row CE，用0.5×四类正反对称KL替代Pair CE。只在真实可逆组对齐F/B，没有另造N反向监督。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；反向KL权重=0.5。
- **本次采用版本：** 第2轮；单模型独立F1=0.860056、结构F1=0.886018。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M093：ModernBERT-base＋原对齐配方

- **基础模型：** `tasksource/ModernBERT-base-nli`。
- **结构：** tasksource/ModernBERT-base-nli替换DeBERTa，仍用原CLS四分类头＋Token Alignment；不使用原NLI三分类头，没有Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段，SDPA后端，float32；本报告冻结使用盘点时保存的epoch2分数，不自动跟随之后的新运行。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.642604、结构F1=0.866803。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../modernbert_nli_base/outputs/run01_original_recipe/baseline_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../modernbert_nli_base/outputs/run01_original_recipe/baseline_seed_42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好14行、破坏18行。
本组融合分数和逐行预测：[S10_predictions.csv](../../S10_predictions.csv)。

<a id="s11"></a>
### S11：几何平均：最高结构分

本组共5个实际模型。独立F1=0.548800，pair-aware F1=0.960847。N对100/106，配对F/B/E对534/554。

操作：下面5个成员各预测一次，做四类log-prob几何平均；每个成员权重均为1/5。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M017 | CLS、对齐、词汇特征拼接版 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.507145 / 0.916871 |
| M036 | 词汇版＋scope200＋关系48 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 7 | 0.631094 / 0.930670 |
| M050 | 无Lexical的原结构对照，seed7 | 四类共同投票 | 3042 | 3042 | 7 | 6 | 0.424967 / 0.912365 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M084 | 原0.9355版：仅加scope200 | 四类共同投票 | 3042＋200＝3242 | 3242 | 42 | 3 | 0.480876 / 0.935493 |

#### M017：CLS、对齐、词汇特征拼接版

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 归一化CLS、token比较特征及词汇特征拼接，共1312维，经联合分类器输出四类；不是三个logit分支直接相加。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Lexical有单独学习率，Pair后按结构F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.507145、结构F1=0.916871。
- **历史选模规则：** `after Pair CE: best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M036：词汇版＋scope200＋关系48

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 0.938的数据与原CLS/Alignment结构，再加20维Lexical；全参数训练。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 先Row再Pair，与历史0.938使用同一3290行训练快照；未另加局部语义头。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第7轮；单模型独立F1=0.631094、结构F1=0.930670。
- **历史选模规则：** `best Dev pair_aware at after_pair; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_aug/outputs/run02_lexical_relation_staged/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_aug/outputs/run02_lexical_relation_staged/epoch_7/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M050：无Lexical的原结构对照，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；用于换seed检验的无词汇对照，不是seed42的0.9248权重。
- **主要参数：** seed=7；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第6轮；单模型独立F1=0.424967、结构F1=0.912365。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/epoch_6/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M084：原0.9355版：仅加scope200

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＝3242。200行范围/修饰与结构对照：属性限制、同义结构改写、中心对象替换、并列范围和顺序等。是历史人工模板草案，不是官方额外金标。
- **训练方式：** 全参数两阶段；与官方3042行原版相比，训练文本新增scope200。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.480876、结构F1=0.935493。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_original_aug/outputs/run01_original_scope/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

本组结构预测相对原0.938：修好20行、破坏5行。
本组融合分数和逐行预测：[S11_predictions.csv](../../S11_predictions.csv)。

<a id="s12"></a>
### S12：几何平均：两项目标同时达到

本组共6个实际模型。独立F1=0.870621，pair-aware F1=0.941924。N对90/106，配对F/B/E对532/554。

操作：下面6个成员各预测一次，做四类log-prob几何平均；每个成员权重均为1/6。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M007 | 原始DeBERTa-base＋对齐＋词汇 | 四类共同投票 | 3042 | 3042 | 42 | 3 | 0.496812 / 0.910326 |
| M009 | DeBERTa-base双向对齐联合头 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 4 | 0.856822 / 0.906326 |
| M018 | 冻结0.938底座，只训练N门 | 四类共同投票 | 3042＋200＋48＝3290 | 3290 | 42 | 4 | 0.825260 / 0.906702 |
| M064 | 0.938数据＋人工局部对照128 | 四类共同投票 | 3042＋200＋48＋128＝3418 | 3418 | 42 | 6 | 0.825607 / 0.902142 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M094 | Qwen2.5-3B-Instruct，C＋reason | 四类共同投票 | 3042＋筛选后的反向N475＝3517 | 3517 | 42 | 4 | 0.838121 / 0.902142 |

#### M007：原始DeBERTa-base＋对齐＋词汇

- **基础模型：** `microsoft/deberta-v3-base`。
- **结构：** microsoft原始预训练底座，没有MNLI/ANLI任务微调；自建CLS四分类头＋Token Alignment＋20维Lexical修正分支。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数、两阶段：先Row CE，再0.5×三组合Pair CE。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.496812、结构F1=0.910326。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_base_lexical/outputs/run01_raw_base/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M009：DeBERTa-base双向对齐联合头

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 两方向都提取原CLS和Token Alignment特征，再送联合四分类头；没有Lexical，不是三个组合分数相加。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 全参数四分类CE，没有独立Pair CE阶段；双向编码和批次组织均与原两阶段版不同。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；文本对单元batch=8；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]。
- **本次采用版本：** 第4轮；单模型独立F1=0.856822、结构F1=0.906326。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties, as original 0.938`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_bidirectional/outputs/run02_original_alignment_joint_seed42/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_bidirectional/outputs/run02_original_alignment_joint_seed42/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M018：冻结0.938底座，只训练N门

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 以已经按Dev选出的0.938权重为冻结底座，保留F/B/E条件判断，只新增并训练N二分类头；没有Lexical。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 修复阶段仅训练n_head，BCE判断N/非N；主干和既有F/B/E输出冻结，没有Pair CE。p(N)=sigmoid(g)，其余概率为(1-p(N))×softmax(F/B/E)。
- **主要参数：** seed=42；主学习率=0.001；计划30轮；max_length=128；训练batch=64。
- **本次采用版本：** 第4轮；单模型独立F1=0.825260、结构F1=0.906702。
- **历史选模规则：** `max min(independent/target_independent, pair_aware/target_pair); earliest tie`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_frozen_n_head/outputs/run01_n_head/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_frozen_n_head/outputs/run01_n_head/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M064：0.938数据＋人工局部对照128

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS＋Alignment；没有Lexical。
- **训练文本：** 3042＋200＋48＋128＝3418。额外128行F/B/E/N各32。16个训练家族，类型为名称/品牌边界、用途与地点、路径与行为者、角色限制。另有64行诊断数据不进入训练或选模。
- **训练方式：** 全参数两阶段；从NLI预训练底座重新开始，不是继续训练已选中的0.938权重。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第6轮；单模型独立F1=0.825607、结构F1=0.902142。
- **历史选模规则：** `best Dev pair-aware weighted F1; independent recorded at the same epoch`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run11_local128_manual/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run11_local128_manual/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run11_local128_manual/epoch_6/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M094：Qwen2.5-3B-Instruct，C＋reason

- **基础模型：** `D:/LLM/model/Qwen2.5-3B-Instruct`。
- **结构：** 生成式Qwen，LoRA r16/alpha32/dropout0.05、all-linear；没有DeBERTa Token Alignment和Lexical。
- **训练文本：** 3042＋筛选后的反向N475＝3517。3517行都训练标签；其中2040行训练解释，包括1756行官方来源和284行反向N来源。解释是额外目标，不是再追加2040行标签文本，也没有把Dev理由输入模型。
- **训练方式：** 标签SFT＋0.2解释SFT，两个独立聊天Prompt、分别按监督token归一化；不使用Pair CE，预测只问标签，不提供Dev解释。
- **主要参数：** seed=42；lr=0.0001；微批次4×累积4；计划5轮；bf16；r=16、alpha=32、dropout=0.05；reason权重=0.2。
- **本次采用版本：** 第4轮；单模型独立F1=0.838121、结构F1=0.902142。
- **历史选模规则：** `pair_aware`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/run_config.json)。
- **训练数据文件：** [train_C_with_reasons.csv](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/prepared/train_C_with_reasons.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../qwen25_3b_reason/outputs/run01_c_reason02/C/best/predictions_with_scores.csv)。
- **已有权重：** adapter_model.safetensors（本机文件，不随仓库提供）。
- **已有权重：** adapter_model.safetensors（本机文件，不随仓库提供）。
- **需保留的说明：** 融合选用训练结束回载best_adapter后的缓存：独立0.838121、结构0.902142；没有使用训练期epoch4较高的0.843581/0.905397。此前回调batch16与结束重评batch8不同，代码后来已统一。

本组结构预测相对原0.938：修好18行、破坏15行。
本组融合分数和逐行预测：[S12_predictions.csv](../../S12_predictions.csv)。

<a id="s13"></a>
### S13：几何平均：官方3042行最高结构分

本组共4个实际模型。独立F1=0.717115，pair-aware F1=0.954786。N对96/106，配对F/B/E对534/554。

操作：下面4个成员各预测一次，做四类log-prob几何平均；每个成员权重均为1/4。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M017 | CLS、对齐、词汇特征拼接版 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.507145 / 0.916871 |
| M047 | 词汇版＋首批100组证据监督 | 四类共同投票 | 3042 | 3042 | 42 | 2 | 0.850874 / 0.872561 |
| M050 | 无Lexical的原结构对照，seed7 | 四类共同投票 | 3042 | 3042 | 7 | 6 | 0.424967 / 0.912365 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | 四类共同投票 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |

#### M017：CLS、对齐、词汇特征拼接版

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 归一化CLS、token比较特征及词汇特征拼接，共1312维，经联合分类器输出四类；不是三个logit分支直接相加。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Lexical有单独学习率，Pair后按结构F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.507145、结构F1=0.916871。
- **历史选模规则：** `after Pair CE: best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M047：词汇版＋首批100组证据监督

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，新增训练期token证据分类头。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合Row CE＋0.5 Pair CE＋0.1 token CE。首批100组标注覆盖175行，仅supported部分参与证据监督；不是把reason文字拼进输入，也不是生成reason。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.850874、结构F1=0.872561。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run08_reason_evidence_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run08_reason_evidence_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** reason在这里指Train理由标注转换成token证据监督；与Qwen生成完整解释的SFT不同。标注由助手制作，未独立人工复核。

#### M050：无Lexical的原结构对照，seed7

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；用于换seed检验的无词汇对照，不是seed42的0.9248权重。
- **主要参数：** seed=7；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第6轮；单模型独立F1=0.424967、结构F1=0.912365。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_reliability/outputs/run01/baseline_seed_7/epoch_6/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好20行、破坏9行。
本组融合分数和逐行预测：[S13_predictions.csv](../../S13_predictions.csv)。

<a id="s14"></a>
### S14：分层融合：0.938作为N门

本组共6个实际模型。独立F1=0.545861，pair-aware F1=0.965418。N对101/106，配对F/B/E对536/554。

操作：由M059“原0.938版：scope200＋关系48”提供N概率；方向成员为M032、M044、M047、M071、M084，各自F/B/E条件概率按1/5等权平均。N门不是先给硬标签，也不是单例统一判N。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M032 | 词汇版，Pair阶段参数锚定 | F/B/E方向专家 | 3042 | 3042 | 42 | 9 | 0.675109 / 0.908480 |
| M044 | 词汇混合训练＋R-Drop | F/B/E方向专家 | 3042 | 3042 | 42 | 2 | 0.849526 / 0.883918 |
| M047 | 词汇版＋首批100组证据监督 | F/B/E方向专家 | 3042 | 3042 | 42 | 2 | 0.850874 / 0.872561 |
| M059 | 原0.938版：scope200＋关系48 | N概率门 | 3042＋200＋48＝3290 | 3290 | 42 | 2 | 0.547177 / 0.938145 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | F/B/E方向专家 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M084 | 原0.9355版：仅加scope200 | F/B/E方向专家 | 3042＋200＝3242 | 3242 | 42 | 3 | 0.480876 / 0.935493 |

#### M032：词汇版，Pair阶段参数锚定

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 保留原CLS＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 先完整Row CE并保存该阶段参数作为锚；Pair更新后按近端公式拉回Row锚点，anchor_strength=1000，之后再评估。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第9轮；单模型独立F1=0.675109、结构F1=0.908480。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run06_row_parameter_anchor/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run06_row_parameter_anchor/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M044：词汇混合训练＋R-Drop

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 每批同时计算Row和Pair；同一批做两次dropout前向，平均CE并加四类对称KL，R-Drop权重1.0。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.849526、结构F1=0.883918。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run05_mixed_rdrop/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run05_mixed_rdrop/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M047：词汇版＋首批100组证据监督

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，新增训练期token证据分类头。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合Row CE＋0.5 Pair CE＋0.1 token CE。首批100组标注覆盖175行，仅supported部分参与证据监督；不是把reason文字拼进输入，也不是生成reason。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.850874、结构F1=0.872561。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run08_reason_evidence_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run08_reason_evidence_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** reason在这里指Train理由标注转换成token证据监督；与Qwen生成完整解释的SFT不同。标注由助手制作，未独立人工复核。

#### M059：原0.938版：scope200＋关系48

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 全参数两阶段，先完整Row CE，再0.5 Pair CE；这就是提供0.938结构分的原模型。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.547177、结构F1=0.938145。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M084：原0.9355版：仅加scope200

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＝3242。200行范围/修饰与结构对照：属性限制、同义结构改写、中心对象替换、并列范围和顺序等。是历史人工模板草案，不是官方额外金标。
- **训练方式：** 全参数两阶段；与官方3042行原版相比，训练文本新增scope200。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.480876、结构F1=0.935493。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_original_aug/outputs/run01_original_scope/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

本组结构预测相对原0.938：修好18行、破坏0行。
本组融合分数和逐行预测：[S14_predictions.csv](../../S14_predictions.csv)。

<a id="s15"></a>
### S15：分层融合：0.9306作为N门

本组共6个实际模型。独立F1=0.486196，pair-aware F1=0.962259。N对99/106，配对F/B/E对536/554。

操作：由M027“原0.9306词汇版，seed42”提供N概率；方向成员为M054、M056、M071、M075、M082，各自F/B/E条件概率按1/5等权平均。N门不是先给硬标签，也不是单例统一判N。

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M027 | 原0.9306词汇版，seed42 | N概率门 | 3042 | 3042 | 42 | 3 | 0.468191 / 0.930569 |
| M054 | scope200＋早期人工80 | F/B/E方向专家 | 3042＋200＋80＝3322 | 3322 | 42 | 2 | 0.815929 / 0.917093 |
| M056 | scope200＋动作事件200 | F/B/E方向专家 | 3042＋200＋200＝3442 | 3442 | 42 | 4 | 0.664316 / 0.924949 |
| M071 | DeBERTa LoRA r16/dropout0.1，seed42 | F/B/E方向专家 | 3042＋200＋48＝3290 | 3290 | 42 | 16 | 0.858277 / 0.913362 |
| M075 | 0.938数据＋语义对照64，两阶段 | F/B/E方向专家 | 3042＋200＋48＋64＝3354 | 3354 | 42 | 4 | 0.464267 / 0.928647 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | F/B/E方向专家 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |

#### M027：原0.9306词汇版，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment＋20维Lexical修正分支。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Row N权重1.5，Pair EE权重1.25。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.468191、结构F1=0.930569。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** [best_model.pth](../../../../../../model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth)。

#### M054：scope200＋早期人工80

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＋80＝3322。80行早期人工对照，涉及修饰限制、上下位、词汇/拼写等价、属性独立、中心词替换、事件状态等；不是后来的relation48。
- **训练方式：** 全参数两阶段；在3242行scope底座上再加早期80行人工草案。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第2轮；单模型独立F1=0.815929、结构F1=0.917093。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run01_original_manual/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run01_original_manual/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run01_original_manual/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M056：scope200＋动作事件200

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Token Alignment；没有Lexical。
- **训练文本：** 3042＋200＋200＝3442。额外200行为20个动作/事件家族，每家族10行。包括动作具体化、同义改写、事件阶段/状态差异；这200行F40、B40、E80、N40。
- **训练方式：** 全参数两阶段；事件/动作200是额外训练数据，不是模型分支。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第4轮；单模型独立F1=0.664316、结构F1=0.924949。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run03_event_200/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run03_event_200/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run03_event_200/epoch_4/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M071：DeBERTa LoRA r16/dropout0.1，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋CLS＋Alignment；没有Lexical。冻结大部分编码器参数，训练LoRA和任务头；这不是Qwen的r16版本。
- **训练文本：** 3042＋200＋48＝3290。在scope底座上加入48行关系对照，包含空间/时间范围、参与者与角色、事件身份、词义包含、结构改写等；新增48行F/B/E/N各12。
- **训练方式：** 混合Row CE＋0.5 Pair CE；LoRA r16/alpha32/dropout0.1，target=query_proj/value_proj，seed42；没有梯度投影。
- **主要参数：** seed=42；主学习率=0.0001；计划20轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第16轮；单模型独立F1=0.858277、结构F1=0.913362。
- **历史选模规则：** `max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run06_relation48_staged/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run17_lora_r16_dropout01/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M075：0.938数据＋语义对照64，两阶段

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 原DeBERTa CLS＋Alignment；没有Lexical、没有LoRA。
- **训练文本：** 3042＋200＋48＋64＝3354。额外64行F/B/E/N各16，分food_activity、aircraft_scope、modifier_scope、event_roles四个家族，各16行；官方Train词汇重组成新对照，涉及论元、上下位、修饰范围、事件角色及状态。
- **训练方式：** 全参数两阶段；开启动态padding和梯度检查点，因而训练轨迹也与历史固定padding的0.938有差别。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第4轮；单模型独立F1=0.464267、结构F1=0.928647。
- **历史选模规则：** `best AFTER-PAIR Dev pair-aware weighted F1; earliest on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/run_config.json)。
- **训练数据文件：** [training_data.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/training_data.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_manual_aug/outputs/run20_semantic64_staged_seed42/best/dev/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。
- **需保留的说明：** 新增标签来自历史助手设计的增强草案，不是官方新增金标；部分设计受Dev错误类别启发，不能宣称开发过程完全独立于Dev。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好19行、破坏3行。
本组融合分数和逐行预测：[S15_predictions.csv](../../S15_predictions.csv)。

<a id="s16"></a>
### S16：仅官方3042行：分层融合，最高结构分

本组共6个实际模型。独立F1=0.481308，pair-aware F1=0.954759。N对98/106，配对F/B/E对532/554。

操作：由M027“原0.9306词汇版，seed42”提供N概率；方向成员为M017、M030、M046、M082、M087，各自F/B/E条件概率按1/5等权平均。N门不是先给硬标签，也不是单例统一判N。

**本组的六个成员全部只用官方3042行训练文本和原始标签，没有scope、relation、closure、人工样本或反向N，也没有reason/evidence额外标注；不需要复制数据。**

本组成员和各自使用的数据：

| 成员 | 清楚一点的名称 | 在本组的职责 | 训练数据 | 行数 | seed | 选中轮次 | 单模型独立/结构F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| M017 | CLS、对齐、词汇特征拼接版 | F/B/E方向专家 | 3042 | 3042 | 42 | 10 | 0.507145 / 0.916871 |
| M027 | 原0.9306词汇版，seed42 | N概率门 | 3042 | 3042 | 42 | 3 | 0.468191 / 0.930569 |
| M030 | 词汇版，N行权重1.8 | F/B/E方向专家 | 3042 | 3042 | 42 | 9 | 0.704990 / 0.919490 |
| M046 | 词汇版，Row CE＋反向KL | F/B/E方向专家 | 3042 | 3042 | 42 | 2 | 0.860056 / 0.886018 |
| M082 | DeBERTa-large＋对齐＋词汇混合训练 | F/B/E方向专家 | 3042 | 3042 | 42 | 10 | 0.820455 / 0.856214 |
| M087 | 冻结底6层，只训练Row CE | F/B/E方向专家 | 3042 | 3042 | 42 | 8 | 0.848510 / 0.887296 |

#### M017：CLS、对齐、词汇特征拼接版

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 归一化CLS、token比较特征及词汇特征拼接，共1312维，经联合分类器输出四类；不是三个logit分支直接相加。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Lexical有单独学习率，Pair后按结构F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.507145、结构F1=0.916871。
- **历史选模规则：** `after Pair CE: best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_feature_fusion/outputs/run01_feature_fusion/feature_fusion_seed_42/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M027：原0.9306词汇版，seed42

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment＋20维Lexical修正分支。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Row N权重1.5，Pair EE权重1.25。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第3轮；单模型独立F1=0.468191、结构F1=0.930569。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run02_branch_lr/lexical_seed_42/epoch_3/predictions_with_scores.csv)。
- **已有权重：** [best_model.pth](../../../../../../model/deberta_lexical/run02_branch_lr/lexical_seed_42/best_model.pth)。

#### M030：词汇版，N行权重1.8

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** 与原词汇版相同的CLS＋Alignment＋Lexical结构。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数两阶段；Row的N权重改为1.8，Pair仍为[1,1,1.25]。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；Row/Pair batch=16/16；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.8]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第9轮；单模型独立F1=0.704990、结构F1=0.919490。
- **历史选模规则：** `best Dev pair-aware weighted F1; earliest epoch on ties`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical/outputs/run04_weight/lexical_seed_42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical/outputs/run04_weight/lexical_seed_42/epoch_9/predictions_with_scores.csv)。
- **已有权重：** [best_model.pth](../../../../../../model/deberta_lexical/run04_weight/lexical_seed_42/best_model.pth)。

#### M046：词汇版，Row CE＋反向KL

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋Alignment＋Lexical，不使用NLI pooler。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 混合组批；保留四类Row CE，用0.5×四类正反对称KL替代Pair CE。只在真实可逆组对齐F/B，没有另造N反向监督。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；反向KL权重=0.5。
- **本次采用版本：** 第2轮；单模型独立F1=0.860056、结构F1=0.886018。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_lexical_mixed_pair/outputs/run07_reversal_kl_seed42/epoch_2/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M082：DeBERTa-large＋对齐＋词汇混合训练

- **基础模型：** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- **结构：** NLI-large底座，hidden_size1024、24层；自建CLS四分类头＋Alignment＋Lexical。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 全参数混合组批，Row CE＋0.5 Pair CE；使用bf16 autocast和梯度检查点，按独立F1选模。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；词汇分支学习率=0.001；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第10轮；单模型独立F1=0.820455、结构F1=0.856214。
- **历史选模规则：** `independent`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_nli_large/outputs/run01_large_mixed_pair/epoch_10/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

#### M087：冻结底6层，只训练Row CE

- **基础模型：** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。
- **结构：** DeBERTa NLI-base＋原CLS四分类头＋Token Alignment，没有Lexical。冻结embedding和底部6层，顶部6层及任务头可训练。
- **训练文本：** 3042。仅官方英文训练文本，没有新增短语对；不同模型仍可能采用OOF加权、证据辅助标注或不同预训练初始化。
- **训练方式：** 只用四类Row CE，Pair权重为0，不做Pair CE。按组组织批次仅为匹配训练安排，训练行每轮经过一次，无增强、无过采样。
- **主要参数：** seed=42；主学习率=2e-05；计划10轮；max_length=128；group batch=8（组数）；Row权重F/B/E/N=[1.0, 1.0, 1.0, 1.5]；Pair权重FB/BF/EE=[1.0, 1.0, 1.25]。
- **本次采用版本：** 第8轮；单模型独立F1=0.848510、结构F1=0.887296。
- **历史选模规则：** `None`。
- **配置依据：** [该次运行保存的run_config.json](../../../../../deberta_row_ce_frozen/outputs/run01_freeze6/run_config.json)。
- **训练数据文件：** [dico_nli_train_track1_participant_labeled.csv](../../../../../../data/train/dico_nli_train_track1_participant_labeled.csv)。
- **本次参与融合的预测：** [predictions_with_scores.csv](../../../../../deberta_row_ce_frozen/outputs/run01_freeze6/epoch_8/predictions_with_scores.csv)。
- **已有权重：** best_model.pth（本机文件，不随仓库提供）。

本组结构预测相对原0.938：修好19行、破坏8行。
本组融合分数和逐行预测：[S16_predictions.csv](../../S16_predictions.csv)。

## 怎样阅读结果，避免误解

1. 97份成员分数不是97种预训练架构。绝大多数共用同一个NLI-base骨干，只是在数据、seed、分支或训练目标上有差别。
2. S03是在本次Dev搜索里同时超过独立0.87和结构0.94的普通平均方案；S01优先结构分；S02优先独立分。它们不是同一个方案的三个不同选模时刻。
3. S14虽然结构分最高，但独立分仍低，不能当成独立判断已达到0.965。
4. 标题“官方3042行”只表示没有新增训练文本，并不表示没有额外证据标注、OOF加权或外部NLI初始化。S06和S13都含辅助证据标注模型。
5. scope、relation、event、local和semantic增强有历史Dev引导开发记录。新样本与Dev的字符串交集为0，不等于模板、语义和研究过程完全独立；隐藏Test未提供，不能声称已证明与Test无交集。
6. 这些高分是在重复使用的Dev上做大量组合搜索后选出的，存在选择偏差。1—3成员穷举，较大方案采用beam扩展，不保证全局最优。
7. 本次没有加载真实模型重跑。权重路径存在只证明文件存在；可复核的是冻结CSV的融合分数及官方评分。LoRA成员还需要对应基础模型和自定义任务头，不能把best_model.pth一律当完整独立模型直接加载。
8. 单模型成绩一般不代表对集成没用；本次某些弱成员参与高分组合，只证明它们在这份Dev上的分数有互补，不能推断其单独训练方法普遍优越。

## 文件索引

- [16组组合汇总](../../recommended_combinations.csv)
- [固定成员、原始分数来源与融合方式](../../selected_recipes.json)
- [本报告35个成员的详细事实快照](../../member_details.json)
- [所有已计算组合](../../all_combinations.csv)
- [97份分数的来源清单](../../members.csv)
- [按成员数量比较](../../best_per_model_count.csv)
- [扩充前的原报告备份](../../RESULTS_before_details.md)
