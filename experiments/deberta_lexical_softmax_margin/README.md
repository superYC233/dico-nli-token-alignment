# Lexical＋距离感知交叉熵

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

基于刚跑完的 `deberta_lexical_pair_routing`，模型仍为 **DeBERTa＋Token Alignment＋Lexical**，只使用官方Train **3042行**。本次改Row损失，Pair仍只更新Alignment。

**flat/distance的seed42已跑完；本次补种子请运行 [run_seed_check.py](run_seed_check.py)。** 它只新增seed7的CE和distance两次训练，再汇总已有seed42。不要再次运行下面的run_all来重复seed42实验。

**直接运行 [run_all.py](run_all.py)**：依次跑统一代价flat、方向编码代价distance，然后生成比较表。两个实验分别从同一个NLI预训练主干初始化，使用同一seed和训练预算。

```powershell
python 'experiments/deberta_lexical_softmax_margin/run_all.py'
```

普通CE基线已有结果，默认不重训。`compare_runs.py`核对训练配置及数据快照哈希后，会把已有CE结果纳入比较；若不匹配，则不混入。已完成且两份最佳权重均存在的flat/distance运行会跳过；中断的运行不自动覆盖，换RUN_NAME重新运行。

| 模式 | Row损失 | 用途 |
|---|---|---|
| ce | 原加权四分类CE | 重跑普通CE对照，可选 |
| flat | CE(logits＋0.5×统一错误代价, gold) | 检验统一要求更大的正确分差 |
| distance | CE(logits＋0.5×方向编码代价, gold) | 检验区分错误类型是否额外有用 |

**只跑距离版**，可直接运行 [train_deberta_softmax_margin.py](train_deberta_softmax_margin.py)。入口默认 `LOSS_MODE='distance'`。运行不同模式不需要新文件夹；修改[config.py](config.py)的LOSS_MODE，或传参数：

```powershell
python 'experiments/deberta_lexical_softmax_margin/train_deberta_softmax_margin.py' --loss-mode flat
```

普通CE重跑使用 `--loss-mode ce`。将 `--margin-weight 0` 传给flat/distance会完全退回普通CE计算路径。调方法时可用 `--validation internal`，它仅从Train按来源和重复文本对整组划分，不读取官方Dev；内部结果与官方660行结果分开记录。

**距离表怎么理解**

行是真实标签，列是候选标签，顺序固定为F、B、E、N：

| Gold / Candidate | F | B | E | N |
|---|---:|---:|---:|---:|
| F | 0 | 2 | 1 | 1 |
| B | 2 | 0 | 1 | 1 |
| E | 1 | 1 | 0 | 2 |
| N | 1 | 1 | 2 | 0 |

F/B方向颠倒、E/N混淆的试验代价设为2，其他错误为1。这是**人为设定、需要验证的方向编码先验**，不是官方语义距离；尤其N汇总了多种其他关系，不能据此断言每条N都具有相同的语义距离，也没有因此生成反向N。

flat所有错误代价都是1。distance原始平均错误代价为4/3，因此训练前除以4/3；实际较近/较远的代价是0.75/1.5，两版平均错误代价均为1。这匹配了平均代价尺度，不保证两版实际梯度范数或每批损失相等。

训练时只给错误类别增加竞争分数，正确类别加0，迫使模型学出更大的正确分差。**预测时使用原始logits**，不使用真实标签、不加入代价。原官方weighted F1不变；新增距离指标只作诊断，不能代替官方成绩。

**一轮训练**

```text
Row：3042行，全部参数更新，使用所选的四分类CE（191步）
  → 保存Row后的两种F1、原始预测与分支logits
Pair：1282个真实FB/BF/EE组，只更新Alignment（81步）
  → 原0.5×三组合Pair CE
  → 保存Pair后的同样记录及修好/破坏数量
下一轮Row恢复全部参数更新
```

主干学习率2e-5，Lexical 1e-3；Row权重1/1/1/1.5，Pair权重1/1/1.25，Pair label smoothing 0.05；batch16、FP32、10轮、seed42。全程一个共享AdamW；冻结参数grad=None，保留旧版dropout、模型和初始化方式。没有新增训练文本、辅助分类头、模型集成或额外主干前向。

模型保存在 `model/deberta_lexical_softmax_margin/`。默认距离版子目录为 `run01_distance_m0p5_official/seed42/`；统一代价版是 `run01_flat_m0p5_official/seed42/`。结果在本目录同名的 `outputs/` 子目录。重复实验改RUN_NAME。

每个版本只保留 `best_independent.pth`、`best_pair_aware.pth` 两份最佳权重，刷新时覆盖，不保存每轮权重。结构最佳只在Pair后选择，独立最佳在两个阶段选择。两份最佳可能不是同一轮；JSON和最终比较表的每一行均列出**同一检查点的两种F1**。

**跑完看什么**

- [compare_runs.py](compare_runs.py)：重新汇总已有结果；比较表位于 `outputs/run01_comparison_official_seed42_m0p5/comparison.csv`。
- 各版本 `epoch_summary.csv`：每轮两阶段F1、完整margin损失、普通CE、margin新增项。
- `pair_changes.csv`：Pair修好/破坏数量及N预测变化。
- 各阶段 `summary.csv`：官方成绩和方向编码距离诊断；`probability_metrics.json`记录未加类别权重的NLL、四类Brier分数及低金标概率错误数。
- `branch_logits.csv`、`predictions_with_scores.csv`：实际预测分数；`pair_branch_changes.csv`检查冻结的CLS/Lexical输出是否不变。
- [evaluate_best.py](evaluate_best.py)：默认重评distance结构最佳；传 `--loss-mode flat` 或 `--kind independent` 选择其他权重。重载完整权重无需重新下载预训练主干。

当前可复用的CE基线：结构最佳检查点为第2轮Pair后，independent/pair-aware＝0.836464/0.892937；独立最佳检查点为第8轮Pair后，0.845797/0.881543。不要把两行的最高分拼成一份模型成绩。

判读顺序：先比较F1，确认是否提高正确标签数量；再检查距离是否降低。如果只有距离降低、F1不升，只能说按这张先验表错得更轻。如果distance没有超过flat，不能把收益归因于距离结构。代价表和系数提前固定，不用Dev逐项调表。

**检查与来源**

[check_setup.py](check_setup.py)仅在CPU上用随机小DeBERTa检查数学损失、加权均值、梯度、普通CE/零margin与稳定基线两轮逐参数一致、Pair冻结与Adam状态、官方评分和模型重载。正式训练由你运行。

核心改动见[losses.py](losses.py)。模型、特征、评分以及分支记录复用项目已有实现；旧目录不修改。理论参考：[Gimpel & Smith，Softmax-Margin CRFs，NAACL 2010](https://aclanthology.org/N10-1112/)。本次是神经四分类的适配，不能照搬原论文的收益结论。

## 补seed7：同种子比较CE与distance

直接运行 [run_seed_check.py](run_seed_check.py)：

```powershell
python 'experiments/deberta_lexical_softmax_margin/run_seed_check.py'
```

默认依次训练 `CE seed7`、`distance seed7`。调用同一个 `train_deberta_softmax_margin.py`，保持原10轮、margin=0.5、代价表、数据、冻结范围和选模规则；每个方法从预训练主干重新初始化。seed42的CE和distance读取历史结果，不重新训练。已有完成记录且两份最佳权重都存在的seed7运行会自动跳过。

新权重在同一个模型根目录下的 `run01_ce_official/seed7/` 和 `run01_distance_m0p5_official/seed7/`。结果在本目录 `outputs/` 的对应子目录中，不需要新建实验文件夹。

跑完自动生成 `outputs/run01_seed_check_official_m0p5/`：

- `per_seed_checkpoints.csv`：每种方法、每个种子的最佳检查点；每行两个F1来自同一份权重。
- `mean_std.csv`：两个种子的均值和样本标准差。按独立选模、按结构选模分别统计；F1仍是0—1数值。
- `paired_deltas.csv`：每个seed内的distance−CE差值，单位为百分点。
- `paired_delta_summary.csv`：同种子差值的均值、样本标准差及正提升种子数。

只汇总、不训练：加参数 `--summary-only`。继续补其他种子可使用 `--seeds 13`，当前先只跑默认seed7。

汇总先核对配置和数据快照；只有CE与distance两种选模结果均完整的种子才计入均值和差值。尚未跑完的seed不填假分数，一个种子时标准差留空。该入口不做预测集成。两个种子只能提供初步复核，不能单凭它们宣称稳定优势。

入口与统计检查：[check_seed_check.py](check_seed_check.py)，使用mock检查启动命令与统计，不启动正式训练。

## 补flat seed7：区分普通margin与距离设计的收益

直接运行 [run_flat_check.py](run_flat_check.py)。只补训一次 **flat seed7**，不改config默认值，不重跑CE、distance或seed42。

```powershell
python 'experiments/deberta_lexical_softmax_margin/run_flat_check.py'
```

沿用官方Train3042／Dev660、10轮、batch16、margin=0.5和Pair阶段仅更新Alignment。flat的所有错误代价为1；distance的错误代价为0.75或1.5，平均均为1。训练函数、选模和预测方式均复用现有代码。

权重保存在 `model/deberta_lexical_softmax_margin/run01_flat_m0p5_official/seed7/`，只保留两份最佳权重。已有完成记录和两份权重时跳过训练。结果保存在 `outputs/run01_flat_m0p5_official/seed7/`。

跑完读取两个种子的CE、flat、distance，生成 `outputs/run01_flat_check_official_m0p5/`，包含每个检查点、均值与样本标准差、同种子差值及其汇总；选模口径分别统计，不拼接不同检查点的最高分：

- **flat−CE**：统一要求更大分类分差是否有效。
- **distance−flat**：区分错误类型是否有额外收益；这是本次重点。
- **distance−CE**：保留已有总收益对照。

仅汇总可加 `--summary-only`。缺少完整结果的种子不计入三方法均值和差值，单种子标准差留空。仍是各单模型的比较，没有融合预测。已在CPU上核对启动参数和统计，正式训练由你运行。

## 两个种子的共同错题与Train近邻

右键运行 [review_common_errors.py](review_common_errors.py)。只读取distance seed42、seed7各自按independent选出的最佳预测和训练快照，用CPU分析，不加载模型、不训练、不改训练数据。

输出在 `outputs/run01_common_errors_official_m0p5/`：

- `common_errors_review.csv`：共同错题、两次预测、置信度及金标概率，末尾留有人工核验栏。
- `train_neighbors.csv`：每条错题的3个Train词面近邻，附原始标签和ID；检索只在Train上建索引，不自动判定错因。
- `error_counts.csv`：N/非N、E/F/B和方向错误的数量。
- `seed_sensitive_errors.csv`：只在一个种子答错的样本。
- `train_label_conflicts.csv`：Train同一规范化有序输入的不同标签记录，不自动改标。
- `README.md`、`summary.json`：核验说明、输入文件及哈希。

这个入口用于决定后续该补什么监督，而不是关闭Lexical的训练入口。共同错误涉及Dev，输出仅用于诊断，不会直接进入训练。重复运行会刷新这些自动分析表，填写人工核验前请先将 `common_errors_review.csv` 另存一份。
