PAWS辅助等价训练
================

运行顺序
--------

1. 运行`prepare_paws_data.py`，下载并固定2000条PAWS训练样本。
2. 运行`train_deberta_paws.py`，开始run03的完整训练。

直接运行训练入口也会自动执行第1步。首次准备需要联网，之后按文件哈希复用。
训练从原NLI底座重新初始化，不加载run02的最佳权重续训。
run02基线：第4轮独立0.856822、pair-aware0.906326。
原0.938模型与run01、run02文件均保留。

模型和DiCo配方不变
-----------------

- 全量DeBERTa-v3-base NLI，原CLS/Token Alignment特征，双向联合四分类头。
- DiCo仍是3290行：官方3042 + scope200 + relation48；不追加PAWS为四类样本。
- seed42，lr2e-5，10轮，float32，AdamW weight_decay0.01。
- DiCo四类CE权重[1,1,1,1.5]，固定max_length128，每批8个文本对/16个编码视图。
- 无LoRA、无三组合Pair CE，没有新的推理模块。
- 每轮都报告独立和pair-aware分数；与run02一样，按pair-aware最高轮保存一份完整权重。

PAWS到底监督什么
---------------

PAWS原标签：1是改写等价，0是非等价。0可能对应DiCo的F、B或N，不能一律改成N。
代码保留独立的`paws_label`，不生成DiCo四类金标。

对已有F/B/E/N四个logits，二元任务的分数是：

    not_equivalent = logsumexp(logit_F, logit_B, logit_N)
    equivalent = logit_E
    paws_loss = CE([not_equivalent, equivalent], paws_label)

这等价于对四类softmax中的p(E)做BCE，使用logsumexp避免log(0)。
正例提高E概率，负例降低E概率，但不规定它是F/B/N中的哪一类。
PAWS的改写约定与DiCo不保证完全相同；这是等价监督迁移实验，不是已证实的提升。

一轮怎么训练
------------

    第1批：DiCo CE → 更新
    第2批：DiCo CE → 更新
    第3批：DiCo CE → 更新
    第4批：DiCo CE梯度 + 0.1 * PAWS CE梯度 → 一次更新
    然后重复……

两个任务先后反向传播、累积到同一次optimizer.step；DiCo图先释放，再前向PAWS。
这样不会同时保留两份encoder激活，也不会产生仅用PAWS的参数更新。
每轮DiCo3290行仍各监督一次，共236次更新，其中59次加入PAWS。
PAWS池有2000条，每轮约看472条，每轮重新打乱池；10轮约4720次辅助样本曝光。
不是每轮把全部2000条额外训完。

权重0.1表示合并梯度中辅助项的系数，不代表AdamW实际参数位移恰好是10%。
`paws_exposure.csv`记录实际取到的source_id，`epoch_summary.csv`记录两种loss和曝光数。
将PAWS_WEIGHT设为0时跳过PAWS前向及取样，但本次先使用默认0.1。

准备数据的规则
--------------

- 来源是PAWS-Wiki Labeled Final人工标注数据，优先读取Google Research HF仓库的train Parquet。
  也兼容本地原始train.tsv/压缩包；压缩包只解析train.tsv。
- 不使用PAWS Dev/Test或无人工审核的noisy/unlabeled版本。
- 对Unicode NFKC、大小写及空白规范化后，排除与DiCo Train/Dev整段文本或无序文本对相同的样本。
- Dev只读取文本用于重复排除，不读取Dev标签筛样本。
- 去掉同文、自相矛盾的重复标签及正反重复；保留完整文本，两个方向均不超过128token。
- 固定seed42抽取1000等价、1000非等价。
- 自然文本会共享单词；这里核查的是整段文本/文本对，不声称词汇零交集。
- 未访问隐藏Test，不声称已经证明与隐藏Test完全无交集。

准备目录：`outputs/paws_prepared_seed42/`。
包含paws_train.csv、preparation.json、overlap_report.json和rejected_rows.csv。
原始训练文件缓存到其raw子目录。配置或已选CSV变化会要求换新的准备目录。
Google Storage旧下载地址已返回AccessDenied，当前使用HF上的训练分区。
手动下载时，把`train-00000-of-00001.parquet`放进该raw子目录，再运行准备脚本。

结果位置
--------

默认RUN_NAME：`run03_paws_aux_seed42`。

- 模型：`D:/model/SEAMEVAL/deberta_bidirectional/run03_paws_aux_seed42/best_model.pth`
- 结果：当前目录`outputs/run03_paws_aux_seed42/`
- 原3290行、PAWS2000行及来源审计均单独复制到该次运行目录。
- `best/dev/`和`best/train/`分别是最佳模型的DiCo Dev、DiCo Train结果。
- 单独重载评估：`evaluate_deberta_paws.py`。
- 改RUN_NAME可以保留多次运行；PAWS_WEIGHT/PAWS_EVERY在新训练入口顶部。

检查
----

运行`check_paws_auxiliary.py --real`可在CPU检查筛选、二元损失、梯度合并、完整权重保存及真实底座前向。
检查使用小型夹具，不下载真实数据、不执行优化器参数更新。
`train_deberta_paws.py --prepare-only`会准备真实数据并分词，但不加载模型或开始训练。

2026-10-24已通过上述离线检查和真实DeBERTa的二元前向/梯度检查。
实际PAWS下载与2000条样本准备尚未运行，先运行准备入口即可；这些检查没有训练成绩。

来源
----

- 论文：https://aclanthology.org/N19-1131/
- 官方项目：https://github.com/google-research-datasets/paws
- HF数据集：https://huggingface.co/datasets/google-research-datasets/paws
- 训练分区：https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/train-00000-of-00001.parquet

论文应披露该外部训练数据和上述过滤/辅助损失，不把PAWS的原始标签说成DiCo人工四类标签。
