# 01｜BERT Baseline：准备、代码流程与现存材料

整理日期：2026-10-27。项目根目录：`G:/pycode/SemEval/Task-2-DiCo-NLI-main`。

**本篇的 baseline 是最初的 BERT 四分类模型。** 后来的 DeBERTa 0.9248、Lexical、增强数据和模型融合是后续实验，不作为本篇的 baseline。

本次只扫描文件、读取配置和代码、核对数据；没有训练、加载大模型推理、修改 Python 代码或删除文件。以下区分“当前代码实际做什么”和“历史运行已经确认什么”。

## 1. Baseline 要回答什么问题

给定两个有顺序的英文短语 `text1`、`text2`，先验证普通预训练编码器能否直接学会四类关系。这是后续增加结构、特征或训练目标时的比较起点。

| 编号 | 简写 | 官方标签 | 含义 |
|---:|---|---|---|
| 0 | F | `FORWARD_ENTAILMENT` | text1 蕴含 text2，反向不成立 |
| 1 | B | `BACKWARD_ENTAILMENT` | text2 蕴含 text1，反向不成立 |
| 2 | E | `EQUIVALENCE` | 两个方向都成立 |
| 3 | N | `NEGATIVE_OTHER` | 不属于上面三类 |

例如，`a red car / a car` 用于说明 F 的方向：前者比后者多了颜色限制。交换输入，方向随之变成 B。这里是解释标签，没有向训练集新增样本。

## 2. 本篇对应的实际文件

| 材料 | 路径及当前状态 |
|---|---|
| BERT baseline 源码 | [bert_baseline.py](../experiments/baselines/bert_baseline.py)，存在 |
| 根目录路径兼容工具 | [project_paths.py](../project_paths.py)，存在 |
| 官方训练集 | [Train CSV](../data/train/dico_nli_train_track1_participant_labeled.csv)，存在 |
| 官方开发集 | [Dev CSV](../data/dev/dico_nli_dev_track1_participant_labeled.csv)，存在 |
| 官方开发集 reference | [Dev reference](../data/dev/dico_nli_dev_track1_reference.csv)，存在 |
| 本地预训练 BERT | `D:/BaiduNetdiskDownload/PretrainedModel/models/google-bert--bert-base-uncased/snapshots/master`，目录存在，含配置、词表和权重 |
| BERT 任务微调权重 | 源码预期为项目 `model/bert_dico_epoch_*.pth`；本次在项目中没有找到 |
| 历史实验文字记录 | [完整实验记录](history/DICO_NLI_FINAL_EXPERIMENT_RECORD.md)，其中 BERT 准确成绩标为缺失 |

预训练 BERT 权重和经过本任务训练的 BERT 权重是两回事：前者现在有，后者本次没有找到。

## 3. 数据准备：只用官方 Train 和 Dev

当前代码直接读取完整官方划分，没有增强、复制、检索示例或 reason。

| 数据 | 行数 | F | B | E | N | pair_id 组数 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3042 | 881 | 881 | 802 | 478 | 1760 |
| Dev | 660 | 190 | 190 | 174 | 106 | 383 |

本次重新读取 CSV 核对了以上数字。每行的字段是：

```text
instance_id, pair_id, text1_lang, text2_lang, text1, text2, label
```

- `text1`、`text2`：进入 BERT 的两个文本。
- `label`：转换为 0、1、2、3，供训练 CE 和开发集评估使用。
- `instance_id`：导出预测时与官方样本对应。
- `pair_id`：数据本身有这个字段，但这个 BERT baseline 不用它构造损失，也不用于联合解码。
- 两个语言字段：当前英文实验不作为额外的模型输入。

Train 有 1282 个两行组和 478 个单例组；Dev 有 277 个两行组和 106 个单例组。baseline 将它们视为普通行来训练和预测，没有利用整组约束。

Dev 不加入训练 DataLoader，但旧脚本在 Dev 上比较不同的 N 分数惩罚值。因此，这些比较属于开发集调参，不能当成独立测试集表现。

## 4. 模型准备：普通 BERT，而非 NLI 微调模型

代码使用 `BertTokenizer` 和 `BertModel`，本地目录对应 `bert-base-uncased`。这里不是后来的 MNLI／ANLI 微调 DeBERTa。

本地配置核对结果：

| 项目 | 配置 |
|---|---|
| 编码器层数 | 12 |
| 隐藏维度 | 768 |
| 每层注意力头数 | 12 |
| 词表大小 | 30522 |
| 大小写处理 | `do_lower_case=True` |
| 本地权重文件 | `model.safetensors` 与 `pytorch_model.bin` 均存在 |
| 本实验使用的接口 | `BertModel`，外接自己的四分类头 |

本地 `config.json` 的 `architectures` 写着 `BertForMaskedLM`，但当前脚本实际实例化的是 `BertModel`。本实验使用编码器表示做四分类，不是在训练填空任务，也不直接复用一个 NLI 分类头。

## 5. 每条数据如何送入模型

`DiCoDataset.__getitem__()` 同时把两个短语交给 tokenizer：

```python
encoded = self.bert_tokenizer(
    text1,
    text2,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)
```

输入形态可以理解为：

```text
[CLS] text1 的 tokens [SEP] text2 的 tokens [SEP] [PAD] ...
```

这里的 128 是整条拼接输入的最大长度，不是每个短语分别 128。

模型接收三项：`input_ids` 表示 token 编号，`attention_mask` 区分真实 token 和补齐位置，`token_type_ids` 区分两个文本段。两个短语在同一次 BERT 编码中相互注意；它不是先独立算两个句向量再比较相似度。

## 6. 四分类头为什么这样写

实际网络是：

```text
text1 + text2
      ↓
BERT 联合编码
      ↓
last_hidden_state[:, 0, :]，取 [CLS] 的 768 维表示
      ↓
Dropout(0.2)
      ↓
Linear(768, 4)
      ↓
F / B / E / N 的四个 logits
```

`[CLS]` 表示整条输入在上下文中的综合信息；分类头学习用它区分四类关系。Dropout 在训练时随机屏蔽一部分特征，起正则作用，评估模式下关闭。最后四个数是未归一化分数，直接交给交叉熵即可，不需要先手动 softmax。

训练时更新 BERT 和分类头的全部参数。这个版本没有冻结、LoRA、Token Alignment、Lexical、Pair CE、pair-aware 或集成。

## 7. 原始训练设置与一轮流程

| 参数 | 当前源码 |
|---|---|
| 训练轮数 | 10 |
| Train／Dev batch size | 16／16 |
| 最大输入长度 | 128 |
| 优化器 | AdamW |
| 学习率 | `2e-5` |
| weight decay | `0.01` |
| 损失 | 普通四分类 `CrossEntropyLoss()`，无类别加权 |
| 训练样本顺序 | `shuffle=True` |
| 开发集顺序 | `shuffle=False` |
| 精度 | 没有显式启用 AMP／bf16，按默认浮点配置运行 |
| 学习率调度／梯度累积 | 没有显式设置 |
| 固定随机种子 | 当前脚本没有设置 |

一轮只做一遍 Row CE：

```text
打乱 Train 的 3042 行
  → 每批最多 16 行
  → BERT 前向，输出四类 logits
  → 与每行金标计算 CE
  → 清空梯度、反向传播、AdamW 更新
  → 遍历全部训练批次
  → 保存 epoch_N.pth
```

按当前 batch size，每轮有 `ceil(3042/16)=191` 次参数更新，10 轮共 1910 次。没有“Row 之后再 Pair”这个第二阶段。

## 8. 预测、N 惩罚和选模必须分开理解

普通预测是对每行的四类 logits 直接 `argmax`，这就是 independent 预测。

旧脚本还包含一个推理实验：

```python
bert_output[:, 3] -= negative_penalty
pred = torch.argmax(bert_output, dim=1)
```

它在预测时降低 N 的分数，尝试减少“其他类别被判成 N”。这不更新参数，不是新损失，也不是 pair-aware。惩罚设为 0 才是未经该修正的原始四分类输出。

**当前 `evaluate_model()` 写的是 `range(8, 9)`，实际只评估第 8 轮。** 虽然函数里有 `best_score` 和 `best_epoch`，也不能据此说它已经从 10 轮中选出了最佳模型。

当前 `__main__` 的实际顺序是：

```text
读取数据和预训练 BERT
  → checkAndprocess_data() 建立 label_id
  → 只加载 epoch_8 微调权重
  → 分别试 N penalty = 0.0 / 0.5 / 1 / 1.5
  → 打印 Weighted F1、分类报告和混淆矩阵
  → 导出每个 penalty 对应的预测 CSV
```

它**没有调用 `train_model()`**。因此，现在直接运行这个旧脚本不会自动从头训练；由于项目中的 `bert_dico_epoch_8.pth` 已未找到，当前也不满足它的评估前提。

如果后续要重做 BERT baseline，需要明确安排“标签初始化 → 训练 → 评估”流程、模型保存目录、选模方式和随机种子。本篇记录现状，没有替用户改动这些代码。

## 9. 保存方式和目前能否复现

源码的训练函数每轮保存：

```text
G:/pycode/SemEval/Task-2-DiCo-NLI-main/model/bert_dico_epoch_1.pth
...
G:/pycode/SemEval/Task-2-DiCo-NLI-main/model/bert_dico_epoch_10.pth
```

它没有在训练中只保留最佳权重，也没有显式保存优化器状态、随机状态、训练配置或 tokenizer 副本。不能把后面新实验的“只保存最佳模型”行为套到这个脚本上。

| 项目 | 当前核对结论 |
|---|---|
| 预训练 BERT 能否定位 | 能，本地目录及权重存在；本次未加载推理 |
| 训练集、Dev 是否齐全 | 齐全，行数和标签已核对 |
| BERT 微调权重是否找到 | 项目内未找到 |
| 当前脚本是否直接开始训练 | 否，入口只做第 8 轮评估 |
| 历史最佳 epoch 是否确认 | 未确认；当前评估第 8 轮不等于历史最佳第 8 轮 |
| BERT 历史准确 F1 是否确认 | 未确认，本篇不填写猜测值 |

项目有一些 `predictions/legacy_calibration/dev_predictions_penalty_*.csv`，但不能仅凭文件名认定为 BERT 结果。本次核对发现，其中 `dev_predictions_penalty_0.0.csv` 与 `dev_deberta_predictions_penalty_0.0.csv` 的 SHA-256 完全相同，模型归属存在歧义。因此，不能把这批文件或对应 penalty 分数直接写成 BERT baseline 的成绩。

## 10. 当前环境与归档指纹

当前 Python 环境为 `G:/anaconda3/envs/nlpbase/python.exe`。下表来自本次读取已安装包的元数据，不代表最初实验的环境，也不表示本次已完成兼容性运行验证。

| 依赖 | 当前版本 |
|---|---|
| torch | `2.11.0+cu128` |
| transformers | `5.17.0` |
| pandas | `2.3.3` |
| scikit-learn | `1.7.2` |
| numpy | `2.2.5` |
| tqdm | `4.70.0` |

本次核对的 SHA-256：

```text
experiments/baselines/bert_baseline.py
af12d9d4f7f2770801243e0501ecdfef91c2bf178ff0df2e05b565692a4fc74b

data/train/dico_nli_train_track1_participant_labeled.csv
5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211

data/dev/dico_nli_dev_track1_participant_labeled.csv
80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511

data/dev/dico_nli_dev_track1_reference.csv
8d5cd11c1c60df3c9b59254d12a60fb2cf43e91dde01b46c8a8f296b806134bc
```

这些指纹记录本次看到的文件版本，不能证明历史训练使用的源码与当前源码完全相同，也不能替代训练记录。

## 附：本轮项目扫描范围

为后续逐篇整理，本轮递归遍历了项目内可访问文件的路径和大小；针对本篇进一步读取了 BERT 源码、模型配置、官方数据和历史记录。不是把所有实验文件内容都逐一研读。

扫描时排除了 `.git`、`__pycache__`、`.pytest_cache`、`.venv`、`node_modules`。以下数字为本篇写入前的快照：

| 项目 | 数量 |
|---|---:|
| 可访问文件 | 22140 |
| Python 文件 | 721 |
| Markdown 文件 | 159 |
| CSV 文件 | 12410 |
| JSON 文件 | 4981 |
| `.pth/.pt/.safetensors/.bin/.ckpt` 文件合计 | 45 |

45 个文件不能写成“45 个已训练模型”：其中包含 CPU 测试小模型、特征缓存和 `training_args.bin`。项目 `model/` 下实际找到 7 个约 736 MB 的 Lexical `best_model.pth`，分别在 `run01`、`run02_branch_lr`、`run03`、`run03_branch_lr5e4`、`run04_e_weight`、`run04_weight`、`run05_PAIRweight` 的 `lexical_seed_42` 子目录中；它们不是 BERT baseline 权重。

项目外的 D 盘模型库不计入以上数字；本篇只定点检查了 BERT 源码所指向的预训练目录。历史报告中的模型路径也不自动等于现存权重。

以下三个目录扫描时报拒绝访问，未改变其权限，也未读取内部内容：

```text
artifacts/dico_lora_check_qu_171wr
experiments/deberta_manual_aug/dico_manual_check_em13pphd
experiments/ensemble_selected/outputs/tmp29tq73du
```

后续文档应继续分别整理普通 DeBERTa、Token Alignment／Real Pair CE、Lexical、数据增强和融合，并为每一版独立核对源码、训练数据、选模指标、预测和权重状态。
