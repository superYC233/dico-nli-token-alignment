# DiCo-NLI：DeBERTa Token Alignment

本项目用于 **SemEval-2027 Task 2 DiCo-NLI 的 Track 1（英语）**：给定两个短语，判断它们是正向蕴含、反向蕴含、等价，还是其他关系。

最终采用单个模型：

**DeBERTa-v3-base + Token Alignment + Real Pair CE + Pair-aware inference**

在官方 DEV 的 660 条样本上，最终 **Weighted F1 = 0.9248**。训练使用完整官方 TRAIN 的 3042 条样本，不加入增强数据。

## 1. 从哪里开始

| 想做什么 | 看这里 |
|---|---|
| 快速核对最终分数，不训练模型 | [evaluate.py](evaluate.py) |
| 从预训练模型开始重新训练最终方案 | [train.py](train.py) |
| 查看最终配置、预测和提交包 | [artifacts/final](artifacts/final/) |
| 了解项目怎么一步步发展到现在 | [项目回顾](docs/DICO_NLI_SIMPLE_REVIEW.md) |
| 找以前的实验脚本 | [文件目录索引](docs/FILE_INDEX.md) |
| 查看完整实验过程和局限 | [完整实验记录](DICO_NLI_FINAL_EXPERIMENT_RECORD.md) |

## 2. 任务是什么

模型输入为 `text1 = A`、`text2 = B`，输出以下四种标签之一：

| 简写 | 官方标签 | 含义 |
|---|---|---|
| F | `FORWARD_ENTAILMENT` | A 蕴含 B，反过来不成立 |
| B | `BACKWARD_ENTAILMENT` | B 蕴含 A，反过来不成立 |
| E | `EQUIVALENCE` | A、B 双向蕴含，语义等价 |
| N | `NEGATIVE_OTHER` | 不属于以上三类 |

例如，`train station` 比 `station` 多了一层限定：前者可以推出后者，后者不能一定推出前者。因此，按这个方向输入时应判断为 F，交换输入后应判断为 B。

## 3. 最终模型做了什么

### DeBERTa + Token Alignment

主干使用已经经过 NLI 训练的 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`。它同时提供两路信息：

```text
输入短语 A、B
      │
NLI-pretrained DeBERTa-v3-base
      ├── CLS 表示 → 四分类分数
      │
      └── Token 表示 → 128 维投影 → A/B 双向软对齐
                  → 比较原表示、对齐表示、差、乘积
                  → mean/max pooling → 四分类分数

两路分数相加 → 最终四分类 logits
```

只看 CLS，相当于判断两个短语的整体关系。Token Alignment 则让模型进一步比较具体的词片段。

还是 `train station` 和 `station`：模型可以发现 `station` 与 `station` 对应得比较好，而 `train` 与对方的匹配情况不同，再结合上下文判断这个差异是否改变蕴含关系。当前实现是软对齐，**没有显式的 NULL／未匹配槽位**；它也不能保证自动理解所有修饰词和同义改写。

### Real Pair CE

训练数据中，同一个真实 `pair_id` 通常有正反两条样本：

```text
A → B：FORWARD
B → A：BACKWARD
```

除了普通四分类训练，模型还把这两条真实样本放在一起考虑，对 `FB / BF / EE` 三种组合计算交叉熵。这让训练目标与正反方向之间的关系一致。

每个 epoch 先完成全部样本的四分类训练，再完成真实 Pair 训练。这两个阶段分别更新参数，Pair 阶段的损失权重为 `0.5`。

### Pair-aware inference

预测时，完整的正反 Pair 联合选择分数最高的 `FB / BF / EE` 组合。只有一条的样本保留模型自己的四分类结果，**没有“没有 Pair 就判为 N”的规则**。

这个约束可以保证方向兼容，但不能保证预测在语义上正确，因此需要同时看 F1 和 HardCons。

## 4. 数据与最终结果

| 数据 | 行数 | 用途 |
|---|---:|---|
| 官方 TRAIN | 3042 | 模型训练，包含 1282 组真实 Pair |
| 官方 DEV | 660 | 选择最佳 epoch、评估最终结果 |

最终方案保留官方 TRAIN 的全部样本，不删除官方数据、不加入人工增强样本。`data/internal_split*` 是历史实验从 TRAIN 中划出的内部数据，**不是官方 DEV，也不参与最终方案的训练划分**。

官方 TRAIN 与 DEV 存在少量规范化文本对重合，最终运行沿用官方原始划分。这里报告的是经过开发集选模后的 **DEV 成绩，不是独立 TEST 成绩**；具体说明见完整实验记录。

| 指标 | 最终结果 |
|---|---:|
| Weighted F1 | **0.9247935854** |
| Accuracy | 0.9242424242 |
| SoftCons | 1.0000000000 |
| HardCons | 0.9241877256 |
| 随机种子 | 42 |
| 总训练轮数 / 最佳轮次 | 10 / 7 |

配置保存在 [config.json](artifacts/final/config.json)，完整评分保存在 [scores.json](artifacts/final/scores.json)。这是已保存运行的结果，重新训练可能受运行环境和随机性影响。

## 5. 如何运行

### 环境

评分入口只使用 Python 标准库和项目内的官方 scorer，需要 Python 3.10 或以上，不需要加载 PyTorch 模型。

训练主要依赖 `torch`、`transformers`、`pandas`、`numpy`、`scikit-learn`、`tqdm`、`sentencepiece`、`protobuf` 和 `safetensors`。本机当前环境如下，供复现时参考：

```text
Python          3.10.21
torch           2.11.0+cu128
transformers    5.17.0
pandas          2.3.3
numpy           2.2.5
scikit-learn    1.7.2
tqdm            4.70.0
sentencepiece   0.2.2
protobuf        5.29.6
safetensors     0.8.0
```

PyTorch 的 CUDA 构建需要与运行机器匹配；`starter_kit/requirements.txt` 是起始代码的依赖文件，不是本项目最终实验的环境锁定文件。整理文件后没有重新安装环境或重新训练。

### A. 只核对已有结果

在项目根目录运行，或直接右键运行 `evaluate.py`：

```bash
python evaluate.py
```

它会读取已保存的最终 DEV 预测，调用官方 scorer，预期输出：

```text
Weighted F1：0.924794
SoftCons：1.000000
HardCons：0.924188
```

评分文件写入 `results/final_baseline/`。这是对预测 CSV 的评分，**不会加载 checkpoint 重新生成预测**。

也可以对自己的预测评分：

```bash
python evaluate.py --predictions path/to/predictions.csv --output-dir results/my_run
```

默认参考答案仍是官方 Track 1 DEV；其他数据需要同时使用 `--gold` 指定相应参考文件。

### B. 重新训练最终方案

确认官方数据文件位于：

```text
data/train/dico_nli_train_track1_participant_labeled.csv
data/dev/dico_nli_dev_track1_participant_labeled.csv
```

然后运行，或右键运行 `train.py`：

```bash
python train.py
```

默认 `USE_AUGMENTATION = False`。脚本会加载 NLI 预训练主干，训练 10 轮，评估各轮 DEV 分数，并导出最佳预测。本机继续使用原来的 D 盘预训练模型缓存；其他机器可以通过 `DICO_MODEL_CACHE` 环境变量指定缓存目录，未配置时使用 Hugging Face 默认缓存。

| 主要参数 | 值 |
|---|---|
| Row / Pair batch size | 16 行 / 16 组 |
| 最大输入长度 | 128 tokens |
| 优化器 / 学习率 / weight decay | AdamW / `2e-5` / `0.01` |
| 四分类权重 F/B/E/N | `[1, 1, 1, 1.5]` |
| Pair 权重 FB/BF/EE | `[1, 1, 1.25]` |
| Pair loss 权重 / label smoothing | `0.5` / `0.05` |
| Alignment 维度 / temperature | 128 / `0.1` |
| Compare 激活函数 | ReLU |

新训练默认写入：

```text
model/deberta_token_alignment_reproduction_control_seed_42/
```

训练后的选模函数会只保留该运行目录中的最佳 `epoch_*.pth`，清理其余 epoch 文件。重复训练前请更换 `RUN_DIR`，避免覆盖上一次运行。已提交的最佳权重保留在另一个目录：

```text
model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth
```

`model/` 中的权重仅保存在本地，已加入 Git 忽略规则，不随源码仓库上传。

## 6. 目录说明

```text
.
├── train.py                        # 最终方案训练入口
├── evaluate.py                     # 已保存预测的官方评分入口
├── project_paths.py                # 项目路径与缓存配置
├── artifacts/final/                # 最终配置、DEV 预测、评分、提交包
├── data/                           # 官方数据及历史内部划分、增强数据
├── docs/                           # 项目回顾、目录索引、迁移清单
├── experiments/                    # 历史实验，按方向分类
│   ├── baselines/
│   ├── pair_training/
│   ├── alignment/
│   ├── auxiliary/
│   └── augmentation/
├── tools/                          # 下载、数据处理、分析和历史预测工具
├── evaluation_functions/           # 官方评分代码
├── starter_kit/                    # 起始代码
├── predictions/                    # 历史预测 CSV
├── results/                        # 评分结果
├── submissions/                    # 已保存的提交文件
├── model/                          # 本地 checkpoint，Git 忽略
└── DICO_NLI_FINAL_EXPERIMENT_RECORD.md
```

历史脚本保留各自的实验配置，有些需要已有 checkpoint 或外部模型。它们用于回顾和对照；复现最终方案使用根目录的 `train.py`。

## 7. 最终提交文件

[submission_track1.zip](artifacts/final/submission_track1.zip) 保存的是最终 **Track 1 DEV 预测包**，ZIP 根目录只有：

```text
track1_predictions.csv
```

CSV 只有 `instance_id,label` 两列，每条 DEV 样本对应一个官方标签。该包不是未公开 TEST 的预测，也不包含训练代码或模型权重。

## 8. 已尝试的方向与剩余问题

项目尝试过 Large、ModernBERT、NULL alignment、局部组合、attention pooling、辅助任务、边界损失和增强数据。它们没有建立足以替换最终方案的可靠优势；部分历史高分存在比较条件或增强来源方面的问题，详见实验记录。

最终模型的 50 条错误中：

- F/B → E：20 条。
- E → F/B：16 条。
- F ↔ B：6 条。
- N 错误：8 条。

主要难点是：**词面差异到底只是等价改写，还是改变了信息包含关系。** 后续研究应先在固定配置、多随机种子和独立评估上确认改进，再考虑从训练数据构造可靠的语义对照监督。

