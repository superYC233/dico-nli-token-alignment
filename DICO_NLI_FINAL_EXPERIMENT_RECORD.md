# DiCo-NLI 完整研究与实验记录

> 整理日期：2026-09-29。任务：SemEval-2027 Task 2 DiCo-NLI，Track 1 英语。
>
> **最终方案已锁定：DeBERTa-v3-base NLI + Token Alignment + Real Pair CE + Pair-aware inference。**
>
> 本次只扫描、核对和编写文档，没有修改训练代码、数据或模型，没有重新训练或运行模型推理。记录中的重算分数来自已有预测 CSV。

## 目录

1. [结论与证据口径](#s1)
2. [任务、数据与验证协议](#s2)
3. [方法演化路线](#s3)
4. [Baseline 到 Real Pair CE](#s4)
5. [Token Alignment 详解](#s5)
6. [Final Model：最终配置与复现定位](#s6)
7. [最终结果与错误分析](#s7)
8. [未采用的结构与训练探索](#s8)
9. [Boundary、Bi-Entailment、Entailment Auxiliary](#s9)
10. [数据增强的演化与有效性](#s10)
11. [关键消融与证据边界](#s11)
12. [Paper Writing Notes](#s12)
13. [Lessons Learned](#s13)
14. [完整结果表与产物索引](#s14)
15. [当前脚本配置索引](#s15)
16. [证据冲突、缺项与归档指纹](#s16)

<a id="s1"></a>
## 1. 结论与证据口径

### 1.1 最终选择

最终保留一个模型：NLI 预训练 DeBERTa-v3-base，使用 CLS 四分类分支和显式 Token Alignment 分支，训练采用四分类 CE 与真实 Pair CE，推理采用 pair-aware 联合选择。

官方训练集完整 **3042 行**；官方 DEV **660 行**；不加入合成样本，不按 DEV 删除官方训练样本。最终采用 seed 42、训练 10 epoch、DEV 选择 epoch 7 的运行：

| 指标 | 最终数值 |
|---|---:|
| Pair-aware Weighted F1 | **0.9247935854136166** |
| Accuracy | 0.9242424242424242 |
| Soft consistency | 1.0000000000000000 |
| Hard consistency | 0.9241877256317690 |
| 正确样本 / 全部样本 | 610 / 660 |
| 完整正确 pair / 可逆 pair | 256 / 277 |

这是**最终选定方案的开发集结果**，不是独立测试集成绩，也不是所有历史文件中的绝对最高数字。NULL-M 的某次官方 DEV 运行达到 0.9290；早期 manual 增强达到 0.9335，但存在样本构造来源问题；内部验证集还出现过 0.9428。这些不能删去，也不能混称为同一方案的结果。

最终选择遵循用户明确的模型范围：以已经复现的 Token Alignment 主干为研究结论；NULL、Boundary、二元蕴含分解、辅助任务和数据增强保留为探索记录。没有足够证据宣称最终方案在所有 seed 上优于全部替代模型。

### 1.2 本次读取范围与可信度

扫描了项目根目录 49 个 Python 脚本、数据和模型目录、18 个 `scores.json`、已有诊断 CSV、评估器及两个 README；进一步沿脚本中的绝对 `RUN_DIR` 查到 `D:/SemEval_experiments`。共核对 64 个实验目录，识别 67 个具有正确提交列且能匹配已知验证集 ID 的预测 CSV。其中 53 个官方 DEV 预测用项目评估器重新计算，结果与独立按 ID 计算一致；其余 14 个对应内部验证集。

项目内未找到完整逐 epoch `.log` 日志。部分训练曲线来自本对话粘贴的控制台输出；层级模型和阶段探针有单独 CSV。未用文件修改时间编造精确实验日期，以下采用**方法演化顺序**，并非逐日实验日志。

证据分层：

- **P：预测复核**。读取保存的预测，按 `instance_id` 与对应 gold 对齐重算；可确认这份预测的质量，不能仅凭 CSV 确认历史所有超参数。
- **S：评分文件**。读取 `results/.../scores.json`；与 P 一致时互相佐证。
- **C：当前代码**。确认当前定义和开关；脚本曾覆盖修改，不自动代表历史运行源码。
- **H：对话日志**。用于补充 epoch、曲线和操作过程，正文显式注明。
- **缺失**：当前项目记录中未找到准确数值，需要人工补充。不会用相邻实验或文件夹名字填补。

<a id="s2"></a>
## 2. 任务、数据与验证协议

### 2.1 输入、标签和反向关系

模型对一组有序短语 `(A, B)` 判断关系，而不是只判断“相似不相似”。

| label | id | 含义 | 交换 A/B 后 |
|---|---:|---|---|
| FORWARD_ENTAILMENT（F） | 0 | A 蕴含 B，单向关系 | B |
| BACKWARD_ENTAILMENT（B） | 1 | B 蕴含 A，单向关系 | F |
| EQUIVALENCE（E） | 2 | 双向蕴含 / 语义等价 | E |
| NEGATIVE_OTHER（N） | 3 | 其他关系 | 不能由本数据的单行记录自动推出反向标签 |

数据字段：`instance_id` 为行级唯一标识；`pair_id` 为关系组标识；`text1_lang`/`text2_lang` 是语言；`text1`/`text2` 是实际输入；`label` 是监督标签。评估 reference 另有 `reverse_pair_id` 指向反向行的 `instance_id`，不要把它误认为普通 `pair_id`。

当前发布数据中，F/B 和 E 都具有两行反向记录；N 是 singleton。这是观察到的数据结构，**最终模型没有把“找不到反向行”直接判成 N**。singleton 仍使用四分类输出。`pair_id` 不拼到文本输入，不作为分类特征；仅用于真实配对训练及当前 pair-aware 推理分组。

### 2.2 数据路径和统计

路径以下均相对项目根目录 `G:/pycode/SemEval/Task-2-DiCo-NLI-main`。

| 数据 | 路径 | 行数 | F | B | E | N | 两行组 / 单行组 |
|---|---|---:|---:|---:|---:|---:|---:|
| 官方 train | `data/train/dico_nli_train_track1_participant_labeled.csv` | 3042 | 881 | 881 | 802 | 478 | 1282 / 478 |
| 官方 DEV | `data/dev/dico_nli_dev_track1_participant_labeled.csv` | 660 | 190 | 190 | 174 | 106 | 277 / 106 |
| internal train | `data/internal_split/train.csv` | 2433 | 708 | 708 | 634 | 383 | 1025 / 383 |
| internal valid | `data/internal_split/valid.csv` | 609 | 173 | 173 | 168 | 95 | 257 / 95 |
| clean train | `data/internal_split_clean/train.csv` | 2430 | 702 | 702 | 636 | 390 | 1020 / 390 |
| clean valid | `data/internal_split_clean/valid.csv` | 612 | 179 | 179 | 166 | 88 | 262 / 88 |
| trial 无标签 | `data/trial/dico_nli_trial_track1_participant_unlabeled.csv` | 797 | — | — | — | — | 336 / 125 |

官方 reference 位于各自 `data/train`、`data/dev` 下的 `*_reference.csv`。已有 `predictions/trial_track1_predictions.csv` 是无标签预测产物，不构成效果提升证据，也不等于已经提交比赛。

### 2.3 clean 到底是什么

`make_pair_split.py` **只读取官方训练集**，按无序文本对分组，再用 `GroupShuffleSplit(test_size=0.2, random_state=42)` 分成 2430/612。该版分组键是：

```python
tuple(sorted((str(text1).strip().casefold(),
              str(text2).strip().casefold())))
```

它把 `(A,B)` 与 `(B,A)` 分入同一集合，忽略大小写和首尾空格；**这份 split 脚本不忽略标点，也不删除内部所有空白**。输出名 clean 只表示通过这套特定规则检查，不能理解成没有任何近重复或语义相似数据。

后来的增强工具 `norm()` 使用 `re.sub(r"[^\w]+", "", text.casefold())`，口径更强。两套规范化不同，不能把增强工具的 `pair_key` 描述套在 split 脚本上。

本次只读核对发现：

| 对比 | 规范化口径 | 相同文本对键 | 涉及训练 pair_id | 涉及训练行 |
|---|---|---:|---:|---:|
| 官方 train vs 官方 DEV | strip + casefold | 7 | 7 | 13 |
| 官方 train vs 官方 DEV | 去非单词字符 + casefold | 8 | 8 | 15 |
| clean train vs clean valid | strip + casefold | 0 | 0 | 0 |
| clean train vs clean valid | 去非单词字符 + casefold | 2 | 2 | 3 |

这些是**数据自身存在的重合**，与“依据 DEV 错误人工写训练样本”是不同问题。用户最终要求完整保留官方 train，最终配置也确实这样做；论文应披露评估协议，不将这个 DEV 分数称为严格去重的独立泛化估计。

### 2.4 比较结果的正确方式

- `0.9398(clean valid, 612 行)` 与 `0.9248(官方 DEV, 660 行)` 不能直接比较高低；训练集也分别是 2430 与 3042 行。
- 严格排除留出重合的增强实验：2430−18+48=2460 行。它与未排除的 clean 基线同时改变了基础数据和增强，不能作为纯增强消融。
- 最后完整官方 train 控制实验：False=3042 行，True=3090 行；两者评估相同 660 行。这一对更直接回答当前 48 行增强是否有帮助。
- `shuffle=True` 已让行和 pair 在训练时随机打乱。无序 `pair_key` 用于分组/排重，不会自动把每个 batch 变成“正反向相邻”。真实 PairDataset 已把同组两行一起取出；“再次改训练顺序”的方案没有已完成的新结果。

<a id="s3"></a>
## 3. 方法演化路线

```text
BERT / DeBERTa 四分类
  ├─ 推理惩罚项、交换一致性约束
  └─ Pair-aware 联合选择 FB / BF / EE
       ↓
按每行构造交换输入的 Pair CE
       ↓
按真实 pair_id 聚合的 Real Pair CE
       ↓
N 四分类权重、EE pair 权重
       ↓
NLI-pretrained DeBERTa-v3-base
       ↓
Pair label smoothing + Token Alignment
       ├─ uniform alignment 对照 / seed 对照
       ├─ NULL、lengthnorm、Conv、attention pool、gate
       ├─ 大模型、原型、分层头、辅助监督、训练调度
       ├─ boundary ranking / 人工增强 / train-only 增强
       └─ Boundary Loss、双向二元蕴含与辅助目标
       ↓
关闭增强的完整官方 train 控制实验复现 0.9248
       ↓
最终冻结 Token Alignment + Real Pair CE + Pair-aware
```

上述分支并非都从完全相同设置出发。例如早期 Large 分支在 NLI-base alignment 之前就存在，后期 Large alignment 又重新做过。记录应沿组件和父版本理解，不能仅凭文件名排序。

<a id="s4"></a>
## 4. Baseline 到 Real Pair CE

### 4.1 BERT / DeBERTa：建立四分类基准

`bert_baseline.py` 使用本地 `google-bert--bert-base-uncased`；`deberta_baseline.py` 使用 `microsoft/deberta-v3-base`。基本方法为编码文本对，CLS→dropout→Linear(4)，用 row CE 学习四类。

两者当前训练函数均可见 AdamW、lr=2e-5、weight_decay=0.01、10 epoch。精确 batch 和当前开关见后面的源码索引。BERT 的历史最优分数没有可可靠归属的实验产物，**当前项目记录中未找到准确数值，需要人工补充**。不能把根目录中未带模型标识的 `penalty_0` 分数直接算给 BERT。

DeBERTa 原始评估 `results/deberta_seed_42/scores.json`：Weighted F1=0.843772，SoftCons=0.877256，HardCons=0.844765。其 pair-aware 产物为 0.897671 / 1.0 / 0.913357。两者说明结构化解码对当前发布数据有效，但仍需原始逐轮日志才能严格证明所有记录使用同一 checkpoint。

### 4.2 Pair-aware：保持两个方向互相兼容

给同一真实组的两行 logits `z1,z2`，只在下面三种组合中选最大者：

```python
score_fb = z1[0] + z2[1]
score_bf = z1[1] + z2[0]
score_ee = z1[2] + z2[2]
```

FB 输出 `(F,B)`，BF 输出 `(B,F)`，EE 输出 `(E,E)`。singleton 保留原始 argmax。该约束不需要 DEV gold label 来选组合，但依赖预测数据提供可用于配对的组结构，以及这些完整组确实属于可逆关系的任务设定。

效果：修正两个方向独立预测造成的不一致。限制：完整 pair 不允许 N，因而会掩盖部分 raw 四分类错误；`soft_cons=1` 是当前合法组合约束的结果，不能解读为每组语义都预测正确。

### 4.3 Consistency 与早期 Pair CE

交换一致性实验的两个保存结果约为 0.8383、0.8389，未超越后来的联合解码/真实 pair 方案。早期 `deberta_pair_ce.py` 的 ConsistencyDataset 对每一行人工构造交换输入，在非 N 行上产生三类 Pair CE，与 row CE 同 batch 联合训练。由于原数据本来含两个方向，一个原始关系可能被重复作为交换训练实例看到。

保存的 `deberta_pair_ce_w0p5_seed_42` 预测为 0.896569。**当前脚本已写成 pair_weight=0.7，RUN_DIR 也为 w0p7，不能把当前 0.7 赋给历史 w0p5 结果。**

### 4.4 Real Pair CE：使用数据里真实的反向行

PairDataset 选择 `pair_id` 计数为 2 的组，取这两条真实记录：

```text
row1=(A,B,F), row2=(B,A,B) → pair_target=0 (FB)
row1=(B,A,B), row2=(A,B,F) → pair_target=1 (BF)
row1=(A,B,E), row2=(B,A,E) → pair_target=2 (EE)
```

训练时对两条输入分别编码，用上述三个和分数计算 CE；没有另加独立 pair classifier。相较自动交换每一行，真实 pair 训练单位更清晰，官方 train 每轮有 1282 个 pair。

最终源码的配对标签构造对非 FB/BF 的两行组使用 `else: pair_label=2`。这依赖当前数据只有 EE 这一剩余组合；不是一个能自动容错任意标签组合的通用实现。当前数据的组结构支持该使用方式。

### 4.5 损失、权重和两阶段更新

最终每个 epoch：先完整遍历 3042 行的 row CE，再完整遍历 1282 组的 Pair CE。

```python
classification_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 1.0, 1.0, 1.5], device=device)
)
pair_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 1.0, 1.25], device=device),
    label_smoothing=0.05,
)
# Row 阶段
loss = classification_criterion(logits, labels)
# Pair 阶段：另一次 optimizer.step()
loss = 0.5 * pair_criterion(pair_logits, pair_labels)
```

不能把它画成每个 batch 都同时优化 `L_row+0.5L_pair`。两阶段共用模型和优化器，但执行顺序、样本量和优化器更新次数有区别。`pair_weight=0.5` 也不意味着最终影响精确为 row 的一半。

Pair smoothing 的软目标在三个组合上分配少量概率，避免过度尖锐的目标；loss 留在 0.3 左右并不能单独证明“没学会”。有平滑/类权重后也不能直接与无平滑 focal 或普通 CE 的绝对 loss 数值比较。

### 4.6 可追溯的早期演化数字

| 保存实验 | 官方 DEV F1 | 作用 / 结论 |
|---|---:|---|
| DeBERTa pair-aware | 0.897671 | 推理结合反向结构 |
| Real Pair CE，w=0.5 | 0.902996 | 真实组监督 |
| +N weight 1.5 | 0.907754 | N 召回改进，保留 |
| +N weight 1.75 | 0.896361 | 加权并非越大越好 |
| +N1.5、EE1.25 | 0.912384 | 加强等价组合监督 |
| NLI-base weighted Real Pair CE | 0.913283 | 换成任务相近的 NLI 预训练模型 |
| NLI-base Pair smoothing，保存文件 0p05 | 0.916043 | 该目录真实结果是 0.9160 |
| Token Alignment，seed42 | **0.924794** | 最终方案 |

“Pair smoothing=0.924559”曾出现在对话归属中，但当前该 smoothing 目录的评分和预测均为 **0.916043**；0.924559 在 GELU alignment / token_alignment_null 保存结果中出现。研究表按实际产物写，不复制错误归属。

<a id="s5"></a>
## 5. Token Alignment 详解

### 5.1 动机和整体结构

CLS 表示用于全局分类。加入 alignment 分支，让模型显式比较两侧对应 token 的表示、差异和共同成分，帮助区分等价改写与单向增加限定的信息。

DeBERTa 本身已经上下文化并能跨句交互，不能说旧模型完全没有局部语义或方向信息。这里添加的是**显式比较路径**，其必要性要靠实验验证。

```text
输入 [CLS] A [SEP] B [SEP]，总长度上限 128
                         │
           NLI-pretrained DeBERTa-v3-base
                H：[batch, length, 768]
              ┌──────────┴─────────────────────┐
              │                                │
           H[:,0,:]                      Linear 768→128
              │                                │
         Dropout 0.2                 cosine similarity / 0.1
              │                                │
         Linear 768→4                  A↔B masked soft alignment
              │                                │
          cls_logits                  [t, aligned, t−aligned, t*aligned]
              │                           每个 token 512维
              │                                │
              │                     共享 Linear 512→128 + ReLU
              │                                │
              │                     A mean/max 256 + B mean/max 256
              │                                │
              │                     LayerNorm512 → Dropout0.2
              │                                │
              │                     Linear512→4（初始权重/偏置0）
              │                                │
              │                           match_logits
              └────────────── + ───────────────┘
                         final_logits (F,B,E,N)
                     ┌────────────┴─────────────┐
                  Row CE                    Real Pair CE
                     │                 两行组合为 FB/BF/EE
                     └──── 同一模型，分阶段更新 ───┘

评估：读取 final_logits → pair-aware → 660 行标签 → 指标
```

图中的 ReLU 以最终成功复现运行的脚本为准。当前原名 `deberta_nli_base_token_alignment.py` 已改为 GELU；历史原版 CSV 与复现控制 CSV 完全相同，但无激活参数的 state_dict 本身不能识别训练时究竟用了哪种激活。

### 5.2 Projection 与 mask

`align_projection=Linear(768,128)`，输出 `tokens:[B,L,128]`。减小匹配及比较开销，提供一个可学习的关系比较空间；不是保留了所有 768 维信息，也不保证 128 是全局最优。

```python
is_sep = input_ids == self.sep_id
sep_count = is_sep.long().cumsum(dim=1)
valid = attention_mask.bool() & ~is_sep & (input_ids != self.cls_id)
mask_a = valid & (sep_count == 0)
mask_b = valid & (sep_count == 1)
```

第一段 SEP 前的有效 token 属于 A，第一和第二 SEP 之间属于 B。CLS、SEP 被显式排除，PAD 由 attention_mask 排除。这依赖当前 tokenizer 的双句特殊符号模板，换 backbone/tokenizer 时应重新核对，而不是机械复用。

代码使用子词 token，“词级对齐”是便于沟通的叫法，不等于经过词级标注的 word alignment。

### 5.3 相似度与双向 soft alignment

```python
normalized = F.normalize(tokens, dim=-1)
scores = torch.bmm(normalized, normalized.transpose(1, 2)) / 0.1
```

得到 `[B,L,L]` 的余弦相似度矩阵。temperature=0.1 把分布变尖锐；它衡量表示相似性，不是直接的蕴含概率。

```python
mask = target_mask.unsqueeze(1)
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
weights = torch.softmax(scores, dim=-1) * mask
weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
aligned = torch.bmm(weights, tokens)
```

A 的各 token 从 B 的有效位置取加权和，B 从 A 取加权和；比较完成后按对应 source mask 池化。对非空目标段，权重归一化为 1，因此没有显式“不匹配任何 token”的选项；NULL 分支就是为此提出的探索。

### 5.4 比较特征与方向

```python
features = compare_layer(torch.cat([
    tokens, aligned, tokens - aligned, tokens * aligned
], dim=-1))
```

四部分分别表达当前 token、对侧对齐内容、有符号差异、逐维交互，拼接为 512 维，经共享比较层压到 128 维。A/B 使用同一 compare_layer。

虽然 cosine 对称，但整个模型并不对称：输入顺序不同、差异有符号、A/B pooled 表示有固定拼接顺序，CLS 也来自有序文本对。因此不能说“cosine 对称，所以模型绝对学不了 F/B”。不足在于相似度不是“是否改变蕴含”的直接监督。

### 5.5 mean/max pooling 和融合

mean 只对有效 token 求平均；max 把无效位置屏蔽为极小值，空段显式回退零向量。每侧 mean128+max128=256；两侧拼成512。

```python
pair_feature = torch.cat([
    self._pool(features_a, mask_a),
    self._pool(features_b, mask_b)
], dim=-1)
match_logits = self.match_output(pair_feature)
final_logits = cls_logits + match_logits
```

匹配头是 LayerNorm512、Dropout0.2、Linear512→4。末层 weight/bias 初始化为0，使起点是 `final_logits=cls_logits`，减少随机新分支立即改变全局决策的影响；**CLS 头本身是新初始化四分类头，不能把起点称为直接保留 NLI 三分类预测**。

零初始化只在初始时屏蔽匹配 logits；第一次反向传播末层可获得梯度，其前面的匹配模块梯度初始受零末层抑制，后续更新才逐渐参与。代码用 `torch.random.fork_rng(devices=[])` 包住新分支初始化，减少分支创建额外消耗 CPU RNG 对其他随机序列的影响；这不等于 GPU 训练完全确定性。

### 5.6 目前的结构局限

对齐后比较层按 token 独立计算，再 mean/max 压缩；没有额外显式短语组合层。不过 backbone 已经带上下文，不能把它描述成完全不理解相邻词。

当前更有证据支持的瓶颈是：突出差异可以减少一侧混淆，却增加另一侧。最终需要区分“等价改写的差异”和“改变信息包含关系的差异”，仅提高差异幅度或移动 E 的分数通常不够。

<a id="s6"></a>
## 6. Final Model：最终配置与复现定位

### 6.1 冻结配置

| 项目 | 最终成功复现脚本的当前值 |
|---|---|
| backbone | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| encoder 加载 | AutoModel；float32；use_safetensors=True |
| 原 NLI pooler/classifier | 不使用；加载报告 unexpected keys 对应这部分 |
| 四分类头 | CLS→Dropout0.2→Linear(hidden,4) |
| SEED | 42（Python、NumPy、torch、CUDA seeds） |
| row batch | 16；官方 train 每轮191 batch |
| pair batch | 16 组，即最多32条有序文本对；每轮81 batch |
| DEV batch | 16；42 batch |
| max_length | 128，双句总 token 上限，truncation=True，padding=max_length |
| epoch | 10；最终 DEV 最优7 |
| optimizer | AdamW |
| lr / weight_decay | 2e-5 / 0.01 |
| lr scheduler | 最终训练函数没有 |
| AMP / 梯度累积 | 最终训练函数没有 |
| row class weights | F/B/E/N = 1/1/1/1.5 |
| row label smoothing | 默认0 |
| Pair CE weight | 0.5 |
| Pair class weights | FB/BF/EE = 1/1/1.25 |
| Pair label smoothing | 0.05 |
| alignment dimension / temperature | 128 / 0.1 |
| compare activation | **ReLU，按成功复现脚本锁定** |
| match branch dropout | 0.2 |
| 数据增强 | False |
| 数据 | 官方 train 全3042 + 官方 DEV 全660 |
| 训练顺序 | row完整一遍→pair完整一遍；两个 loader 均 shuffle=True |
| 保存策略 | 每轮保存；评估后当前函数删除非最优 epoch 文件 |

### 6.2 最终文件

- 研究方法原名：[deberta_nli_base_token_alignment.py](deberta_nli_base_token_alignment.py)。当前 GELU，主入口训练调用已注释，不能无说明地将它当成最近复现运行的完整配置。
- **最近成功复现的执行脚本**：[deberta_nli_base_token_alignment_trainonly_aug_dev.py](deberta_nli_base_token_alignment_trainonly_aug_dev.py)，`USE_AUGMENTATION=False`。尽管文件名含 aug，这个开关下没有增强。
- 对应模型：[epoch_7.pth](model/deberta_nli_base_token_alignment_official_full_control_seed_42/epoch_7.pth)。本次确认文件存在，大小736096223 bytes；未加载权重重新推理。
- 对应预测：[dev_predictions_pair_aware.csv](model/deberta_nli_base_token_alignment_official_full_control_seed_42/dev_predictions_pair_aware.csv)。
- 原基线评分：[scores.json](results/deberta_nli_base_token_alignment_seed_42/scores.json)。
- 原基线保存目录：`model/deberta_nli_base_token_alignment_seed_42`。

两份基线预测 CSV 字节完全相同，按官方 reference 重算均为0.9247935854。最佳 epoch7 由当前保留文件及对话中完整10轮日志共同支持。仅保留一个 epoch 文件通常提示它被选中，但对其他实验不能单靠文件名冒充有完整最佳轮次日志。

本次不重命名或改写训练脚本。未来阅读时优先认配置、RUN_DIR、预测指纹，避免根据 `aug`、`clean` 或 `0p3` 字样猜当前行为。直接运行上述复现脚本会重新训练并写入原 RUN_DIR；这份记录不是要求现在再次运行。

### 6.3 最终10轮曲线（H）

来自用户最近粘贴的无增强控制运行；保存预测只对应最优轮，其他轮未重新评估。

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV F1 | .8902 | .9011 | .9139 | .8957 | .9138 | .8764 | **.9248** | .8944 | .9105 | .8805 |

最终选择的是经过10轮搜索的开发集峰值，不能把它当成预先固定7轮在未知测试集上的无偏成绩。后期 row loss 很低但 DEV 起伏明显，说明优化训练目标和泛化并不等价；现有记录不足以把全部波动归因于某一个模块。

<a id="s7"></a>
## 7. 最终结果与错误分析

### 7.1 分类报告

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| F | .9365 | .9316 | .9340 | 190 |
| B | .9365 | .9316 | .9340 | 190 |
| E | .8587 | .9080 | .8827 | 174 |
| N | 1.0000 | .9245 | .9608 | 106 |
| macro avg | .9329 | .9239 | .9279 | 660 |
| weighted avg | .9262 | .9242 | .9248 | 660 |

### 7.2 混淆矩阵：行是真实，列是预测

| gold \ pred | F | B | E | N |
|---|---:|---:|---:|---:|
| F | **177** | 3 | 10 | 0 |
| B | 3 | **177** | 10 | 0 |
| E | 8 | 8 | **158** | 0 |
| N | 1 | 1 | 6 | **98** |

总计50条错误：F/B→E有20条，E→F/B有16条，F↔B有6条，N错误8条。**E与F/B边界占36/50=72%的错误**，方向完全反转只占6/50=12%。42条 paired错误对应21组错误，另8条为N singleton错误。

N precision=1.0 部分受当前解码约束及发布数据结构影响：非 N 都在被强制FB/BF/EE的paired部分。不能把它单独视为 raw N 分类器没有任何误报。

### 7.3 一致性指标的精确定义

根据 `evaluation_functions/metrics.py`：

- SoftCons：一组预测是否反向兼容，例如 F/B、B/F、E/E；允许“兼容但都错”。
- HardCons：同组两条预测是否**同时等于各自 gold**。
- Weighted F1：所有行四分类 F1 按真实 support 加权；“Pair-aware Weighted F1”只是先做 pair-aware 再计算普通 weighted F1，并非新增一种F1公式。

最终 SoftCons=277/277=1；HardCons=256/277≈0.924188。两者不能混用；HardCons也不是模型 raw 输出的一致率。

### 7.4 差异增强实验实际改变了哪些错误

以下均为官方 DEV 660行预测重算，不把 clean 混入。

| 实验 | F1 | F/B→E | E→F/B | F↔B | N错 |
|---|---:|---:|---:|---:|---:|
| 最终 Token Alignment | .924794 | 20 | 16 | 6 | 8 |
| Attention pooling | .919820 | 14 | 22 | 10 | 7 |
| Contrast排序 | .925905 | 16 | 20 | 8 | 5 |
| Boundary保存预测 | .914506 | 12 | 32 | 6 | 6 |
| Entailment auxiliary | .883745 | 18 | 26 | 4 | 28 |
| train-only增强24组 | .914775 | 32 | 14 | 6 | 5 |

Attention 的 E/F/B 总边界错误仍是36条，没有净减少；Contrast也仍是36条，整体小幅收益主要来自N，而不是证明成功解决了等价边界。Boundary的边界错误变成44条。增强改善E召回和N，但F/B→E增加12条，整体损失更大。

因此，观察支持“当前尝试经常在两类边界之间做取舍”，不支持“只要突出局部差异就能全面提升”。词面差异是否改变蕴含仍是主要语义难点。

<a id="s8"></a>
## 8. 未采用的结构与训练探索

### 8.1 Large / ModernBERT

目的：检验更大容量或不同NLI预训练能否补足语义知识。Large使用 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`；ModernBERT脚本指向本地 `tasksource--ModernBERT-large-nli`。

- Large alignment保存结果0.898715，E recall=134/174≈0.7701。
- Large NULL-v2保存结果0.908271，E recall=140/174≈0.8046。
- Large type auxiliary评分文件0.899774。
- ModernBERT以及若干早期Large分支：有代码，但当前项目记录中未找到准确数值，需要人工补充。

后期Large NULL十轮对话日志最优在第3轮，继续到10轮没有超越；因此不能把这次失败仅解释为“还没跑够”。但这些运行同时使用了不同batch、分组LR和预训练数据，不是只改变参数量的严格容量实验。结论限于当前配置成本更高且未超越目标基线，不证明large普遍不如base。

### 8.2 Type / direction auxiliary、复杂pair头、原型和分层

- Type auxiliary试图把N/方向/等价结构显式监督；普通base保存结果0.892869，未超越对应weighted Real Pair主线。Large type有0.899774，但不是最终NLI-base alignment对照。
- Direction auxiliary让F/B/E参与辅助分类、N不参与；普通方向辅助保存预测0.908948，校准后0.911788。NLI-base direction脚本另有权重0.2；不能将其他目录的0.916043随意归给它。
- 多原型头：把CLS投影到关系空间，每类设多个原型，以相似度聚合分类；k1=.908659，k3=.898678。提高可表达性没有在这些运行上改善最终关系分类。
- Hierarchical当前模型有关系存在性head和方向head，保存预测.883999；当前代码Pair路径还使用no_grad/特征detach。由于源码经历修改，这不是可将所有变化归因于“分层思想”的干净实验。
- 复杂Large pair head：源码可见，当前项目记录中未找到准确数值，需要人工补充。

这些尝试说明更结构化的头不是自动收益；不少辅助标签只是原四分类标签的再编码，没有新增能解释“改写是否保持语义”的证据。

### 8.3 NULL / unmatched及length normalization

动机：常规softmax强迫每个token在对侧找到匹配；NULL添加“无对应”的去向。尝试包括scalar NULL、NULL强度、分支LR、长度归一化等。

官方DEV保存结果：早期NULL=.901876；分支LR目录=.902995；NULL-M seed42=.928987、seed7=.910091；lengthnorm目录=.912274。另一个 `token_alignment_null_seed_42` 保存预测=.924559。目录中的NULL字样不能替代历史配置核验。

NULL-M seed42确实高于最终.924794；不能写成“NULL从未提升”。但seed7较弱，当前两seed有限对照不能证明其稳定优势，且内部clean的后续真实修改运行只有.902921。此次按用户决定保留更简单的最终结构。

原token clean与一次NULL clean预测完全一致（均.939778），早期internal也同为.922583。分数或预测相同只能说明观察相同；缺少当时源码快照，不能仅凭相同结果断言NULL一定未启用。后续run2需要单列。

### 8.4 Conv composition / attention pooling / gate

这些分支针对alignment比较后马上池化的问题：Conv让邻近比较特征交流；attention pooling学习选择重要位置；gate调节分支影响。

- compose官方DEV .918678；attention pooling .919820，均未超越.924794。
- gate clean .927201，只能和clean token .939778比较。
- 当前compose/attention/gate训练源码在根目录已不在，保存预测仍在；模块细节来自历史讨论，不能恢复到精确每层配置。

Attention减少F/B→E却增加E→F/B，说明更集中地看差异未带来两方向同步改善。不能以此证明模型已准确理解差异类型。

### 8.5 Focal、EE权重、学习率与训练顺序

- Focal clean=.910461，N只正确66/88。不能用更小focal训练loss与原平滑CE比较“学得更好”。
- EE margin/加权类clean运行=.918038；文件名和历史日志显示目标在E边界，但没有完整保留源文件，具体实现以缺失标注处理。
- Group LR alignment=.919161；早期scheduler=.883898，低LR15epoch NLI-base=.889658。
- Interleaved方向实验=.883829；Large interleaved有代码、缺可核对分数。
- R-Drop internal=.907927，对应原internal=.922583，不能与官方DEV数字直接比较。

阶段探针 `model/deberta_nli_base_stage_probe_seed_42/stage_probe.csv` 提供了一项可解释证据：

| epoch | row后 Pair F1 | Pair后 Pair F1 | row后N正确 | Pair后N正确 |
|---|---:|---:|---:|---:|
| 1 | .865161 | .866915 | 78 | 67 |
| 2 | .880443 | .880773 | 86 | 80 |
| 3 | .881978 | .883532 | 81 | 78 |

这些轮次中Pair阶段提升pair-aware F1但降低N正确数，符合阶段目标可能相互影响的假设；它来自早期模型，不证明最终alignment的波动都由此导致。最终仍保留已复现的两阶段训练，没有把interleaved写成最终方法。

### 8.6 诊断、校准与边界对照挖掘

早期 NLI-base weighted 模型保存的 `deep_analysis/all_dev_analysis.csv` 含 raw_pred 与 pair_pred。本次复核：raw Weighted F1=.8214081201，pair-aware=.9132831128；改变90条预测，纠正70条、破坏1条，净增加69条正确。这是**早期CLS-only NLI-base模型**的诊断，不能挪作最终alignment的raw结果。该CSV的`label`列是gold，不能误当预测而算出虚假的1.0分数。

`ee_calibration_scores.csv`记录方向辅助模型E惩罚0、0.5、1、2的F1分别为.908948、.911788、.902598、.899248；校准0.5略好，但没有超过当时weighted主线。校准只是改变分数偏好，不是新的encoder能力。

`probe_bidirectional_nli.py`用于检查现成NLI分数是否能区分等价边界。对话日志（H）报告全部pair AUC=.921105、原模型判错18组上的AUC=.375；当前未找到独立保存的probe分数文件，因此保留为日志证据。错误子集很小，不能据此否定全部NLI教师；它至少没有提供“教师恰好能修复当前错误”的支持。

`mine_boundary_pairs.py`寻找共享短语、一个等价关系和一个方向关系的候选对照。D盘现存初始候选26行，full/raw/unreviewed各89行；`approved`列与文件名本身不等于完成语义核验。后续contrast运行日志报告18个核验锚点，额外排序loss，官方DEV=.925905；其训练源码未在当前根目录保留，精确权重和候选快照对应关系需补充。当前manual的两份triplet CSV均为18行，不能替代更早40-pair增强运行的原始输入快照。

<a id="s9"></a>
## 9. Boundary、Bi-Entailment、Entailment Auxiliary

### 9.1 Boundary Loss

动机：直接监督“方向关系还是等价”，针对F/B与E边界。使用三类pair logits再聚合为二类，不增加独立编码器。

当前 `deberta_nli_base_token_alignment_boundary.py`：

```python
directional_score = torch.logsumexp(pair_logits[:, :2], dim=1) - log(2)
boundary_logits = torch.stack([directional_score, pair_logits[:, 2]], dim=1)
boundary_target = (pair_label == 2).long()
loss = 0.5 * pair_loss + 0.1 * boundary_loss
```

纯logsumexp表示FB和BF两个互斥事件的合计分数；减log2变为log-mean-exp，对directional聚合减去常数。**logsumexp并非数学错误**；这两者改变了二类目标的隐含偏好。若两个directional logits都等于E，纯logsumexp会多出log2；减log2消除这项数量差异，但不自动保证更好的语义边界。

当前代码 `boundary_weight=0.1`，目录却仍为 `boundary0p3_seed_42`。保存预测F1=.914506，保留epoch10文件；没有完整run config可确认这份预测究竟对应修改前0.3还是当前0.1。logsumexp与log-mean-exp各自独立分数**当前项目记录中未找到准确数值，需要人工补充**。

保存结果F/B→E=12、E→F/B=32，E recall=.8161，低于基线.9080；边界错误合计44条高于36。可表述为“这一保存运行更偏向directional，未净改善边界”，不宜写成因果定论“所有boundary监督都只能移动阈值”。该目标没有新增语义标注，主要是重加权既有pair标签。

### 9.2 Pure Bi-Entailment

设两个判断分别为A是否蕴含B、B是否蕴含A，四种位组合映射F/B/E/N：

```text
(1,0)→F；(0,1)→B；(1,1)→E；(0,0)→N
```

`D:/SemEval_experiments/deberta_nli_base_token_alignment_bientail_seed_42/dev_predictions_bientail.csv`可核实：Weighted F1=**.851986**，SoftCons=.920578，HardCons=.873646；目录保留epoch3。原纯双二元训练实现未在当前根目录找到，`deberta_nli_base_entailment_classifier.py`现在已经是下一节辅助方案，不能用它反推纯二元运行的具体loss与权重。

| gold \ pred | F | B | E | N |
|---|---:|---:|---:|---:|
| F | 171 | 1 | 4 | 14 |
| B | 1 | 171 | 4 | 14 |
| E | 8 | 8 | 142 | 16 |
| N | 9 | 11 | 9 | 77 |

N recall=77/106=.7264，N precision=77/121=.6364；44条真实非N被判成N。它的SoftCons也不是1，说明保存结果没有受到与最终模型完全相同的FB/BF/EE硬限制，不能视为仅换训练目标、推理完全相同的消融。

**需要纠正的理论结论：**四类当然可以编码成两个二进制位。若进一步用两个独立Bernoulli的乘积构造四类概率，则联合分布被限制，有 `P(E)P(N)=P(F)P(B)`，比任意四分类分布自由度少；但这不意味着两位标签不能表示四类，也不是这次实验已经证明独立假设是唯一失败原因。能确认的是当前实现和训练/解码组合未达到最终基线。

### 9.3 Entailment Auxiliary

当前脚本保留四分类logits，通过logsumexp构造两个二类边际监督：

```text
A→B=yes: F或E；no: B或N
B→A=yes: B或E；no: F或N
L_entail = (CE_AB + CE_BA)/2
L = L_4class + 0.2 * L_entail
```

这里没有独立新增二元head，两个辅助分布由同一四分类输出聚合。理论上是重新强调标签分组，而不是新增知识。

保存F1=**.883745**，SoftCons=1，HardCons=.913357；E recall=148/174=.8506，N recall=78/106=.7358。N错误28条，明显超过最终8条。

**关键实现差异：**当前 `train_model_entailment_aux()` 只遍历row loader，没有Real Pair CE训练循环。因此这不能写成“在完全不动最终训练的情况下，只加0.2辅助loss就下降”。实际同时取消了Pair训练，compare层也为当前GELU版本，历史因果归属不够纯。最终不采用该运行，但保留这一限制以免论文误读。

<a id="s10"></a>
## 10. 数据增强的演化与有效性

### 10.1 早期synthetic与人工边界数据

`data/train/dico_nli_train_track1_augmented_v1.csv`现有5442行，对比原3042增加2400行；标签为F1181/B1181/E1802/N1278。对应保存weighted版本DEV=.896206。原始生成过程和质量核验材料未完整保留，不能只用文件名证明标注质量或无泄露。

早期manual40组80行运行得到.933544（H最佳epoch3，P/S一致）。后来已讨论候选设计与DEV错例相关的风险，因此**不作为可证明无DEV介入的提升证据，不进入最终方案**。当前manual脚本/CSV已变成其他版本，不能用今天通过的排重检查为早期历史运行补发“无泄露证明”。

随后18组triplet产生36对72行的clean实验，保存预测.924406；另一manual clean目录=.923818。它们与官方DEV .933544不是同一验证集，也不必然是同一批增强样本。

### 10.2 可审计train-only 24组

当前生成器：[make_train_only_boundary_augmentation.py](make_train_only_boundary_augmentation.py)。产物：[train_only_boundary_24pairs.csv](data/augmentation/train_only_boundary_24pairs.csv)。

24个无序pair、48条有序行：E12对（24行）、directional12对（F12+B12）。构造来源仅使用内部clean TRAIN的金标边；包含：

- EE传递7组：两条训练等价边共享端点，组合未共享端点。
- ED替换6组：用训练等价边替换有向边的一端，保持方向。
- manual改写/限定11组：人工根据所列训练边编写；程序对这类只做来源与重合检查，**不自动证明语义标注正确**。

每行保留 `source_pair_ids`、`derivation`。读取内部valid与官方DEV的文本/ID用于拒绝重合，不读取其label生成监督。检查生成文本、来源文本、来源ID与留出数据的规范化精确重合，以及与已有官方train文本对的重复。

这些检查能说明代码中的来源限制和精确排重结果，不能证明完全不存在语义近重复，也不能仅凭代码注释证明作者构思过程未受任何既往DEV知识影响。DEV文本用于排除与DEV标签用于造样本应明确区分。

### 10.3 最后直接可比的官方DEV实验

| 项目 | 无增强控制 | 加48行 |
|---|---:|---:|
| 官方train行数 | 3042 | 3090 |
| Pair数 | 1282 | 1306 |
| 官方DEV | 660 | 660 |
| 最佳epoch（H+保存文件） | 7 | 3 |
| Weighted F1（P） | **.924794** | .914775 |
| F/B→E | 20 | 32 |
| E→F/B | 16 | 14 |
| N错误 | 8 | 5 |
| 总错误 | 50 | 57 |

当前增强在这次完整官方协议下下降约1.00个百分点。它让E召回稍升、N改善，却显著增加F/B→E，不支持直接扩大量就会提升。训练样本数和batch数也随增强变化，随机顺序及优化轨迹会变；单seed不能证明每条新数据都无用，但足以说明这次不应采用它替换已复现基线。

clean/strictclean增强.938061与clean原版.939778也没有提供明确优势，且基础训练集排重不同。最终关闭增强，不继续把已排除的manual高分作为主结果。

<a id="s11"></a>
## 11. 关键消融与证据边界

### 11.1 对齐与均匀平均

| seed | Token Alignment | Uniform Alignment | 差值 |
|---|---:|---:|---:|
| 42 | .924794 | .915088 | +.009706 |
| 7 | .912365 | .891261 | +.021104 |

两个已保存seed中，基于相似度的alignment都优于uniform，支持有选择的跨文本匹配有用。两个seed的均值分别约.918579和.903174；这是已保存运行的描述性统计，不是充分统计显著性证明。完整历史配置快照缺失，不能声称除alignment外的所有运行细节都被严格锁定。

### 11.2 最終选择没有抹去反例

NULL-M seed42=.928987、seed7=.910091，两次平均约.919539，略高于上述token均值.918579。差异很小且样本仅2个seed，不能用“多seed均值更好”作为选择原token的理由。Contrast单次.925905也略高。最终选择依据是用户锁定的架构范围、原基线重复得到相同预测、结构更简单及证据/来源更清晰；不是宣称数学或统计上已压倒这些变体。

### 11.3 目前还没有成立的因果结论

- NLI-base weighted→final提升，不只涉及alignment，还包含smoothing等变化；不能把整个差值都算作alignment净贡献。
- 原型、分层、Large比较部分来自不同训练配置，不能当作统一预算全参数搜索后的最优对决。
- 内部valid被多轮查看后同样成为开发数据；名字clean不保证选择偏差消失。
- 数据增强筛选曾受DEV错误关注引导，后续精确去重不自动让早期运行成为无偏测试。
- final_soft_cons=1主要来自解码规则；若想量化编码器学到多少一致性，应另报告raw指标，但最终checkpoint的raw预测当前未保存，本次没有重新推理，不能拿早期CLS-only的raw .821408顶替。

<a id="s12"></a>
## 12. Paper Writing Notes

### Problem / Motivation

本任务要求分清等价、正向蕴含、反向蕴含和其他关系。最终错误的72%集中在等价与单向蕴含之间。动机是同时利用**文本内部的细粒度对应差异**和**数据提供的双向关系结构**。

### Method 可用描述

采用NLI预训练DeBERTa-v3-base；从编码后的token投影得到双向软对齐表示，拼接原表示、对齐表示、有符号差和逐维乘积，经共享比较层与mean/max池化产生四分类残差logits；与CLS logits相加。训练依次使用加权row CE与真实反向组上的三类Pair CE；推理在真实可逆组上选择FB/BF/EE，singleton按四分类预测。

### Evidence 应怎样写

1. 官方DEV最终Weighted F1=.924794，SoftCons=1，HardCons=.924188；明确是DEV选最优epoch。
2. 保存的两个seed中，token alignment优于uniform matching，支持匹配内容有贡献。
3. Large尝试未自动提升，边界/二元任务/数据增强也非稳定收益；描述实验条件，避免扩大为普遍规律。
4. attention与boundary的错误分布支持“单纯强调差异/重分配分数不足以同时改善E和directional”。
5. 不能写“binary标签编码必然损失四类信息”；应写当前独立建模、训练及解码方案未成功，存在分布约束和协议混杂。

### 论文写作候选表述

- “构建了结合显式token比较与真实反向样本监督的关系分类系统。”
- “在当前Track1实验中，分析了局部差异聚合、双向组监督和一致性解码之间的作用。”
- “报告了等价/单向蕴含边界的错误迁移，以及若干未带来稳定收益的结构与数据增强探索。”

这些是系统设计与实验分析的候选贡献，不是已核实的首次提出或SOTA。alignment/compare/pooling、label smoothing、成对约束本身有已有研究背景；正式论文需单独核对并引用原始文献。本次文档没有进行网络文献检索，因此不编造论文归属。

### 论文结果发布前还缺什么

列出限制供将来使用，而不是要求现在继续跑：最终配置的更多seed及方差、独立测试评估、严格同配置的模块消融、raw和pair-aware并列指标、历史版本/数据/权重快照、训练时实际软件与GPU版本。现有日志不能恢复历史精确环境；不要用今天安装版本冒充当时版本。

<a id="s13"></a>
## 13. Lessons Learned

1. **先固定可比较的数据协议。** 612行clean F1与660行DEV F1的混比造成过误判；最终采用官方train/DEV并明确保留发布数据重合。
2. **用预测和gold按ID核对。** 文件夹名曾与真实权重/数据集不一致；仅看运行名会误记结果。
3. **保存失败实验也保存条件。** Boundary目录0p3与代码0.1、原token GELU与复现ReLU、旧manual数据被替换都说明源码与结果需要一同归档。
4. **低训练loss不等于高DEV。** 最终第7轮最好，第10轮更低row loss却只有.8805；平滑CE和focal的loss值也不能直接比较。
5. **语义边界不能只靠强调差异。** Attention的E/F/B错误总数不变；Boundary反而增加，支持观察两方向净变化。
6. **Pair结构有用，但解码收益与模型能力要分开。** SoftCons被硬约束保证；HardCons和raw错误仍需关注。
7. **N加权有局部证据，强度不是越大越好。** 在同系列普通base保存结果中，1.5优于1.75；NLI-base的1.75也没有成功。
8. **扩大模型和堆任务没有自动收益。** Large alignment、二元蕴含及辅助任务本次未改善，不能据此推出理论不可能。
9. **增强需要语义质量和来源审计。** 48行train-only没有提升；早期manual高分不能抵消DEV介入风险。去重检查也不替代人工语义核验。
10. **避免重复覆盖/误跑。** 历史出现过多次10轮训练、目录缺失、第10轮保存失败及坏checkpoint读取；从日志只能确认I/O失败，不能无磁盘证据断言一定磁盘满。当前评估函数会删除非最佳checkpoint，解释了为什么很多目录只剩一个文件。
11. **多次DEV选择有偏差。** 本项目大量候选和epoch都使用DEV选择；最终0.9248应作为开发结果报告。
12. **此次停在清晰的方案上。** 最终四组件已经锁定，本文负责恢复证据，不再追加创新模块、调参或训练任务。

<a id="s14"></a>
## 14. 完整结果表与产物索引

下面由本次扫描到的预测文件自动生成。每行路径定位一份实际产物；并非每份CSV都对应独立训练，某些是同一次训练的不同后处理或重复归档。

表中“留存epoch”只表示当前目录有这些文件，不单独担保它们是完整搜索后的最优；明确最优轮次的主实验已在正文用日志佐证。路径 `G:model/...` 对应项目内 `model/...`；`D:...` 对应 `D:/SemEval_experiments/...`。同目录若无特殊说明，预测名为 `dev_predictions_pair_aware.csv`。

### 14.1 官方 DEV（660行）

| # | 保存预测 / 实验目录 | F1 | SoftCons | HardCons | 留存epoch |
|---|---|---:|---:|---:|---|
| 1 | [dev_deberta_predictions_penalty_0.0.csv](<dev_deberta_predictions_penalty_0.0.csv>) | 0.836313 | 0.906137 | 0.862816 | 未找到 |
| 2 | [dev_predictions_penalty_0.0.csv](<dev_predictions_penalty_0.0.csv>) | 0.836313 | 0.906137 | 0.862816 | 未找到 |
| 3 | [dev_predictions_penalty_0.25.csv](<dev_predictions_penalty_0.25.csv>) | 0.725827 | 0.722022 | 0.664260 | 未找到 |
| 4 | [dev_predictions_penalty_0.5.csv](<dev_predictions_penalty_0.5.csv>) | 0.726296 | 0.725632 | 0.667870 | 未找到 |
| 5 | [dev_predictions_penalty_1.0.csv](<dev_predictions_penalty_1.0.csv>) | 0.735155 | 0.747292 | 0.685921 | 未找到 |
| 6 | [dev_predictions_penalty_1.5.csv](<dev_predictions_penalty_1.5.csv>) | 0.730743 | 0.765343 | 0.689531 | 未找到 |
| 7 | [dev_predictions_penalty_1.csv](<dev_predictions_penalty_1.csv>) | 0.735155 | 0.747292 | 0.685921 | 未找到 |
| 8 | [deberta_consistency_seed_42 / dev_predictions_penalty_0.0.csv](<model/deberta_consistency_seed_42/dev_predictions_penalty_0.0.csv>) | 0.838251 | 0.891697 | 0.841155 | 未找到 |
| 9 | [deberta_consistency_w0p1_seed_42 / dev_predictions_penalty_0.0.csv](<model/deberta_consistency_w0p1_seed_42/dev_predictions_penalty_0.0.csv>) | 0.838916 | 0.891697 | 0.848375 | 未找到 |
| 10 | [deberta_nli_base_hierarchical_manual_seed_42](<model/deberta_nli_base_hierarchical_manual_seed_42/dev_predictions_pair_aware.csv>) | 0.883999 | 1.000000 | 0.909747 | 7 |
| 11 | [deberta_nli_base_null_alignment_branchlr2e4_seed_42](<model/deberta_nli_base_null_alignment_branchlr2e4_seed_42/dev_predictions_pair_aware.csv>) | 0.902995 | 1.000000 | 0.916968 | 3 |
| 12 | [deberta_nli_base_null_alignment_M_seed_42](<model/deberta_nli_base_null_alignment_M_seed_42/dev_predictions_pair_aware.csv>) | 0.928987 | 1.000000 | 0.927798 | 3 |
| 13 | [deberta_nli_base_null_alignment_M_seed_7](<model/deberta_nli_base_null_alignment_M_seed_7/dev_predictions_pair_aware.csv>) | 0.910091 | 1.000000 | 0.898917 | 6 |
| 14 | [deberta_nli_base_null_alignment_seed_42](<model/deberta_nli_base_null_alignment_seed_42/dev_predictions_pair_aware.csv>) | 0.901876 | 1.000000 | 0.938628 | 10 |
| 15 | [deberta_nli_base_pair_smoothing_0p05_ee1p0_seed_42](<model/deberta_nli_base_pair_smoothing_0p05_ee1p0_seed_42/dev_predictions_pair_aware.csv>) | 0.914819 | 1.000000 | 0.913357 | 5 |
| 16 | [deberta_nli_base_pair_smoothing_0p05_seed_42](<model/deberta_nli_base_pair_smoothing_0p05_seed_42/dev_predictions_pair_aware.csv>) | 0.916043 | 1.000000 | 0.916968 | 8 |
| 17 | [deberta_nli_base_prototype_k1_seed_42](<model/deberta_nli_base_prototype_k1_seed_42/dev_predictions_pair_aware.csv>) | 0.908659 | 1.000000 | 0.935018 | 7 |
| 18 | [deberta_nli_base_prototype_k3_seed_42](<model/deberta_nli_base_prototype_k3_seed_42/dev_predictions_pair_aware.csv>) | 0.898678 | 1.000000 | 0.920578 | 6 |
| 19 | [deberta_nli_base_token_alignment_attn_pool_seed_42](<model/deberta_nli_base_token_alignment_attn_pool_seed_42/dev_predictions_pair_aware.csv>) | 0.919820 | 1.000000 | 0.916968 | 3 |
| 20 | [deberta_nli_base_token_alignment_compose_seed_42](<model/deberta_nli_base_token_alignment_compose_seed_42/dev_predictions_pair_aware.csv>) | 0.918678 | 1.000000 | 0.920578 | 3 |
| 21 | [deberta_nli_base_token_alignment_contrast_seed_42](<model/deberta_nli_base_token_alignment_contrast_seed_42/dev_predictions_pair_aware.csv>) | 0.925905 | 1.000000 | 0.920578 | 3 |
| 22 | [deberta_nli_base_token_alignment_GELU_seed_42](<model/deberta_nli_base_token_alignment_GELU_seed_42/dev_predictions_pair_aware.csv>) | 0.924559 | 1.000000 | 0.924188 | 3 |
| 23 | [deberta_nli_base_token_alignment_group_lr_seed_42](<model/deberta_nli_base_token_alignment_group_lr_seed_42/dev_predictions_pair_aware.csv>) | 0.919161 | 1.000000 | 0.913357 | 3 |
| 24 | [deberta_nli_base_token_alignment_manual_boundary_seed_42](<model/deberta_nli_base_token_alignment_manual_boundary_seed_42/dev_predictions_pair_aware.csv>) | 0.933544 | 1.000000 | 0.931408 | 3 |
| 25 | [deberta_nli_base_token_alignment_null_seed_42](<model/deberta_nli_base_token_alignment_null_seed_42/dev_predictions_pair_aware.csv>) | 0.924559 | 1.000000 | 0.924188 | 3 |
| 26 | [deberta_nli_base_token_alignment_official_full_aug24_seed_42](<model/deberta_nli_base_token_alignment_official_full_aug24_seed_42/dev_predictions_pair_aware.csv>) | 0.914775 | 1.000000 | 0.906137 | 3 |
| 27 | [deberta_nli_base_token_alignment_official_full_control_seed_42](<model/deberta_nli_base_token_alignment_official_full_control_seed_42/dev_predictions_pair_aware.csv>) | 0.924794 | 1.000000 | 0.924188 | 7 |
| 28 | [deberta_nli_base_token_alignment_seed_42](<model/deberta_nli_base_token_alignment_seed_42/dev_predictions_pair_aware.csv>) | 0.924794 | 1.000000 | 0.924188 | 7 |
| 29 | [deberta_nli_base_token_alignment_seed_7](<model/deberta_nli_base_token_alignment_seed_7/dev_predictions_pair_aware.csv>) | 0.912365 | 1.000000 | 0.902527 | 6 |
| 30 | [deberta_nli_base_uniform_alignment_seed_42](<model/deberta_nli_base_uniform_alignment_seed_42/dev_predictions_pair_aware.csv>) | 0.915088 | 1.000000 | 0.909747 | 3 |
| 31 | [deberta_nli_base_uniform_alignment_seed_7](<model/deberta_nli_base_uniform_alignment_seed_7/dev_predictions_pair_aware.csv>) | 0.891261 | 1.000000 | 0.920578 | 1 |
| 32 | [deberta_pair_ce_w0p5_seed_42](<model/deberta_pair_ce_w0p5_seed_42/dev_predictions_pair_aware.csv>) | 0.896569 | 1.000000 | 0.920578 | 未找到 |
| 33 | [deberta_real_pair_ce_direction_aux_w0p1_seed_42 / dev_predictions_ee_calibrated.csv](<model/deberta_real_pair_ce_direction_aux_w0p1_seed_42/dev_predictions_ee_calibrated.csv>) | 0.911788 | 1.000000 | 0.924188 | 6 |
| 34 | [deberta_real_pair_ce_direction_aux_w0p1_seed_42](<model/deberta_real_pair_ce_direction_aux_w0p1_seed_42/dev_predictions_pair_aware.csv>) | 0.908948 | 1.000000 | 0.920578 | 6 |
| 35 | [deberta_real_pair_ce_direction_interleaved_seed_42](<model/deberta_real_pair_ce_direction_interleaved_seed_42/dev_predictions_pair_aware.csv>) | 0.883829 | 1.000000 | 0.895307 | 10 |
| 36 | [deberta_real_pair_ce_w0p5_nweight1p5_eeweight1p25_seed_42](<model/deberta_real_pair_ce_w0p5_nweight1p5_eeweight1p25_seed_42/dev_predictions_pair_aware.csv>) | 0.912384 | 1.000000 | 0.916968 | 10 |
| 37 | [deberta_real_pair_ce_w0p5_nweight1p5_seed_42](<model/deberta_real_pair_ce_w0p5_nweight1p5_seed_42/dev_predictions_pair_aware.csv>) | 0.907754 | 1.000000 | 0.916968 | 1,2,3,4,5,6,7,8 |
| 38 | [deberta_real_pair_ce_w0p5_nweight1p75_seed_42](<model/deberta_real_pair_ce_w0p5_nweight1p75_seed_42/dev_predictions_pair_aware.csv>) | 0.896361 | 1.000000 | 0.916968 | 未找到 |
| 39 | [deberta_real_pair_ce_w0p5_seed_42](<model/deberta_real_pair_ce_w0p5_seed_42/dev_predictions_pair_aware.csv>) | 0.902996 | 1.000000 | 0.927798 | 1,2,3,4,5,6,7,8,9,10 |
| 40 | [deberta_real_pair_ce_weights_augmented_v1_seed_42](<model/deberta_real_pair_ce_weights_augmented_v1_seed_42/dev_predictions_pair_aware.csv>) | 0.896206 | 1.000000 | 0.902527 | 1 |
| 41 | [deberta_real_pair_ce_weights_nli_base_lr5e6_ep15_seed_42](<model/deberta_real_pair_ce_weights_nli_base_lr5e6_ep15_seed_42/dev_predictions_pair_aware.csv>) | 0.889658 | 1.000000 | 0.920578 | 13 |
| 42 | [deberta_real_pair_ce_weights_nli_base_n1p75_seed_42](<model/deberta_real_pair_ce_weights_nli_base_n1p75_seed_42/dev_predictions_pair_aware.csv>) | 0.890178 | 1.000000 | 0.913357 | 2 |
| 43 | [deberta_real_pair_ce_weights_nli_base_seed_42](<model/deberta_real_pair_ce_weights_nli_base_seed_42/dev_predictions_pair_aware.csv>) | 0.913283 | 1.000000 | 0.935018 | 5 |
| 44 | [deberta_real_pair_ce_weights_scheduler_seed_42](<model/deberta_real_pair_ce_weights_scheduler_seed_42/dev_predictions_pair_aware.csv>) | 0.883898 | 1.000000 | 0.913357 | 8 |
| 45 | [deberta_real_pair_ce_weights_type_aux_w0p3_seed_42](<model/deberta_real_pair_ce_weights_type_aux_w0p3_seed_42/dev_predictions_pair_aware.csv>) | 0.892869 | 1.000000 | 0.916968 | 8 |
| 46 | [deberta_seed_42](<model/deberta_seed_42/dev_predictions_pair_aware.csv>) | 0.897671 | 1.000000 | 0.913357 | 未找到 |
| 47 | [deberta_nli_base_null_lengthnorm_seed_42](<D:/SemEval_experiments/deberta_nli_base_null_lengthnorm_seed_42/dev_predictions_pair_aware.csv>) | 0.912274 | 1.000000 | 0.924188 | 3 |
| 48 | [deberta_nli_base_token_alignment_bientail_seed_42 / dev_predictions_bientail.csv](<D:/SemEval_experiments/deberta_nli_base_token_alignment_bientail_seed_42/dev_predictions_bientail.csv>) | 0.851986 | 0.920578 | 0.873646 | 3 |
| 49 | [deberta_nli_base_token_alignment_boundary0p3_seed_42](<D:/SemEval_experiments/deberta_nli_base_token_alignment_boundary0p3_seed_42/dev_predictions_pair_aware.csv>) | 0.914506 | 1.000000 | 0.909747 | 10 |
| 50 | [deberta_nli_base_token_alignment_entailaux0p2_seed_42](<D:/SemEval_experiments/deberta_nli_base_token_alignment_entailaux0p2_seed_42/dev_predictions_pair_aware.csv>) | 0.883745 | 1.000000 | 0.913357 | 10 |
| 51 | [deberta_nli_large_token_alignment_null_v2_seed_42](<D:/SemEval_experiments/deberta_nli_large_token_alignment_null_v2_seed_42/dev_predictions_pair_aware.csv>) | 0.908271 | 1.000000 | 0.909747 | 3 |
| 52 | [deberta_nli_large_token_alignment_seed_42](<D:/SemEval_experiments/deberta_nli_large_token_alignment_seed_42/dev_predictions_pair_aware.csv>) | 0.898715 | 1.000000 | 0.895307 | 3 |
| 53 | [token_alignment_overlap_ee_clean_seed_42](<D:/SemEval_experiments/token_alignment_overlap_ee_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.923182 | 1.000000 | 0.924188 | 2 |

### 14.2 原内部验证（609行）

| # | 保存预测 / 实验目录 | F1 | SoftCons | HardCons | 留存epoch |
|---|---|---:|---:|---:|---|
| 1 | [deberta_nli_base_null_alignment_M_internal_seed_42](<model/deberta_nli_base_null_alignment_M_internal_seed_42/dev_predictions_pair_aware.csv>) | 0.922583 | 1.000000 | 0.929961 | 2 |
| 2 | [deberta_nli_base_token_alignment_internal_seed_42](<model/deberta_nli_base_token_alignment_internal_seed_42/dev_predictions_pair_aware.csv>) | 0.922583 | 1.000000 | 0.929961 | 2 |
| 3 | [deberta_nli_base_token_alignment_rdrop_internal_seed_42](<model/deberta_nli_base_token_alignment_rdrop_internal_seed_42/dev_predictions_pair_aware.csv>) | 0.907927 | 1.000000 | 0.910506 | 2 |

### 14.3 clean内部验证（612行）

| # | 保存预测 / 实验目录 | F1 | SoftCons | HardCons | 留存epoch |
|---|---|---:|---:|---:|---|
| 1 | [deberta_nli_base_null_alignment_M_clean_run2_seed_42](<model/deberta_nli_base_null_alignment_M_clean_run2_seed_42/dev_predictions_pair_aware.csv>) | 0.902921 | 1.000000 | 0.912214 | 3 |
| 2 | [deberta_nli_base_null_alignment_M_clean_seed_42](<model/deberta_nli_base_null_alignment_M_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.939778 | 1.000000 | 0.942748 | 4 |
| 3 | [deberta_nli_base_pair_smoothing_clean_seed_42](<model/deberta_nli_base_pair_smoothing_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.942759 | 1.000000 | 0.946565 | 2 |
| 4 | [deberta_nli_base_token_alignment_clean_seed_42](<model/deberta_nli_base_token_alignment_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.939778 | 1.000000 | 0.942748 | 4 |
| 5 | [deberta_nli_base_token_alignment_hard_e_clean_seed_42](<model/deberta_nli_base_token_alignment_hard_e_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.929816 | 1.000000 | 0.942748 | 4 |
| 6 | [deberta_nli_base_token_alignment_manual_boundary_clean_18_seed_42](<model/deberta_nli_base_token_alignment_manual_boundary_clean_18_seed_42/dev_predictions_pair_aware.csv>) | 0.924406 | 1.000000 | 0.931298 | 10 |
| 7 | [deberta_nli_base_token_alignment_manual_boundary_clean_seed_42](<model/deberta_nli_base_token_alignment_manual_boundary_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.923818 | 1.000000 | 0.919847 | 3 |
| 8 | [deberta_nli_base_token_alignment_strictclean_aug24_seed_42](<model/deberta_nli_base_token_alignment_strictclean_aug24_seed_42/dev_predictions_pair_aware.csv>) | 0.938061 | 1.000000 | 0.938931 | 4 |
| 9 | [token_alignment_ee_margin_clean_seed_42](<D:/SemEval_experiments/token_alignment_ee_margin_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.918038 | 1.000000 | 0.923664 | 4 |
| 10 | [token_alignment_focal_clean_seed_42](<D:/SemEval_experiments/token_alignment_focal_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.910461 | 1.000000 | 0.938931 | 3 |
| 11 | [token_alignment_gate_clean_seed_42](<D:/SemEval_experiments/token_alignment_gate_clean_seed_42/dev_predictions_pair_aware.csv>) | 0.927201 | 1.000000 | 0.923664 | 3 |

### 14.4 已保存评分 JSON（S）

用于核对原始报告，不强行把同名或模糊命名结果合并。尤其`penalty_0`与`deberta_penalty_0`不是同一个结果。

| 结果文件 | Weighted F1 | SoftCons | HardCons |
|---|---:|---:|---:|
| [deberta_consistency_seed_42/scores.json](<results/deberta_consistency_seed_42/scores.json>) | 0.838251 | 0.891697 | 0.841155 |
| [deberta_consistency_w0p1_seed_42/scores.json](<results/deberta_consistency_w0p1_seed_42/scores.json>) | 0.838916 | 0.891697 | 0.848375 |
| [deberta_direction_aux_ee_penalty_0p5/scores.json](<results/deberta_direction_aux_ee_penalty_0p5/scores.json>) | 0.911788 | 1.000000 | 0.924188 |
| [deberta_large_type_aux_w0p3_scores/scores.json](<results/deberta_large_type_aux_w0p3_scores/scores.json>) | 0.899774 | 1.000000 | 0.931408 |
| [deberta_nli_base_pair_smoothing_0p05_seed_42/scores.json](<results/deberta_nli_base_pair_smoothing_0p05_seed_42/scores.json>) | 0.916043 | 1.000000 | 0.916968 |
| [deberta_nli_base_token_alignment_manual_boundary_seed_42/scores.json](<results/deberta_nli_base_token_alignment_manual_boundary_seed_42/scores.json>) | 0.933544 | 1.000000 | 0.931408 |
| [deberta_nli_base_token_alignment_seed_42/scores.json](<results/deberta_nli_base_token_alignment_seed_42/scores.json>) | 0.924794 | 1.000000 | 0.924188 |
| [deberta_pair_ce_w0p5_seed_42/scores.json](<results/deberta_pair_ce_w0p5_seed_42/scores.json>) | 0.896569 | 1.000000 | 0.920578 |
| [deberta_penalty_0/scores.json](<results/deberta_penalty_0/scores.json>) | 0.836313 | 0.906137 | 0.862816 |
| [deberta_real_pair_ce_w0p5_nweight1p5_eeweight1p25_seed_42/scores.json](<results/deberta_real_pair_ce_w0p5_nweight1p5_eeweight1p25_seed_42/scores.json>) | 0.912384 | 1.000000 | 0.916968 |
| [deberta_real_pair_ce_w0p5_nweight1p5_seed_42/scores.json](<results/deberta_real_pair_ce_w0p5_nweight1p5_seed_42/scores.json>) | 0.907754 | 1.000000 | 0.916968 |
| [deberta_real_pair_ce_w0p5_seed_42/scores.json](<results/deberta_real_pair_ce_w0p5_seed_42/scores.json>) | 0.902996 | 1.000000 | 0.927798 |
| [deberta_seed_42/scores.json](<results/deberta_seed_42/scores.json>) | 0.843772 | 0.877256 | 0.844765 |
| [deberta_seed_42_pair_aware/scores.json](<results/deberta_seed_42_pair_aware/scores.json>) | 0.897671 | 1.000000 | 0.913357 |
| [deberta_seed_42_pair_aware_equivalence_penalty/scores.json](<results/deberta_seed_42_pair_aware_equivalence_penalty/scores.json>) | 0.897671 | 1.000000 | 0.913357 |
| [null_alignment_M_seed_42/scores.json](<results/null_alignment_M_seed_42/scores.json>) | 0.928987 | 1.000000 | 0.927798 |
| [penalty_0/scores.json](<results/penalty_0/scores.json>) | 0.726928 | 0.718412 | 0.664260 |
| [penalty_1/scores.json](<results/penalty_1/scores.json>) | 0.735155 | 0.747292 | 0.685921 |

### 14.5 没有可匹配预测的留存目录

这些目录不自动代表已完成有效实验；可能是只创建了目录、产物已移走，或仅有诊断文件。

- `D:/SemEval_experiments/deberta_nli_large_token_alignment_null_seed_42`：空目录。
- `model/deberta_diagnostic_seed_42`：空目录。
- `model/deberta_nli_base_stage_probe_seed_42`：epoch_1.pth, epoch_2.pth, epoch_3.pth, stage_probe.csv。
- `model/deberta_nli_base_token_alignment_official_dev_aug24_seed_42`：epoch_1.pth, epoch_2.pth, epoch_3.pth。
- `model/deberta_nli_base_token_alignment_official_full_FALSE_control_seed_42`：空目录。

<a id="s15"></a>
## 15. 当前脚本配置索引

此处逐一记录根目录相关脚本当前的有效代码配置，包括模型、数据入口、RUN_DIR、训练参数及主入口。**它是2026-09-29源码快照的文本摘要，不是给所有历史结果补配的超参数。** 注释中的旧参数不当作当前值。

只有源码而无结果的分支在前文标注缺失；只有结果而源码已缺失的分支在结果表仍予保留。下载脚本只管理预训练模型；`predict_unlabeled.py`用于推理；`make_pair_split.py`用于内部划分；它们本身不是新的模型效果实验。

### 15.1 bert_baseline.py

[查看脚本](<bert_baseline.py>)。以下为当前有效代码摘录。

```python
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_path = 'D:/BaiduNetdiskDownload/PretrainedModel/models/google-bert--bert-base-uncased/snapshots/master'
if __name__ == '__main__':
    checkAndprocess_data()
    for penalty in [0.0, 0.5, 1, 1.5]:
        (best_epoch, best_score, best_preds) = evaluate_model(negative_penalty=penalty)
        print(f'negative_penalty={penalty}，Weighted F1={best_score:.4f}')
        submission_output(best_preds, penalty)
# def create_dataloaders(batch_size=16):
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# train_model（源码第149行）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.2 deberta_baseline.py

[查看脚本](<deberta_baseline.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    (dev_logits, dev_labels) = collect_dev_logits()
# def create_dataloaders(batch_size=16):
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# train_model（源码第172行）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
```

### 15.3 deberta_large.py

[查看脚本](<deberta_large.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_v3_large_nli_full_opt_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第391行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW([{'params': deberta_model.model.deberta.parameters(), 'lr': 3e-06}, {'params': deberta_model.model.pooler.parameters(), 'lr': 1e-05}, {'params': deberta_model.model.classifier.parameters(), 'lr': 2e-05}], weight_decay=0.01)
epochs = 10
pair_weight = 0.5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.4 deberta_large_baseline.py

[查看脚本](<deberta_large_baseline.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_v3_large_nli_pair_ce_w0p5_baseline_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
    analyze_pair_vs_negative_errors(best_preds)
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第391行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=5e-06, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.5 deberta_large_interleaved.py

[查看脚本](<deberta_large_interleaved.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_v3_large_nli_pair_ce_w0p5_interleaved_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第387行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=5e-06, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.6 deberta_large_pair_head.py

[查看脚本](<deberta_large_pair_head.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_v3_large_pair_head_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
    analyze_pair_errors(best_preds)
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第472行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=5e-06, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.7 deberta_large_type_aux.py

[查看脚本](<deberta_large_type_aux.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_large_type_aux_w0p3_nweight1p2_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
    analyze_pair_vs_negative_errors(best_preds)
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第498行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=5e-06, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.8 deberta_nli_base_direction_aux.py

[查看脚本](<deberta_nli_base_direction_aux.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_direction_aux_w0p2_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    print(f'本次模型：{model_name}')
    print(f'运行设备：{device}')
    print(f'保存目录：{RUN_DIR.resolve()}')
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第411行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
direction_weight = 0.2
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.9 deberta_nli_base_entailment_classifier.py

[查看脚本](<deberta_nli_base_entailment_classifier.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('D:\\SemEval_experiments') / f'deberta_nli_base_token_alignment_entailaux0p2_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_entailment_aux()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloader(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_entailment_aux（源码第1147行）
row_dataloader = create_my_dataloader(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
entailment_weight = 0.2
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
```

### 15.10 deberta_nli_base_hierarchical.py

[查看脚本](<deberta_nli_base_hierarchical.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_hierarchical_manual_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    print(f'本次模型：{model_name}')
    print(f'运行设备：{device}')
    print(f'保存目录：{RUN_DIR.resolve()}')
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第544行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.11 deberta_nli_base_multi_prototype.py

[查看脚本](<deberta_nli_base_multi_prototype.py>)。以下为当前有效代码摘录。

```python
SEED = 42
NUM_PROTOTYPES = 1
PROTOTYPE_DIM = 128
PROTOTYPE_TEMPERATURE = 0.1
PROTOTYPE_LOGIT_SCALE = 20.0
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main') / 'model' / f'deberta_nli_base_prototype_k{NUM_PROTOTYPES}_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    print(f'本次模型：{model_name}')
    print(f'每类中心数量：{NUM_PROTOTYPES}')
    print(f'运行设备：{device}')
    print(f'保存目录：{RUN_DIR.resolve()}')
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'\n最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
    print('对照版 Weighted F1：0.913283')
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第507行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.12 deberta_nli_base_null_alignment.py

[查看脚本](<deberta_nli_base_null_alignment.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main') / 'model' / f'deberta_nli_base_null_alignment_branchlr2e4_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    print(f'本次模型：{model_name}')
    print(f'运行设备：{device}')
    print(f'保存目录：{RUN_DIR.resolve()}')
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'\n最佳轮次：{best_epoch}')
    print(f'最佳 Weighted F1：{best_score:.4f}')
    print('原版 Weighted F1：0.913283')
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第678行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW([{'params': list(deberta_model.deberta_model.parameters()) + list(deberta_model.output.parameters()), 'lr': 2e-05}, {'params': list(deberta_model.alignment.parameters()) + list(deberta_model.alignment_output.parameters()), 'lr': 0.0002}], weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.13 deberta_nli_base_null_alignment_M.py

[查看脚本](<deberta_nli_base_null_alignment_M.py>)。以下为当前有效代码摘录。

```python
SEED = 7
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_null_alignment_M_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1048行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.14 deberta_nli_base_null_alignment_M_clean.py

[查看脚本](<deberta_nli_base_null_alignment_M_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_null_alignment_M_clean_run2_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1039行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.15 deberta_nli_base_null_alignment_M_internal.py

[查看脚本](<deberta_nli_base_null_alignment_M_internal.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_null_alignment_M_internal_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split/train.csv')
dev_data = pd.read_csv('./data/internal_split/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1043行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.16 deberta_nli_base_pair_smoothing.py

[查看脚本](<deberta_nli_base_pair_smoothing.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_pair_smoothing_0p05_ee1p0_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第414行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.17 deberta_nli_base_pair_smoothing_clean.py

[查看脚本](<deberta_nli_base_pair_smoothing_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_pair_smoothing_clean_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第408行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.18 deberta_nli_base_stage_probe.py

[查看脚本](<deberta_nli_base_stage_probe.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_nli_base_stage_probe_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    summary = pd.read_csv(RUN_DIR / 'stage_probe.csv')
    print('\n===== 六条阶段记录 =====')
    print(summary.to_string(index=False))
    print('记录文件：', (RUN_DIR / 'stage_probe.csv').resolve())
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第482行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 3
pair_weight = 0.5
dev_dataloader = create_dev_dataloader(batch_size=16)
```

### 15.19 deberta_nli_base_token_alignment.py

[查看脚本](<deberta_nli_base_token_alignment.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1143行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.20 deberta_nli_base_token_alignment_boundary.py

[查看脚本](<deberta_nli_base_token_alignment_boundary.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('D:\\SemEval_experiments') / f'deberta_nli_base_token_alignment_boundary0p3_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1146行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
boundary_weight = 0.1
```

### 15.21 deberta_nli_base_token_alignment_clean.py

[查看脚本](<deberta_nli_base_token_alignment_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_clean_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1141行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.22 deberta_nli_base_token_alignment_hard_e_clean.py

[查看脚本](<deberta_nli_base_token_alignment_hard_e_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_hard_e_clean_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    analyze_equivalence_errors()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1160行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.23 deberta_nli_base_token_alignment_internal.py

[查看脚本](<deberta_nli_base_token_alignment_internal.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_internal_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split/train.csv')
dev_data = pd.read_csv('./data/internal_split/valid.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1142行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.24 deberta_nli_base_token_alignment_manual_boundary.py

[查看脚本](<deberta_nli_base_token_alignment_manual_boundary.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_manual_boundary_18_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
manual_file = Path('D:\\SemEval_experiments\\manual_boundary_triplets.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    train_data = add_manual_boundary_data(train_data, dev_data, manual_file)
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1332行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.25 deberta_nli_base_token_alignment_manual_boundary_clean.py

[查看脚本](<deberta_nli_base_token_alignment_manual_boundary_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_manual_boundary_clean_18_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
manual_file = Path('D:\\SemEval_experiments\\manual_boundary_triplets_internal_clean.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    train_data = add_manual_boundary_data(train_data, dev_data, manual_file)
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1337行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.26 deberta_nli_base_token_alignment_null_v2.py

[查看脚本](<deberta_nli_base_token_alignment_null_v2.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'D:\\SemEval_experiments\\deberta_nli_base_null_lengthnorm_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1147行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.27 deberta_nli_base_token_alignment_trainonly_aug_clean.py

[查看脚本](<deberta_nli_base_token_alignment_trainonly_aug_clean.py>)。以下为当前有效代码摘录。

```python
SEED = 42
USE_AUGMENTATION = True
run_name = 'aug24' if USE_AUGMENTATION else 'control'
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_strictclean_{run_name}_seed_{SEED}'
train_data = pd.read_csv('./data/internal_split_clean/train.csv')
dev_data = pd.read_csv('./data/internal_split_clean/valid.csv')
train_data = train_data.loc[~train_data['pair_id'].isin(overlap_pair_ids)].copy()
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1183行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.28 deberta_nli_base_token_alignment_trainonly_aug_dev.py

[查看脚本](<deberta_nli_base_token_alignment_trainonly_aug_dev.py>)。以下为当前有效代码摘录。

```python
SEED = 42
USE_AUGMENTATION = False
run_name = 'aug24' if USE_AUGMENTATION else 'control'
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_token_alignment_official_full_{run_name}_seed_{SEED}'
(train_data, dev_data) = load_official_data(USE_AUGMENTATION)
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1177行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.29 deberta_nli_base_uniform_alignment.py

[查看脚本](<deberta_nli_base_uniform_alignment.py>)。以下为当前有效代码摘录。

```python
SEED = 7
RUN_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main\\model') / f'deberta_nli_base_uniform_alignment_seed_{SEED}'
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.ReLU())
# train_model_real_pair_ce（源码第1021行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.30 deberta_nli_large_token_alignment.py

[查看脚本](<deberta_nli_large_token_alignment.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'D:\\SemEval_experiments\\deberta_nli_large_token_alignment_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1145行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW([{'params': deberta_model.deberta_model.parameters(), 'lr': 5e-06}, {'params': deberta_model.output.parameters(), 'lr': 2e-05}, {'params': list(deberta_model.align_projection.parameters()) + list(deberta_model.compare_layer.parameters()) + list(deberta_model.match_output.parameters()), 'lr': 2e-05}], weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.31 deberta_nli_large_token_alignment_null.py

[查看脚本](<deberta_nli_large_token_alignment_null.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'D:\\SemEval_experiments\\deberta_nli_large_token_alignment_null_v2_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
self.compare_layer = nn.Sequential(nn.Linear(4 * 128, 128), nn.GELU())
# train_model_real_pair_ce（源码第1105行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW([{'params': deberta_model.deberta_model.parameters(), 'lr': 5e-06}, {'params': deberta_model.output.parameters(), 'lr': 2e-05}, {'params': list(deberta_model.align_projection.parameters()) + list(deberta_model.compare_layer.parameters()) + list(deberta_model.match_output.parameters()) + [deberta_model.null_similarity], 'lr': 2e-05}], weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.32 deberta_pair_ce.py

[查看脚本](<deberta_pair_ce.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_pair_ce_w0p7_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dataloaders(batch_size=16):
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_consistency_train_loader(batch_size=16):
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# train_model_pair_ce（源码第236行）
train_dataloader = create_consistency_train_loader(batch_size=16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.7
```

### 15.33 deberta_pair_consistency.py

[查看脚本](<deberta_pair_consistency.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    analyze_pair_aware_errors(best_preds)
    submission_output(best_preds)
# def create_dataloaders(batch_size=16):
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_consistency_train_loader(batch_size=16):
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# train_model（源码第237行）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
# train_model_consistency（源码第291行）
train_dataloader = create_consistency_train_loader(batch_size=16)
epochs = 1
consistency_weight = 1.0
```

### 15.34 deberta_real_pair_ce.py

[查看脚本](<deberta_real_pair_ce.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_real_pair_ce_w0p5_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第374行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.35 deberta_real_pair_ce_direction_aux.py

[查看脚本](<deberta_real_pair_ce_direction_aux.py>)。以下为当前有效代码摘录。

```python
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
```

### 15.36 deberta_real_pair_ce_weights.py

[查看脚本](<deberta_real_pair_ce_weights.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_real_pair_ce_w0p5_nweight1p5_eeweight1p25_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    compare_predictions()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第408行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.37 deberta_real_pair_ce_weights_nli_base.py

[查看脚本](<deberta_real_pair_ce_weights_nli_base.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_real_pair_ce_weights_nli_base_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
if __name__ == '__main__':
    print(f'本次模型：{model_name}')
    print(f'运行设备：{device}')
    print(f'保存目录：{RUN_DIR.resolve()}')
    checkAndprocess_data()
    analyze_best_model()
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第410行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.38 deberta_real_pair_ce_weights_scheduler.py

[查看脚本](<deberta_real_pair_ce_weights_scheduler.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_real_pair_ce_weights_scheduler_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第419行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
```

### 15.39 deberta_real_pair_ce_weights_type_aux.py

[查看脚本](<deberta_real_pair_ce_weights_type_aux.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/deberta_real_pair_ce_weights_type_aux_w0p3_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_name = 'microsoft/deberta-v3-base'
if __name__ == '__main__':
    checkAndprocess_data()
    train_model_real_pair_ce()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dev_dataloader(batch_size=16):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=16):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)
# train_model_real_pair_ce（源码第462行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=16)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.5], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.25], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(deberta_model.parameters(), lr=2e-05, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

### 15.40 download.py

[查看脚本](<download.py>)。以下为当前有效代码摘录。

```python
model_name = 'microsoft/deberta-v3-base'
```

### 15.41 download_DeBERTa-v3-base-mnli-fever-anli.py

[查看脚本](<download_DeBERTa-v3-base-mnli-fever-anli.py>)。以下为当前有效代码摘录。

```python
model_name = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
```

### 15.42 download_DeBERTa-v3-large-mnli.py

[查看脚本](<download_DeBERTa-v3-large-mnli.py>)。以下为当前有效代码摘录。

```python
model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
```

### 15.43 download_ModernBERT.py

[查看脚本](<download_ModernBERT.py>)。以下为当前有效代码摘录。

```python
model_name = 'tasksource/ModernBERT-large-nli'
```

### 15.44 make_pair_split.py

[查看脚本](<make_pair_split.py>)。以下为当前有效代码摘录。

```python
PROJECT_DIR = Path('G:\\pycode\\SemEval\\Task-2-DiCo-NLI-main')
if __name__ == '__main__':
    create_clean_pair_split()
```

### 15.45 make_train_only_boundary_augmentation.py

[查看脚本](<make_train_only_boundary_augmentation.py>)。以下为当前有效代码摘录。

```python
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / 'data/internal_split_clean/train.csv'
VALID_PATH = PROJECT_DIR / 'data/internal_split_clean/valid.csv'
OFFICIAL_DEV_PATH = PROJECT_DIR / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
OUTPUT_PATH = PROJECT_DIR / 'data/augmentation/train_only_boundary_24pairs.csv'
if __name__ == '__main__':
    build_dataset()
```

### 15.46 mine_boundary_pairs.py

[查看脚本](<mine_boundary_pairs.py>)。以下为当前有效代码摘录。

辅助工具，函数入口：。

### 15.47 modernbert_large_baseline.py

[查看脚本](<modernbert_large_baseline.py>)。以下为当前有效代码摘录。

```python
SEED = 42
RUN_DIR = Path(f'./model/modernbert_large_nli_pair_ce_w0p5_seed_{SEED}')
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
model_path = 'D:\\BaiduNetdiskDownload\\PretrainedModel\\models\\tasksource--ModernBERT-large-nli'
if __name__ == '__main__':
    checkAndprocess_data()
    (best_epoch, best_score, best_preds) = evaluate_pair_aware_model()
    submission_output(best_preds)
    print(f'最佳轮次：{best_epoch}')
    print(f'最佳 Pair-aware F1：{best_score:.4f}')
# def create_dev_dataloader(batch_size=8):
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
# def create_my_dataloaders(batch_size=8):
row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)
pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)
# train_model_real_pair_ce（源码第408行）
(row_dataloader, pair_dataloader) = create_my_dataloaders(batch_size=8)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
pair_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
pair_criterion = nn.CrossEntropyLoss(weight=pair_weights)
optimizer = torch.optim.AdamW(modernbert_model.parameters(), lr=5e-06, weight_decay=0.01)
epochs = 10
pair_weight = 0.5
```

该脚本当前RUN_DIR对应的完整指标未在本次产物扫描中找到。Weighted F1 / SoftCons / HardCons及历史最佳epoch：当前项目记录中未找到准确数值，需要人工补充。

### 15.48 predict_unlabeled.py

[查看脚本](<predict_unlabeled.py>)。以下为当前有效代码摘录。

```python
PROJECT_DIR = Path(__file__).resolve().parent
if __name__ == '__main__':
    data = load_unlabeled_data()
    model = load_model()
    logits = predict_logits(model, data)
    save_predictions(data, logits)
```

### 15.49 probe_bidirectional_nli.py

[查看脚本](<probe_bidirectional_nli.py>)。以下为当前有效代码摘录。

```python
if __name__ == '__main__':
    probe_bidirectional_nli()
```


<a id="s16"></a>
## 16. 证据冲突、缺项与归档指纹

### 16.1 已澄清的冲突

| 常见误记 | 此次核对后的处理 |
|---|---|
| smoothing 0p05就是.924559 | 该保存目录P/S均.916043；.924559出现在其他alignment变体 |
| 原token脚本直接代表.9248的所有配置 | 当前原名脚本GELU；最近成功复现控制脚本ReLU |
| Boundary0p3肯定权重0.3 | 当前代码0.1；历史运行权重不能从目录确认 |
| Entailment Aux只增加一个loss | 当前函数还取消了Pair训练；不是严格单变量 |
| clean一定比official更低/更可靠 | 数据集和规模不同；规则有限，且反复调参也有偏差 |
| 所有NULL都没有超过基线 | NULL-M单次.928987确实超过；不据此宣称稳定胜出 |
| manual .9335是最终最好模型 | 有历史DEV介入来源问题；不纳入最终选择 |
| N precision1说明模型绝不误报N | 当前硬解码排除了paired N；指标受结构影响 |
| 只剩epoch7证明训练只跑了7轮 | 最终日志跑10轮，评估代码删除非最佳权重 |
| 两bit无法表达四类 | 可以表达；独立概率乘积才对联合分布增加限制 |

### 16.2 未能完整恢复的内容

BERT可靠归属的最优值、ModernBERT具体成绩、部分Large复杂头/训练调度的完整结果、N权重1.4的独立实验、纯Bi-Entailment训练源码、Boundary两种聚合与权重的逐次配置、所有被覆盖文件的历史快照、最终raw预测、所有seed和训练环境，以及完整逐epoch日志：**当前项目记录中未找到准确数值或完整材料，需要人工补充。**

README只作为当前本地任务/评估说明使用，没有在线核实比赛规则或时间。目录扫描未发现项目根README，`starter_kit/README.md`提到的根任务文档在这份目录中缺失。`evaluation_functions/README.md`及实际metrics.py是本记录的一致性定义来源。

### 16.3 文件指纹

以下SHA-256来自本次读取的小型源码/CSV，方便以后发现覆盖变化。未对数百MB checkpoint做完整哈希或加载校验；留存文件存在并不等于本次已验证权重可成功推理。

| 文件 | SHA-256 |
|---|---|
| [deberta_nli_base_token_alignment.py](<deberta_nli_base_token_alignment.py>) | `477f5b23c2ccecede9c26552f4eb5bb0055d3f7a263a6fa6fbb398635e764d47` |
| [deberta_nli_base_token_alignment_trainonly_aug_dev.py](<deberta_nli_base_token_alignment_trainonly_aug_dev.py>) | `7bc3fc3ba13efa6a796153b4d38e17619b66932d2d935a139e9b14c9cd6f0d2e` |
| [make_pair_split.py](<make_pair_split.py>) | `21dc5fc53c8abfd2e03bed69e900037ea9c42792e538cdcd83235552068b9449` |
| [make_train_only_boundary_augmentation.py](<make_train_only_boundary_augmentation.py>) | `90b761d997adaaf431779df534c6c3fd804a4806f6ee5feee1fea0c1ee042adb` |
| [dico_nli_dev_track1_participant_labeled.csv](<data/dev/dico_nli_dev_track1_participant_labeled.csv>) | `80b7152f7ccf0d349d42ad2f098988106689b918b36337e5756b121d17513511` |
| [dico_nli_train_track1_participant_labeled.csv](<data/train/dico_nli_train_track1_participant_labeled.csv>) | `5e8dd9f3a7e000933fec3c9e64d064f9a25586f141f58d6e6afb47c4fc797211` |
| [最终控制运行预测](<model/deberta_nli_base_token_alignment_official_full_control_seed_42/dev_predictions_pair_aware.csv>) | `f583960bd3c95eaef490b50e0000f0d2b10f5e065cd62bf34b45a449c9ed7405` |

---

**最终结论再次锁定：DeBERTa-v3-base NLI + Token Alignment + Real Pair CE + Pair-aware inference；官方完整train，无增强；选定运行官方DEV Weighted F1=0.9247935854，epoch7。其他分支均为探索记录。**
