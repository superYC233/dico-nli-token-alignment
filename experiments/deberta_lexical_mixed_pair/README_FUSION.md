# 已有模型的预测融合：不训练，先验证互补性

历史0.938/0.9355/0.9306/0.9248高分版本的融合改用`analyze_high_fusion.py`，见`README_HIGH_FUSION.md`。双模型等权概率平均已得到pair-aware 0.939879；以下保留首轮独立预测成员的实验记录。

运行本目录`analyze_fusion.py`，读取已经保存的Dev四类分数，计算固定等权方案、分层方案以及分组bootstrap。

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_mixed_pair/analyze_fusion.py
```

全部路径及方案在`config_fusion.py`。输出在`outputs/fusion01_existing_predictions/`。不加载模型、不更新参数；这是Dev缓存预测上的离线融合，不是对新输入重新执行多模型推理。对应checkpoint的存在性已记录，内容尚未重新加载核验。

## 借鉴哪些做法

[作者模型页](https://huggingface.co/LakoreAI/dico-nli-checkpoints)描述了普通分数集成，以及用四分类模型的N概率作gate、用关系专家提供F/B/E条件分布的分层组合。

本次只借鉴这两种思路，不使用作者“单例直接判N”的结构先验，也不声称复现其DEV选择集成的0.961。

### 普通概率平均

```text
每个模型的原四类logits → softmax → 各模型四类概率等权平均
```

取平均分布的最大值作为独立预测，再可选做正反联合解码。必须按instance_id对齐，不能直接按CSV原行位置相加。

### 平均log-probability

```text
每个模型的log_softmax分数 → 等权平均 → 重新归一化
```

这是几何池化。在相同权重、温度1的条件下，它的argmax与原logits平均相同，不能声称log_softmax消除了模型分数尺度差异。概率平均与几何池化的行为不同，因此都报告；没有扫描温度或连续权重。

### 分层组合

```text
g = 四分类gate模型的P(N)平均
q = 关系模型在F/B/E内部的条件概率平均

P(N) = g
P(F/B/E) = (1-g) * q
```

gate只使用独立预测正常的模型；旧两阶段Lexical可作为方向候选，不把其过高N概率直接当gate。

所有样本采用相同公式，没有用真实类别逐行选模型，也不根据单例/配对身份设定N。改变q的尖锐程度也可能改变四类argmax，因此N正确数上升不能单独解释为gate排序变好。

## 实际成员

- `lexical_kl`：本轮原Alignment+Lexical、CE+KL，保存ep2，独立0.860056。
- `native_kl`：原生cross-encoder CE+KL，保存step935，独立0.854245。
- `pooler`：旧混合+pooler，保存ep10。
- `oof`：旧OOF难例加权，保存ep7。
- `staged_lexical`：旧两阶段Lexical，保存ep3；作为方向专家候选。
- `qwen_r16`：原Qwen3-0.6B C+r16最佳Adapter。

前五个使用官方3042行；Qwen沿用历史Train C（3042+475条反向N）。融合没有新增训练数据，但论文仍需披露这个成员的数据来源。没有使用原mixed ep2等仅保留预测、没有相应权重的轮次，也没有引入存在历史增强来源问题的0.938权重。

## 首轮离线结果

| 方案 | independent | pair-aware | N正确 | pair-aware F/B/E正确 |
| --- | ---: | ---: | ---: | ---: |
| 最新单模型lexical_kl | 0.860056 | 0.886018 | 67 | 520 |
| 三模型概率平均prob_diverse3 | 0.866410 | 0.914033 | 80 | 524 |
| 同三模型几何池化log_diverse3 | 0.855009 | 0.926162 | 82 | 530 |
| 四模型分层hier_direction2 | 0.866550 | 0.903408 | 77 | 520 |

三模型是lexical_kl、native_kl、qwen_r16。四模型分层的gate为lexical_kl/native_kl/pooler，direction为lexical_kl/staged_lexical。

这些数每行均来自同一个融合方案，没有拼接不同方案的独立/结构最好值。F/B/E到530是集成结果，不是单模型能力突破；它反驳的是“此前观察范围就是不可超过的硬天花板”。0.926162也没有超过历史0.938的最高结构分。

独立正确数：单模型569，三模型概率平均570，四模型分层571。独立F1的小幅提升尚未表现为大量净修复。

## 验证边界

对383个source pair_id做5000次成组bootstrap，配对行不会分开抽样。对照是最新lexical_kl单模型，而非原0.938。

三模型概率平均的独立增益约+0.64个百分点，95%区间约[-1.45,+2.73]，包含0；其结构增益约+2.80个百分点，区间约[+0.97,+4.79]。三模型几何池化结构增益约+4.01个百分点，区间约[+1.96,+6.34]。

区间是在现有已训练/已选定模型上，条件于当前Dev的抽样诊断，不覆盖训练种子变化，也没有消除比较多个组合、选择checkpoint造成的乐观偏差。当前组合是Dev探索性选择，不能写成独立验证的显著提升。

同时输出`independent`、允许NN竞争的`twin`和旧三组合`pair_aware`。后两项使用输入分组做联合解码；全部模式的单例都由模型概率决定，不强制N。选定方案的F1、SoftCons和HardCons已与官方评分器逐项核对。

## 产物

- `summary.csv`：所有单模型和固定方案的三种读出。
- `paired_bootstrap.csv`：固定比较的源组bootstrap。
- `fixed_broken.csv`：相对最新单模型修好和破坏数量。
- `error_overlap.csv`：成员错误重合和互补数量。
- `*_predictions.csv`：融合后的四类分数、独立/twin/pair-aware预测。
- `manifest.json`：成员、输入哈希、权重存在性、配方与研究口径。
- `official_checks/`：选定方案的官方评分结果。

建议先保留可实际运行的三模型概率平均和几何池化作为两个固定候选，不继续在Dev上进行大规模贪心选成员或逐行人工路由。要部署到新输入，需要依次运行各成员并使用相同Prompt、tokenizer、dtype和标签顺序，再应用这里的融合公式；不能把已保存Dev CSV当作新输入预测。
