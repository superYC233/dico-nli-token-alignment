# 原Alignment + Lexical：只替换Pair CE为反向KL

直接右键运行本目录`train_deberta_reversal.py`：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_mixed_pair/train_deberta_reversal.py
```

本次使用`config_reversal.py`，不会改原`config.py`或其他训练入口。后续同一方案只改新配置的RUN_NAME。

## 保留的部分

- **原DeBERTa + Token Alignment + Lexical模型**，直接复用`deberta_lexical_stage_probe/model.py`，没有pooler或新分类头。
- 从相同NLI初始化重新训练，不加载0.938、pooler版或Child-D的任务权重继续训练。
- 官方Train3042、Dev660，不加scope/relation/外部数据。
- 原run01混合训练：每轮1760组、8组最多16行、220步；原行各一次，1282真实反向组各一次。
- 主干与原分类/Alignment头学习率2e-5，Lexical学习率1e-3，AdamW weight_decay0.01，10epochs，float32，固定长度128。
- Row CE权重[1,1,1,1.5]，Row label smoothing仍是0。
- 每轮末评估，不加入warmup、线性衰减、早停或新增梯度裁剪。

这里的“原版”指原Lexical混合版run01，不是无Lexical、带3290行增强的历史0.938两阶段版本。

## 唯一的训练目标变化

```python
# 旧
loss = row_ce + 0.5 * pair_ce

# 新
loss = row_ce + 0.5 * reversal_kl
```

原Pair CE及其EE权重1.25、三组合label smoothing0.05随该损失一起移除。新的KL不使用金标组合权重，不把旧Pair CE暗中加回去。

KL直接复用刚完成验证的`deberta_reversal_kl/train_deberta_reversal.py`里的compute_loss公式；只复用数学函数，没有使用原生分类器的模型、配置或训练循环。

```text
p = softmax(原行F/B/E/N logits)
q = softmax(真实反向行logits，交换F和B)
KL = 0.5 × [KL(p||q) + KL(q||p)]
```

所有四类概率都参加KL，两边都反传、不detach。只在已有真实F/B或E/E反向记录上计算；N单例仅接受自己的Row CE，不人为构造NN。

每批仍只跑一次编码器，先得到该批所有原行logits，再计算Row CE和KL，一次反向更新；没有单独Pair阶段，也没有第二次dropout前向，因此不是R-Drop。

KL不等于正确性：两边一起错答N也可能KL为0，完整Row CE会惩罚这个错误。系数同为0.5不代表与原Pair CE具有相同梯度强度，日志会同时保留KL原值与加权贡献。

## 选模与结果

本次按**independent W-F1**保存最佳，并报告同一个checkpoint的pair-aware。历史run01按pair-aware保存权重，选模设置不同，但不会改变固定10轮的训练更新；比较独立能力时，应比较逐轮预测中的独立最佳，而不是拿另一个epoch的保存权重冒充该成绩。

只在独立F1提高时覆盖一份完整best_model.pth，逐轮预测和指标保留。旧pair-aware解码不变，配对行只选FB/BF/EE；单例仍由模型预测，没有单例强制N规则。

- 权重：`D:/model/SEAMEVAL/deberta_lexical_mixed_pair/run07_reversal_kl_seed42/best_model.pth`。
- 结果：本目录`outputs/run07_reversal_kl_seed42/`。
- `best_summary.csv`、`best_metrics.json`：同一权重的独立与结构成绩。
- `epoch_summary.csv`：逐轮F1、四类正确数、Row/KL损失及训练覆盖数。
- 训练后运行`evaluate_reversal.py`，完整重载并核对最佳轮次逐条分数。

`test_reversal.py`只进行CPU检查，不执行optimizer.step。训练由用户运行。

借鉴来源：[MinLee0210/dico-nli-challenge的CE + reversal-KL说明](https://github.com/MinLee0210/dico-nli-challenge#configuration)。本次是保留我们架构的目标替换试验，不是作者多语言系统/结构先验集成的复现，也不预先承诺提升。
