# 原run02词汇版＋Mixout

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

先右键运行 [train_deberta_mixout.py](train_deberta_mixout.py)，默认只训练一份p=0.1的单模型。使用已有本地NLI模型缓存，无需下载新模型或安装Mixout库。

正式训练交给使用者。本目录附带的CPU检查只使用随机初始化的小模型，不代表真实成绩。

## 这次改什么

基线是无文本增强的DeBERTa＋CLS＋Token Alignment＋Lexical，历史run02第3轮：官方Track1英文Dev 660行，pair-aware Weighted F1为0.930569，同一权重的independent为0.468191。

数据只用官方Train 3042行，正常读取每行一次；没有加入scope、relation、复制行或反向N。网络、原有dropout、两个参数组的学习率、类别权重、损失与Row→Pair顺序恢复历史run02设置。没有LLRD。

唯一新增的训练方法是：将主干已有nn.Linear替换为MixLinear，固定初始NLI线性权重W0作为参考，训练时使用：

```text
W_effective = W0 + mask × (W - W0) / (1 - p)
p = 0.1，mask ~ Bernoulli(0.9)
```

同一输入维度的mask在所有输出维度共享，沿用作者线性层的掩码方式。参考权重是buffer，不接收梯度，不随epoch、Row或Pair刷新。原生autograd对上述公式求导；作者旧代码的自定义backward省略了1/(1-p)，本实现不沿用这一梯度缩放差异。

Row和Pair阶段都会启用Mixout。bias、embedding、LayerNorm以及新增CLS／Alignment／Lexical层仍按原方式训练，不做Mixout。原有dropout也保留；作者示例会关闭部分dropout，因此这里是保持本项目控制变量的Mixout适配实验，不是照搬其全部配方。

方法针对小样本微调中过度偏离初始权重的问题。它不消除三组合Pair CE对F/B/E公共平移不敏感的性质，也不保证提高分数。

## 一轮流程与参数

1. 对3042行完成Row四分类训练。
2. 评估Row后的independent与pair-aware，保存预测和四类logits。
3. 对1282个真实双行组完成三组合Pair CE训练。
4. 评估Pair后两项指标，保存预测、logits及修好／破坏统计。

两个阶段共享一个AdamW，延续原run02的优化器状态。额外评估会恢复随机数状态，不额外改变后续训练的随机序列。

| 参数 | 值 |
|---|---|
| 主干 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| 主干、CLS、Alignment学习率 | 2e-5 |
| Lexical学习率／隐藏维度 | 1e-3／32 |
| Row类别权重F/B/E/N | 1／1／1／1.5 |
| Pair候选权重FB/BF/EE | 1／1／1.25 |
| Pair损失系数／label smoothing | 0.5／0.05 |
| batch | Row16行，Pair16组，Dev16行 |
| 轮数／最大长度 | 10／128 |
| 精度／weight decay | float32／0.01 |

Mixout不增加主干前向次数，但固定参考权重、掩码及权重计算会增加显存和时间，实际开销看日志。参考权重容量和替换层列表分别打印到控制台、记录在mixout_layers.json。

## 运行方式

直接运行训练入口即可，默认p=0.1，seed42。重新实验修改[config.py](config.py)中的RUN_NAME；已有目录不会被覆盖。命令行也可以使用：

```powershell
python experiments/deberta_lexical_mixout/train_deberta_mixout.py
```

p=0同代码对照（无需现在重新跑，已有LLRD 1.0结果可作历史对照）：

```powershell
python experiments/deberta_lexical_mixout/train_deberta_mixout.py --mixout 0
```

p=0不替换模块、不生成mask。CPU检查验证了这一分支与原Row→Pair训练在小模型上逐参数一致；真实模型完整复现仍需用户实际运行，不能用CPU检查替代。

如果先用Train内部验证，运行时加`--validation internal`。内部划分合并相同pair_id及规范化相同无序文本对，正反行不会拆开；该分支不读取官方Dev。训练／内部验证是2430／612行，与3042／660的官方口径分开报告。

## 保存哪些结果

默认模型目录：`model/deberta_lexical_mixout/run01_mixout0p1_official/seed42/`。

- `best_pair_aware.pth`：只在每轮Pair后按pair-aware F1选最佳，保持run02口径。
- `best_independent.pth`：在每轮Row后、Pair后按independent F1选最佳。
- 对应的`best_*_metrics.json`记录epoch、阶段以及该份权重的两项成绩，平分保留更早的一份。

只维护上述两份最佳权重，不为每轮保存模型。它们是不同时间选出的两个单模型，不融合；不能把两份权重各自最好的指标写成同一个checkpoint的成绩。存储需要容纳两份完整模型以及保存时的一份临时文件。

训练时固定参考权重不写入这两份预测checkpoint；保存的是普通原模型state_dict。评估脚本按原模型严格加载，用训练好的普通权重预测，既不采样Mixout，也不需要初始NLI权重。预测仍用原pair-aware；单例由模型四分类，不强制N。

本目录结果在`outputs/run01_mixout0p1_official/seed42/`：

```text
epoch_summary.csv
best_pair_aware_metrics.json / best_pair_aware_summary.csv
best_independent_metrics.json / best_independent_summary.csv
training_data.csv / validation_data.csv / validation_reference.csv
run_config.json / learning_rates.csv / mixout_layers.json
epoch03/
  after_row/predictions_with_scores.csv
  after_pair/predictions_with_scores.csv
  pair_changes.csv
```

两阶段的predictions_with_scores.csv都保存F/B/E/N原始logits及两种预测。

右键运行[evaluate_best.py](evaluate_best.py)重新评估结构最佳权重。评估独立最佳权重用：

```powershell
python experiments/deberta_lexical_mixout/evaluate_best.py --kind independent
```

## 比较规则

先与原run02的0.930569比较结构分，再看同一个checkpoint的独立分及E↔F/B错误。独立分改善而结构分下降属于取舍，不能直接宣称整体提升；单种子改善后再固定配方补种子。

检查入口：[check_setup.py](check_setup.py)。它检查历史参数、数据划分、掩码与梯度、参考权重冻结、p=0训练一致性、预测权重严格重载和两种选模规则，不运行真实模型训练。

方法来源：[Mixout论文，ICLR 2020](https://arxiv.org/abs/1909.11299)、[作者仓库bloodwass/mixout](https://github.com/bloodwass/mixout)，关键文件为`mixout/mixout.py`、`mixout/mixlinear.py`、`example_huggingface.py`。本目录用原生PyTorch实现公式，不额外依赖其旧版包。
