# 原run02词汇版：Row与Pair分开AdamW

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

直接右键运行 [train_deberta_separate_optimizers.py](train_deberta_separate_optimizers.py)。默认只跑`separate`实验，正式训练交给使用者。

数据是官方Train 3042行、官方Track1英文Dev 660行。模型是原run02的DeBERTa＋CLS＋Token Alignment＋Lexical。没有新增文本、复制或反向N，也没有Mixout、LLRD。使用已有NLI模型缓存，无需下载新模型。

## 改了哪里

原来Row和Pair共用一个AdamW。现在两个阶段使用同一份模型权重，各自维护一个AdamW：

```python
# 只在训练开始前创建一次
row_optimizer = build_optimizer(model)
pair_optimizer = build_optimizer(model)

for epoch in range(1, EPOCHS + 1):
    train_rows(model, row_loader, row_optimizer, row_criterion, device)
    # 评估Row后，保存独立分最佳权重
    train_pairs(model, pair_loader, pair_optimizer, pair_criterion, device)
    # 评估Pair后，保存独立分／结构分最佳权重
```

每个优化器都管理同一个模型的全部可训练参数。Row使用自己的动量与步数，Pair使用自己的动量与步数；它们分别跨轮保留，切换阶段时不重建、不清状态。两个优化器先后更新同一份权重，预测时仍是一个模型。

这可以检验共享AdamW历史状态是否加剧Row→Pair的退化。它会同时分开一阶动量、二阶动量和Adam步数，结果不能仅归因于其中某一项。Pair的N输出行不再继承Row动量，但共享主干的更新、weight decay仍可能改变N预测。三组合Pair CE的平移不变性也仍然存在，因此不保证涨分或消除退化。

## 参数和每轮流程

固定恢复run02配置，不读取原目录后来修改过的训练超参数：

| 项目 | 设置 |
|---|---|
| 主干 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| 主干、CLS、Alignment学习率 | 2e-5 |
| Lexical学习率／隐藏维度 | 1e-3／32 |
| Row权重F/B/E/N | 1／1／1／1.5 |
| Pair权重FB/BF/EE | 1／1／1.25 |
| Pair系数／label smoothing | 0.5／0.05 |
| batch | Row16行、Pair16组、Dev16行 |
| 轮数／最大长度／精度 | 10／128／float32 |
| weight decay | 0.01 |

每轮先对3042行完成191个Row批次，保存Row后的预测和四类logits；再对1282个真实双行组完成81个Pair批次，保存Pair后的相同结果及修好／破坏统计。单例N只参与Row，预测时仍由模型四分类，不强制N。

主干前向次数不增加。两个AdamW会比原版多保存一套一阶、二阶状态，float32下大约增加1.37 GiB显存；默认batch保持原配置。

## 运行

默认只运行双优化器：

```powershell
python experiments/deberta_lexical_separate_optimizers/train_deberta_separate_optimizers.py
```

已有LLRD 1.0结果可作历史共享优化器对照，不自动重复训练。如需同代码共享对照，运行：

```powershell
python experiments/deberta_lexical_separate_optimizers/train_deberta_separate_optimizers.py --optimizer shared
```

配置在[config.py](config.py)。另开实验改RUN_NAME，补种子改SEED；已有实验目录不会覆盖。`--prepare-only`只检查数据；`--validation internal`从Train按来源和规范化无序文本对整组划分2430／612行，不读取官方Dev。

## 保存与评估

默认模型目录：`model/deberta_lexical_separate_optimizers/run01_separate_official/seed42/`。

- `best_pair_aware.pth`：只从每轮Pair后按结构Weighted F1选最佳，保持原run02口径。
- `best_independent.pth`：从每轮Row后、Pair后按独立Weighted F1选最佳。
- `best_*_metrics.json`：保存各自epoch、阶段和同一份权重的两项成绩，平分保留更早者。

只保留两份最佳权重，不逐轮存模型。它们各自都是单模型，不融合，不能把各自最高分拼成同一个checkpoint的成绩。只保存预测权重，不保存优化器续训状态；“跨轮保留”指同一次训练进程内保留。

本目录`outputs/run01_separate_official/seed42/`保存：

- `epoch_summary.csv`：每轮两个阶段、两种预测口径。
- `epoch03/after_row/predictions_with_scores.csv`与`after_pair/...`：四类logits、独立预测与结构预测。
- `epoch03/pair_changes.csv`：Pair前后修好、破坏及非N→N数量。
- `optimizer_states.csv`：各阶段两个优化器的步数、状态容量和三个输出头N行的一阶动量范数，用于核对状态隔离。
- `training_data.csv`、`validation_data.csv`、`run_config.json`：训练／验证快照和配置。

右键运行[evaluate_best.py](evaluate_best.py)重新评估结构最佳权重。评估独立最佳权重加`--kind independent`。加载的是普通完整模型，不需要恢复优化器。

CPU检查入口：[check_setup.py](check_setup.py)。检查共享分支与原训练逐参数一致、两个优化器管理同一份模型但状态互不修改、状态跨轮保留、N输出动量与权重重载。小模型检查中的分数是测试夹具，不是真实训练结果。

比较重点是同一轮Pair后独立分下降多少、非N→N是否减少，以及结构分能否保持或超过原run02的0.930569。状态隔离通过只说明实验实现正确，不证明它一定改善Dev。
