# Lexical + 原0.938的3290行

直接右键运行 `train_deberta_lexical_relation.py`。

```python
RUN_NAME = 'run02_lexical_relation_staged'
TRAINING_MODE = 'staged'
SELECTION_METRIC = 'pair_aware'
```

模型为DeBERTa + Token Alignment + 原20维Lexical分支。直接复用原0.938保存的训练快照：官方3042 + scope200 + 关系48，共3290行。没有新增数据，没有加入局部语义头。

训练沿用以前方式：全参数、FP32、seed42、10轮、Row batch16、Pair batch16组；主模型2e-5、Lexical 1e-3；Row权重[1,1,1,1.5]，Pair权重[1,1,1.25]、平滑0.05、总Pair权重0.5。每轮先206步Row CE，再87步Pair CE。

按Pair阶段后的Dev pair-aware F1保存唯一最佳模型，同时保存该轮independent分数。两阶段的预测分别保存，不保存每轮模型。后续复验改RUN_NAME。

- 模型：`D:/model/SEAMEVAL/deberta_lexical_aug/run02_lexical_relation_staged/best_model.pth`
- 结果：本目录`outputs/run02_lexical_relation_staged/`
- 重新评估：右键 `evaluate_lexical_relation.py`。
- 只准备数据：训练入口加 `--prepare-only`，不会加载模型或训练。

原Lexical+scope200脚本、原0.938脚本与旧结果均保留。本次复用旧训练数据及其开发过程，不重新声称人工增强是独立于历史Dev分析设计的。
