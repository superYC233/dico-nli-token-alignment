# DeBERTa 最小对照诊断

当前入口：`run_all.py`。右键运行即可，**不训练、不修改原模型或 Train/Dev**。

```powershell
& G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_minimal_contrasts/run_all.py
```

当前 `config.py` 的 RUN_NAME 是 `run02_binding`。`run01` 是第一轮探索记录，不需要另外操作。

## 测 0.9248 原版

右键运行 `evaluate_original.py`。它固定加载原版 `epoch_7.pth`，先逐条复现官方 Dev 历史预测，再评估同一批 200 条新诊断题。无需改冻结版训练配置，也不会启动训练。

结果保存到 `outputs/run03_original_binding`：`dev/summary.csv` 是原版复现成绩；`independent/provisional_summary.csv` 是独立诊断；`pair_aware/provisional_summary.csv` 是使用自建分组结构后的诊断。后者依赖分组元数据，不能当作纯语义判断能力。两种诊断都区分未审核建议标签与已批准标签。

## 已经完成

- 从 Train 选择 20 个中心词不同的来源家族，记录原始 instance_id。
- 每个家族建立原题、限定改写、等价改写、并列修饰范围、对象替换、属性归属、并列换序对照。
- 共 240 行；2 条原题与 Dev 重叠，整对排除。实际评估 238 行：38 条已见来源对照、200 条新探针。
- 200 条新探针均未与 Train 或 Dev 的有序/反向规范化文本对重合。
- 使用现有纯 Row CE 冻结模型 epoch8，batch16、float32、长度128。
- 重跑 Dev 与原预测逐条一致。诊断只看独立预测，不报告自建 pair_id 的 pair-aware 分数。

## 先看什么

1. `REPORT.md`：结果及下一步解释。
2. `prepared/run02_binding/review_focus.csv`：60 行范围/归属/顺序对照，无模型答案。
3. `outputs/run02_binding/provisional_summary.csv`：各类建议标签匹配数。
4. `outputs/run02_binding/disagreements_to_review.csv`：模型与建议标签不一致的明细。

## 如何审核

全部新探针初始为 `pending`。建议标签是研究假设，尚未经过独立人工审核；审核前没有正式的“已审核准确率”。

正式审核文件是 `prepared/run02_binding/review.csv`。每行仅改三个字段：

- `review_status`：同意并确定标签时填 `approved`；无法确定时保持 `pending`；不用该题时填 `rejected`。
- `review_label`：`approved` 行填写你确认的 F/B/E/N，可以与建议标签不同。
- `review_note`：记录理由或疑问。

不要改文本、ID、来源或 proposed_label；要改题目时应另建 RUN_NAME。审核后重新运行 `run_all.py`，不会覆盖审核意见。

只有批准的新探针进入 `approved_summary.json`、`approved_predictions.csv` 和 `approved_confusion.csv`；已见 Train 来源不计为新增探针成绩。`review_focus.csv` 只是方便阅读的浓缩表，不自动修改审核状态。

## 文件职责

- `config.py`：模型位置、20 个 Train 来源、输出位置。
- `prepare_diagnostics.py`：生成候选、记录来源、排除 Dev 重叠、保留审核意见。
- `evaluate_diagnostics.py`：加载旧模型、复现 Dev、预测探针、区分临时和审核后成绩。
- `run_all.py`：一键入口。
- `test_diagnostics.py`：五项数据和审核逻辑检查，不加载模型。

这些家族共享少量模板，不能当作 20 种独立句法现象。探针开发已受此前 Dev 分析启发，不是新的独立测试集。后续若做增强，请另选新增增强家族，保留这里的探针；原 Train 里已有来源仍然存在，因此只能检验额外结构组合的迁移，不是完全未见词汇泛化。
