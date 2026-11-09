# 泛化与数据审计

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

先阅读 `REPORT.md`。本次已经运行完，不需要重新训练。

目标固定为 `deberta_row_ce_frozen/run01_freeze6` 的 epoch 8 最佳模型。
原始数据、原训练脚本、模型权重保持原样。审计输出位于 `outputs/row_ce_frozen_epoch8/`。

复现顺序：

```powershell
python experiments/deberta_generalization_audit/export_predictions.py
python experiments/deberta_generalization_audit/prepare_sources.py
python experiments/deberta_generalization_audit/analyze.py
python experiments/deberta_generalization_audit/review_errors.py
python experiments/deberta_generalization_audit/probe_surface.py
python experiments/deberta_generalization_audit/validate_audit.py
```

`export_predictions.py` 会验证权重/数据哈希，未改变时复用导出结果。首次推理及探针需要加载原模型；下载脚本仅首次需要联网。

初审表不是修订金标。探针不是正式 Dev；所有审计文件均不得直接当作新增训练集。
