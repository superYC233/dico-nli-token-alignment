# 原版 DeBERTa + 范围增强

本目录只放原版全参数 DeBERTa + Token Alignment 的增强和复制对照。
冻结参数实验保留在 `experiments/deberta_row_ce_frozen`。

## 训练入口

右键运行本目录 `train_original_aug.py`。训练配置在该文件顶部：

```python
RUN_NAME = 'run02_original_copy'
AUGMENT_MODE = 'copy'
```

`scope` 是原始 3042 行加 200 行范围增强；`copy` 是相同数量、相同类别的原样复制对照；`none` 是原始训练集。换实验只改 RUN_NAME 和 AUGMENT_MODE。

保持原版全参数训练、float32、lr=2e-5、batch16、10 轮、seed42。每轮先对所有行做 Row CE，再对真实配对组做 `0.5 * Pair CE`；N 单例只参加 Row CE。scope/copy 均为 3242 行、1362 个配对组，每轮 203 个 Row 步和 86 个 Pair 步。

按 Dev pair-aware W-F1 选最佳轮次，只保存一份最佳模型，同时报告独立预测。诊断在最佳模型选好后评估，不参与选模。

## 已有结果和保存位置

- `outputs/run01_original_scope`：范围增强已完成，第 3 轮 pair-aware W-F1=0.935493、独立 W-F1=0.480876。
- `outputs/run02_original_copy`：当前复制对照的数据已准备，训练由用户运行。
- 模型权重位置保持 `D:/model/SEAMEVAL/deberta_original_aug/<RUN_NAME>/best_model.pth`。

此前存放在冻结目录下的原版增强/复制结果已整体移入本目录，原始预测、分数和训练配置保留。

增强标签是未独立审核的人工模板标签。训练仍从相同 NLI 初始化重新开始，不加载原版 epoch7 续训，不增加词汇分支。

## 文件职责

- `train_original_aug.py`：本目录独立训练配置、原版两阶段循环、选模与保存。
- `scope_augmentation.py`：纯数据构造；不依赖冻结配置，也供冻结实验复用。
- `outputs/<RUN_NAME>`：训练数据快照、逐轮预测、分数和诊断。

只准备数据时使用 `train_original_aug.py --prepare-only`；默认右键运行会正式训练。

## 数据重合审计（不训练）

右键运行 `audit_augmentation.py`，默认检查原版和词汇增强共用的 200 条 scope 增强。
结果写入原版 scope 结果目录的 `overlap_audit`，不修改已有训练数据。

检查正向/反向文本对的原样重合、Unicode/大小写/空白规范化重合、忽略标点的候选重合，
并导出近似文本候选供人工复核。原始 Train–Dev 已有重合另表报告。
`augmentation_audit.csv` 中引用的来源 ID 会核对是否来自 Train；这不等于独立审核了增强标签。
JSON 保存输入文件及脚本 SHA256。零精确重合不能证明没有语义重合或方法选择偏差。

Test 文本尚不可见时，报告明确标注未检查；之后可仅提供无标签文本再核验：

```powershell
G:/anaconda3/envs/nlpbase/python.exe experiments/deberta_original_aug/audit_augmentation.py --test-path "官方Test文本.csv" --output-dir "experiments/deberta_original_aug/outputs/run01_original_scope/overlap_audit_with_test"
```

用 `--augmentation` 指定其他实验保存的 `extra_train_rows.csv`，可以核验其他增强版本。
如需按审计结果重新过滤训练数据，须遵守届时的 Test 使用规则，并另外记录改动；本脚本只读检查。
