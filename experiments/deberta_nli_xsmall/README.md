# DeBERTa NLI xsmall

先运行 `download_model.py`，下载完成后运行 `train_deberta_xsmall.py`。
所有训练由你启动，脚本不会自动接着跑其他实验。

## 本次改什么

- 主干改成 `cross-encoder/nli-deberta-v3-xsmall`，约7080万参数，隐藏维度384。
- 同时使用这个checkpoint的Tokenizer。
- 训练数据仍为官方原始Train的3042行；不做增强、OOF或难例加权。
- 保留原Token Alignment、20维词汇特征、词汇分支。
- 保留四类混合分组训练：同一次前向计算 `Row CE + 0.5 * Pair CE`。
- 参数对照 `deberta_lexical_mixed_pair/outputs/run01_mixed_pair/run_config.json`：主干学习率2e-5、词汇分支1e-3、10轮、float32、每步8组、推理batch16。
- 按Dev独立预测W-F1保存一个最佳模型，同时报告该模型的pair-aware成绩。Base对照按同样规则取其第2轮；原Base目录的best_model.pth是按pair-aware保存的第3轮，不能混作同一个checkpoint。

这里替换的是预训练checkpoint，因此尺寸和之前的NLI训练数据都发生了变化。
一次结果不能单独证明差异只来自参数量；本实验用于判断这个更小的现成模型是否适用。

## 文件

- `config.py`：模型路径、RUN_NAME和训练参数。
- `download_model.py`：下载到 `D:/LLM/model/nli-deberta-v3-xsmall`，仅下载一份safetensors。
- `data.py`：原混合训练的数据分组逻辑。
- `train_deberta_xsmall.py`：训练并保存最佳模型。
- `evaluate_best.py`：训练后可单独重评最佳模型。
- `evaluate_train_dev.py`：同一最佳模型评估Train和Dev，检查泛化差距。
- `test_cpu.py`：数据、损失、384维前向和推理模式检查，不下载模型、不训练。

模型、词汇特征和评分复用 `deberta_lexical_stage_probe` 中现有实现，未修改旧文件。
原checkpoint的三分类头不用于四分类；四分类头和Alignment仍重新初始化。

## 运行顺序

1. 运行 `download_model.py`。
2. 运行 `train_deberta_xsmall.py`。
3. 训练后运行 `evaluate_train_dev.py`，同时检查Train/Dev的独立和结构成绩。

默认模型目录：`D:/model/SEAMEVAL/deberta_nli_xsmall/run01_xsmall_mixed_pair/`。
默认结果目录：本目录下 `outputs/run01_xsmall_mixed_pair/`。

训练只在最佳分数刷新时覆盖 `best_model.pth`，不会每轮保留模型权重。
每轮预测和指标CSV会保留在结果目录，汇总为 `epoch_summary.csv`。
`best_summary.csv` 对应已保存模型，不能把别的轮次的两种指标拼在一起。

重复实验只改 `config.py` 的 `RUN_NAME`；更换seed时也应更换RUN_NAME。
已有输出目录不会覆盖。Dev用于选模和报告，不进入训练损失；它已被反复用于开发，不能当作未使用测试集。
