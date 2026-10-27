# DeBERTa NLI large 对照

模型：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。

模型卡：https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

## 按顺序运行

本机已经有完整的预训练模型缓存，`config.py` 已指向该缓存，可以直接右键运行
`train_deberta_large.py`。换机器需要下载时，先运行 `download_model.py`。
训练结束已经有每轮分数；需要重新加载最佳权重核对时，
再右键运行 `evaluate_best.py`。

也可以在 PowerShell 中逐行运行：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_nli_large/download_model.py'
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_nli_large/train_deberta_large.py'
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_nli_large/evaluate_best.py'
```

下载脚本使用 Hugging Face，固定同一个 revision，复用已有缓存，保留 safetensors
权重和 tokenizer，不下载重复的 bin 和 ONNX 权重。下载中断后可以重新运行同一脚本。
训练和评估只读本地模型文件。

## 文件怎么对应

| 文件 | 用途 |
| --- | --- |
| `config.py` | 模型路径、RUN_NAME、学习率、batch、损失权重、选模依据 |
| `download_model.py` | 下载预训练模型；不会训练 |
| `data.py` | 读取原始 Train/Dev，提取词汇特征，把可逆对作为训练组 |
| `lexical_features.py` | 与原词汇版相同的 20 维文本特征 |
| `model.py` | CLS + Token Alignment + Lexical，隐藏维度自动适配 large |
| `evaluation.py` | 独立预测、pair-aware 解码、官方评分、混淆矩阵 |
| `train_deberta_large.py` | 每批联合训练，每轮评估，刷新独立 F1 时保存最佳权重 |
| `evaluate_best.py` | 加载 best_model.pth，再评估同一份 Dev |
| `evaluate_train_dev.py` | 同一份最佳权重分别评估 Train/Dev，检查拟合与泛化差距，只推理 |
| `test_cpu.py` | 不下载模型、不训练的 CPU 数据、损失、推理和保存加载检查 |

## 本次具体设置

- 原始 Train 3042 行，Dev 660 行，不增加数据。
- 全量微调，不使用 LoRA，也不冻结底层。
- 沿用 `deberta_lexical_mixed_pair/run01_mixed_pair` 的模型结构及损失。
- 主干、CLS 和 Token Alignment 学习率 `2e-5`，词汇分支 `1e-3`。
- `Row CE + 0.5 * Pair CE`，Row 权重 `[1, 1, 1, 1.5]`，Pair 权重
  `[1, 1, 1.25]`，Pair label smoothing `0.05`。
- 8 组/批，最多 16 行；每轮 1760 组、220 步；10 轮、seed 42。
- 每轮所有训练行各参与一次 Row CE，1282 个真实可逆对各参与一次 Pair CE。
- 开启 bf16 autocast 和梯度检查点，参数保持 float32；AdamW `foreach=False`
  减少更新时的临时显存。训练中评估和单独评估均固定 batch 16、bf16。
- **按 independent Weighted F1 保存最佳模型**。同时记录该 checkpoint 的
  pair-aware 分数，不能把另一轮的 pair-aware 峰值说成这份权重的成绩。
- 只覆盖更新一份 `best_model.pth`，不会每轮留下整份模型权重。
  每轮的预测 CSV 和分数仍会保留。

这不是只改变模型容量的严格消融：除了 base 换 large，NLI 初始化的数据组合也不同，
并且为控制显存启用了混合精度和梯度检查点。比较时应说明这些设置。

## 保存在哪里

- 预训练缓存根目录：`D:/BaiduNetdiskDownload/PretrainedModel/models`
- 实际加载目录：该缓存下
  `models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli/snapshots/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`
- 训练权重：`D:/model/SEAMEVAL/deberta_nli_large/run01_large_mixed_pair/best_model.pth`
- 本轮模型信息：同目录的 `best_metrics.json` 和 `run_config.json`
- 每轮成绩：本实验目录下 `outputs/run01_large_mixed_pair/epoch_summary.csv`
- 最佳权重两种成绩：同结果目录的 `best_summary.csv`

下一次实验只改 `config.py` 的 `RUN_NAME`。已有同名输出目录时脚本会停止，避免覆盖
旧实验；不需要再建立一份代码目录。评估时让 `RUN_NAME` 指向要评估的那次运行。

针对 12GB 显存启用了节省显存的设置，但实际峰值需要在你启动训练后确认。
若仍显存不足，先把 `GROUP_BATCH_SIZE` 改为 4；这会改变每批组成和每轮更新次数，
应当作为另一次运行记录，而不是宣称与 batch 8 完全等价。

## 已检查与未检查

已完成语法和 CPU 检查，没有替你下载大模型或启动训练。
CPU 检查覆盖：3042 行每轮各一次、1282 对的索引/方向、N 单例监督、联合损失、
Dev 预测不消耗训练随机数、模型适配 1024 维隐藏状态、保存加载后的输出一致。

另外用随机初始化的极小 DeBERTa 在本机 GPU 上验证了 bf16 前向推理和梯度检查点
接口兼容；没有执行反向传播或参数更新。这不等于验证过 large 的训练显存。

实际 large 权重的加载、训练速度、显存峰值和最终成绩，需要下载并运行后确认。
