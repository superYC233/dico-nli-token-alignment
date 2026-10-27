# Qwen：MNLI 中间任务微调 → DiCo C+r16

默认模型是原来的 `Qwen3-0.6B-Base`。所有新增内容放在本目录，不修改原 Qwen / DeBERTa 实验。

## 运行顺序

1. 运行 `train_mnli.py`：首次自动下载 MNLI，固定抽取 30000 条训练 + 1000 条内部验证；训练 1 轮，保存 `mnli/final_adapter`。
2. 运行 `train_dico.py`：自动加载上一步 Adapter，**继续更新同一个 LoRA**；使用原 C 数据、r16/alpha32、短 Prompt、完整词表 completion-only SFT；训练 5 轮。
3. 运行 `evaluate_dico.py`：评估第二阶段的 `best_adapter`，同时报告独立预测与 pair-aware。

三个脚本都可以在 IDE 中直接运行。无需先合并基础模型，也无需改旧 `train_lora_abc.py`。
**不能直接运行不做修改的旧训练脚本接续第一阶段**，因为旧脚本会重新初始化 LoRA；第二步请用这里的 `train_dico.py`。

模型输出默认位于：

```text
D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_mnli/run02_fast/
  mnli/final_adapter/           第一阶段 Adapter
  dico_from_mnli/C/
    checkpoint-.../            最佳 checkpoint，最多保留一个
    best_adapter/              同一个最佳模型的便捷导出
```

第一阶段只在结尾保存模型；第二阶段只保留一个最佳 checkpoint。`best_adapter` 是它的便捷副本，不是另一个轮次。
只保存 LoRA，不复制整份 Qwen 权重，也不保存优化器续训状态。
第二阶段重建优化器与学习率计划，但保留第一阶段学到的 LoRA 权重。
最终 Adapter 已经包含两阶段训练的结果，预测时只加载原始 Base + 最终 Adapter，不需要再叠加 MNLI Adapter。

## 数据和标签

- 数据源：[nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli)。只使用它的 `train`，不用其官方 validation/test 训练。
- 标签按数据集自身的 `ClassLabel.names` 读取，保留 `entailment / neutral / contradiction`，不把 entailment 等同于 DiCo 的 F。
- 三类近似等量，种子固定为 42。内部验证也从 Train 划出；同一标准化 premise 不会跨训练/内部验证分区。
- 去掉所选数据中的有序重复对；超过 512 token 的整条样本跳过，防止截断掉答案。
- 用 NFKC + casefold + 空白归一化，排除与可见 DiCo Train/Dev 任一文本完全相同的 MNLI 样本。该排除只看文本，不用 DiCo 标签。
- 该检查不能排除近似改写、子串或隐藏 Test 的重叠，不能作为“绝无数据重叠”的证明。
- 保存抽样后的完整 CSV、源行号、MNLI pairID、数据指纹、筛选数量及 SHA256；复用缓存时校验配置和文件哈希。
- 第一阶段从未用 DiCo Dev 选模。第二阶段仍按 Dev pair-aware W-F1 选最佳，和当前主线口径相同。

准备好的样本和清单在 `outputs/run01/prepared_mnli/`，由 `MNLI_DATA_RUN_NAME` 指定；训练输出则由 `RUN_NAME` 指定。这样加速重跑可以直接复用已准备的数据。下载后的完整源数据缓存放项目根目录 `.hf_qmnli/`，节省 Windows 路径长度。
`MNLI_REVISION` 默认为 `main`；正式复验应保留抽样 CSV / 清单，或将此项改为确认过的 Hub commit。

## 配置

所有参数在 `config.py`：

- `MNLI_TRAIN_ROWS=30000`、`MNLI_EPOCHS=1`：首次只固定这一组探索，没有保证涨分；长句较多，不能套用 DiCo 的每轮耗时。
- 第一阶段由 batch 4 × 累计 4 改成 16 × 1；第二阶段仍为 16 × 1；有效 batch 均为 16。第一阶段验证 batch 也改为 16。保持梯度检查点，不减少样本或截断长度。批次形状变化可能导致数值和 dropout 序列不同，不保证逐位复现旧轨迹。
- 两阶段 lr 都先固定 `1e-4`，不同时扫参数。
- 两阶段均使用 TRL 的 `chunked_nll`：仍是完整词表 CE，只分块计算监督位置的 logits；不是四类 CE，也没加 Pair CE 或词汇分支。
- bf16、非重入梯度检查点用于控制显存；右补齐训练、左补齐预测；Prompt 不参与损失，答案和 EOS 参与损失。
- 改训练参数 / 重跑时修改 `RUN_NAME`，脚本会拒绝覆盖已有训练目录。改变数据准备参数时，另改 `MNLI_DATA_RUN_NAME`，避免误用不同抽样名单。
- 如果只想先跑小规模，**第一次运行前**将 `MNLI_TRAIN_ROWS` 改为 10000；不要把不同规模当作同一次实验。

可选的严格对照：设置 `USE_MNLI_ADAPTER=False`，只运行第二步，会输出到 `dico_from_base`。这能在相同的新训练代码下比较“补 MNLI / 不补 MNLI”，避免把旧代码版本差异当收益。

正在运行的 Python 进程不会自动读取文件修改。此次加速需要先停止旧进程，再运行 `train_mnli.py`；使用新目录从头训练，不续接旧进程的临时状态。若 16 条出现 CUDA OOM，可改成 8 × 2 并再次更换 `RUN_NAME`。

## 如何判断效果

主要比较最终 pair-aware W-F1，同时看 N 与 F/B/E 正确数；不以 N 单项增加代替总体改善。
增加了外部监督和训练预算，报告需披露 MNLI 来源、筛选/抽样规则及额外训练步数。
即使没有提升，也不能据此证明 Qwen 的问题必然来自架构，或断言它原来没有接触过 NLI 数据。

## 本地检查

运行 `test_cpu.py`：只使用真实 tokenizer 与随机微型 Qwen 检查两阶段 Adapter 接续、监督位置、保存重载、选模和解码；不下载 MNLI，不加载大模型，不启动正式 GPU 训练。
