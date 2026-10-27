# Qwen3-1.7B：原 C + r16 SFT + 预测期 pair-aware

本目录沿用 `experiments/qwen_lora` 主线的流程和文件命名：标准 SFT 训练、C 数据、每轮 F1 选模、最佳 Adapter 预测。原 0.6B 实验不修改。

## 运行顺序

使用现有 nlpbase 解释器，按顺序右键运行：

1. `download_model.py`：ModelScope 下载 `Qwen/Qwen3-1.7B` 到 `D:/LLM/model/Qwen3-1.7B`。如果你已在下载同一个目录，等现有下载完成即可。
2. `check_setup.py`：检查 tokenizer、全部输入、模型配置和权重分片；不加载真实模型权重。
3. `train_lora_abc.py`：正式训练，自动保留 Dev pair-aware F1 最高的一轮。
4. `evaluate_lora_pair.py`：加载 `best_adapter`，输出 independent 和 pair-aware 成绩。

不需要调整 PyCharm 工作目录。

## 与原 qwen_lora 主线的区别

- 基础模型换为经过指令训练的 `Qwen/Qwen3-1.7B`，不是 `Qwen3-1.7B-Base`。
- tokenizer 随模型一起换，不读取原 0.6B 的全局 tokenizer。
- 原短 Prompt 的任务内容保持一致，使用 1.7B 自己的 chat template，并明确 `enable_thinking=False`。
- 关闭思考时，官方模板可能在输入末尾预填 `<think>\n\n</think>\n\n`。这个空块属于输入前缀，不是额外的训练目标。completion 仍只有 F/B/E/N 一个 token 加 EOS。
- micro-batch=8、gradient_accumulation_steps=2，有效 batch=16；Dev loss 的 batch=8，选模与正式预测的 batch 都为16。显存峰值需要实际训练确认。
- 开启梯度检查点和分块全词表 CE，减少显存占用；没有改成四类 CE。

## 沿用的设置

- C 数据直接调用原 `qwen_lora/data_variants.py`：3042 + 475 条反向 N = 3517 条 Train；Dev 660 条。
- r=16、alpha=32、dropout=0.05、all-linear；lr=1e-4、5 epochs、bf16、seed=42。
- 标准 completion-only SFT，没有 Pair CE、对比损失、alignment 或模型集成。
- 每轮按 `eval_pair_f1` 保存最佳 checkpoint，训练结束自动加载最佳权重，再导出 `best_adapter`。
- 预测只生成一步，读取四个候选分数，另存全词表生成、非法输出数和两种读出的差异。原 0.6B 基线上的两种读出完全一致。
- pair-aware 仍只对两行组选择 FB/BF/EE，单例保留模型预测，不强制改成 N。

## 路径

参数和路径在 `config.py`、`train_lora_abc.py`。

- 最佳 Adapter：`D:/LLM/train_model/DiCo-NLI/qwen3_1p7b/run01/C/best_adapter`
- 每轮评分：同一训练目录下 `dev_scores/epoch_*/`
- 最终评分：本目录 `outputs/run01/C/`
- 数据缓存：项目根目录 `.hf_q17`，使用短路径避免 Windows 长路径错误。

公共预测和官方评分代码复用 `experiments/generative_nli.py`。模型、tokenizer 和完整 Prompt 均由本目录显式传入，不会误用其他实验的模型。

`test_training_cpu.py` 仅用真实 tokenizer 加随机小 Qwen 做两步 CPU 流程检查，不训练真正的 1.7B；正常跑实验不需要运行它。

模型说明：https://huggingface.co/Qwen/Qwen3-1.7B
