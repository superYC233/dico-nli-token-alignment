# SmolLM2-1.7B Base + C 数据 + LoRA

按顺序右键运行，继续使用 `G:/anaconda3/envs/nlpbase/python.exe`：

1. `download_model.py`：第一次下载模型及 tokenizer。权重约 3.4GB，保存到 `D:/LLM/model/SmolLM2-1.7B`。
2. `train_lora.py`：从该基础模型重新训练，每轮预测 Dev，按 pair-aware W-F1 保存最佳 Adapter。
3. `evaluate_lora_pair.py`：加载 `best_adapter`，输出独立预测与 pair-aware 的官方成绩。

路径均在 `smol_config.py`，不需要改变 PyCharm 工作目录。训练与预测入口必须使用本目录的文件。

模型输出：`D:/LLM/train_model/DiCo-NLI/smollm2_1p7b/run01/C/best_adapter`。

评分输出：本目录的 `outputs/run01/C/`。

## 本轮设置

- Base 版，使用与原 Qwen 一致的旧短 Prompt；不使用聊天模板。
- 复用原 `qwen_lora/data_variants.py` 的 C 构造函数，当前 Train 3042 + 475 条反向 N = 3517，Dev 660。
- r=16、alpha=32、dropout=0.05、all-linear、lr=1e-4、5 epochs、bf16、completion-only 全词表 SFT（标签 + EOS）。
- 为控制 12GB 显存，训练 micro-batch 4、累积 4 次，有效 batch 16；启用梯度检查点和分块 CE。实际显存仍需运行后确认。
- 原生 Dev loss 以 batch 8 计算；用于选模的生成预测始终 batch 16。每轮记录两种解码成绩，按 pair-aware W-F1 选模。
- 使用 SmolLM2 tokenizer，训练前检查全部输入长度、四个单 token 标签和 completion 边界。
- 正式读出为四个候选的 argmax，同时记录全词表一步生成及非法输出数。原 Qwen 基线 660 条的这两种读出完全一致；新模型若不一致会明确报告。
- 沿用 FB/BF/EE 结构解码，单例保留模型预测；不是单例强制 N。

此处没有修改 Qwen 主线，没有辅助损失，也没有模型集成。这是主干、tokenizer、参数量及 micro-batch 布局一起变化的跨模型对照，不能只把差异归因于模型家族。

模型来源：https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
