# Qwen3-4B-Instruct-2507 单模型实验

使用现有 nlpbase 解释器。各脚本按绝对路径定位数据，无需调整 PyCharm 工作目录。

## 运行顺序

1. `download_model.py`：使用 ModelScope 下载至 `D:/LLM/model/Qwen3-4B-Instruct-2507`。如果同一目录已经在下载，等待原下载完成，不同时启动第二份下载。
2. `check_setup.py`：只加载 tokenizer，检查 C 数据、Prompt、答案 token 和权重文件名单；不加载 4B 权重。
3. `train_lora.py`：正式训练，按每轮 Dev pair-aware W-F1 保存最佳模型。
4. `evaluate_lora_pair.py`：自动读取最佳 Adapter 和训练时保存的 Prompt，输出两种解码成绩。

训练结果：`D:/LLM/train_model/DiCo-NLI/qwen3_4b_instruct_2507/run01/C/best_adapter`。

评分结果：本目录 `outputs/run01/C/`。参数与路径集中在 `config.py` 和 `train_lora.py`。

## 本轮设置

- 复用原 C 构造函数：当前 3042 + 475 条反向 N = 3517 条 Train；Dev 保持 660 条。
- r=16、alpha=32、dropout=0.05、all-linear、lr=1e-4、5 epochs、seed=42。
- 普通 completion-only SFT，监督答案标签及 EOS，不叠加辅助损失、不进行模型集成。
- 旧短 Prompt 作为 user 消息，用当前 2507 Instruct 的 chat template 包装。该模型为非思考版本，不传 enable_thinking，也不添加思考链。
- 训练与预测共用保存的完整模板和本模型 tokenizer；不读取原 0.6B 的全局 tokenizer。
- BF16 LoRA，micro-batch 1、累积 16 次，有效 batch 16。启用梯度检查点、分块全词表 CE；max_length=512，并检查全部输入没有截断。
- Dev loss 的 batch=1；用于选模和正式预测的生成 batch 均为 16。实际 12GB 显存峰值需正式运行后确认；没有自动切换量化或批次大小。
- 读出使用 F/B/E/N 四个候选分数 argmax，另存全词表一步生成、非法生成数及两种读出的差异。原 0.6B 基线 660 条的两种读出已验证一致。
- pair-aware 限制两行组为 FB/BF/EE，单例保留模型预测。

下载未完成的 `.incomplete` 分片不能作为完整权重使用；训练前会检查 index 中列出的每个分片。

数据缓存位于项目根目录 `.hf_q4b`，使用短目录避免 Windows 长路径问题。与 SmolLM2 共用 `experiments/generative_nli.py` 中的模型无关工具，但模型、tokenizer、模板、缓存和结果目录各自独立。

这是完整模型配置的对照：相对 0.6B Base，同时改变了参数量、预训练/指令训练来源、输入聊天模板与 micro-batch 布局，不能将分数差只归因于参数量。

模型来源：https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
