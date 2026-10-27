# Qwen2.5-3B-Instruct：C + reason

右键运行 `train_lora_reason.py`。训练后运行 `evaluate_lora_reason.py`。

复用原C的3517行标签数据，2040行supported解释作为独立生成任务。
损失为标签平均CE + 0.2 × 解释平均CE。两个任务均使用本地Instruct聊天模板，预测不输入解释。
LoRA r16 / alpha32 / dropout0.05 / all-linear，lr=1e-4，5轮。
默认微批次4 × 累积4，有效batch16；开启梯度检查点，适配12GB显存。
每轮报告independent、pair-aware；按pair-aware选最佳并导出best_adapter。

基础模型：`D:/LLM/model/Qwen2.5-3B-Instruct`。
模型输出：`D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_reason/run01_c_reason02/C/best_adapter`。
结果保存在本目录outputs，不覆盖0.6B实验。

可选参数：`--prepare-only`只准备数据；`--baseline`跳过解释损失并自动添加baseline运行后缀。
后续实验只修改config.py中的RUN_NAME和参数。

本版为Instruct，0.6B原版为Base，因此跨模型比较同时改变了初始化和聊天格式。
