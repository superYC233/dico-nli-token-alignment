# 已有模型的结构分流实验

右键运行 `evaluate_hybrid.py`。只读取已保存的 Dev CSV、组合答案并调用官方评分器，不训练、不运行 GPU 模型。

比较五行：原 Qwen、历史 DeBERTa、Qwen 与 alignment 等权融合、Qwen/DeBERTa 分流、融合 Qwen/DeBERTa 分流。

分流规则固定为：输入文件中同一 pair_id 有两行时用 Qwen 的 pair-aware 结果；只有一行时用 DeBERTa 的原始四类预测。没有将单例直接指定为 N，没有读取金标选择每条样本的专家。

两个 Qwen 的融合在四类概率上等权平均，再比较 FB/BF/EE 三个组合。alignment 使用 `alignment/epoch_4`，不是 `alignment_global`。

输出在 `qwen_lora/outputs/hybrid_decode/run01/`，每个方法分别保存官方预测及评分，同时保留汇总、逐条预测和输入配置。

这是集成与结构元数据实验，不是 Qwen 单模型能力提高。专家分工受到已有 Dev 分析启发，属于探索结果。后续新数据必须先由相同模型产生预测，再按同一规则分流；独立测试结果仍未知，且需要确认测试输入保留相应分组信息。
