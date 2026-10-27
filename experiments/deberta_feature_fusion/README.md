# CLS + Alignment + Lexical特征融合

直接右键运行 **train_deberta_feature_fusion.py**。原实验不修改。

这版从无增强0.930569 Lexical配方开始，只改融合架构，不叠加上一版CNN。

## 具体改动

原来是三个分支分别给四类分数，再相加。

现在：

    CLS表示768维 ── LayerNorm ─────────────┐
    原Token Alignment汇总512维 ── LayerNorm ┤
    原20维Lexical ── Linear/ReLU/LN ── 32维 ┘
                         ↓
                   拼接为1312维
                         ↓
             Linear(1312,128) + ReLU + Dropout
                         ↓
                   Linear(128,4)

只有一个四分类头。不再有CLS、Alignment、Lexical三个独立四分类输出。三个分支都直接接受同一Row CE和Pair CE的梯度；不新增辅助损失。

共同分类头采用正常的非零随机初始化，所以第一批Row CE即可把梯度传入对齐和词汇特征编码器。新的分类头与归一化属于本次融合架构变化，随机轨迹不要求复现旧模型。

Token Alignment的128维投影、cosine/0.1、soft alignment、差分/乘积比较、mean/max pooling保留。不是改成硬对齐，不是完整RE2，不使用CNN。

## 训练设置

- NLI主干：MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli，读取已下载本地缓存。
- 官方Train3042行、Dev660行，真实Pair1282组；不加scope或其他人工、外部数据。
- 从NLI初始化重新训练，不从0.9306/0.938的已选DiCo模型续训。
- 全参数微调，没有冻结或LoRA。
- 每轮先191批Row CE，评估；再81批0.5×Pair CE，再评估。
- 10轮，seed42，FP32，max_length128，Row/Pair/Dev batch均16。
- 主干、对齐、共同分类头2e-5；lexical_encoder使用原词汇分支学习率1e-3。
- Row权重[1,1,1,1.5]；Pair权重[1,1,1.25]，label smoothing0.05。
- 推理仍用FB/BF/EE三组合；单例保留模型自己的四类argmax。

共享分类头不会自动消除三组合Pair CE的自由方向；这次检验的是特征融合，训练目标保持原样。

## 配置与输出

所有配置在config.py。默认RUN_NAME='run01_feature_fusion'，重复实验只需更换RUN_NAME。

模型目录：D:/model/SEAMEVAL/deberta_feature_fusion/run01_feature_fusion/feature_fusion_seed_42/。

结果目录：本目录outputs/run01_feature_fusion/feature_fusion_seed_42/。

只在Pair阶段后的结构F1刷新时保存best_model.pth，始终保留一份最佳完整模型，不保存每轮模型或优化器状态。

- epoch_summary.csv：每轮after_row/after_pair，各自independent/pair_aware。
- best_summary.csv：同一份最佳权重的两项成绩。
- epoch_N/：Pair后的预测及评分；epoch_N/after_row/：Row后的结果。
- diagnostics.json：解码修好/破坏、非N行三类排序，仅诊断，不参与训练或改答案。
- comparison_with_lexical/：最佳权重相对历史0.930569的修好/破坏样本。

训练后右键evaluate_best.py，重新加载本目录的最佳完整权重预测Dev。重载使用保存的融合维度和结构参数。

## 只检查，不训练

右键check_setup.py：核对数据/训练参数、原对齐公式、三个特征分支的梯度、优化器分组、保存重载和历史官方评分。不执行参数更新。

真实GPU检查：

    & 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_feature_fusion/check_setup.py' --gpu

只做真实Row16行、Pair16组的前后向，不执行optimizer.step。显存峰值不含正式训练的Adam状态。

2026-10-25本机检查已通过：数据/训练参数与旧版一致；原对齐公式不变；只有一个四分类头；三个特征分支首批Row/Pair梯度非零；训练循环梯度权重一致；完整state重载与历史评分复现正常。真实NLI模型GPU前后向通过，峰值约7.348 GiB（不含Adam状态）。没有训练或更新参数。

训练入口加--prepare-only则只读取、检查数据，不加载大模型。
