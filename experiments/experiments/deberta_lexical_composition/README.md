# 官方3042行 + Lexical + 比较后CNN组合

直接右键运行 **train_deberta_composition.py**。本次使用NLI版DeBERTa，起点是无增强的0.930569 Lexical配方，不是原始微软base，不加载旧0.938权重续训。

## 本次改动

原来：DeBERTa → 对齐 → token比较 → mean/max pooling。

现在：DeBERTa → 对齐 → token比较 → 两层共享CNN → mean/max pooling。

A/B分别应用同一个CNN，每层清除其他文本、分隔符及padding位置的激活。128维、两层kernel=3，共增加98,560参数。不增加主干前向，没有新增dropout。CNN使用2e-5，原20维Lexical仍使用1e-3。

原Alignment分类头保持零初始化：第一批先让分类输出层产生梯度，后续更新再将监督传入比较层及CNN。这与原版Alignment分支的启动方式相同。CNN的参数初始化使用fork_rng，不推进其他模块和训练的随机序列。

这是ESIM式比较后组合，借鉴RE2的轻量CNN，不是完整复现RE2：

- ESIM：https://aclanthology.org/P17-1152/
- RE2：https://aclanthology.org/P19-1465/
- 作者CNN：https://github.com/alibaba-edu/simple-effective-text-matching-pytorch/blob/master/src/modules/encoder.py

## 数据与训练

- 官方Train3042行、Dev660行；F881/B881/E802/N478。
- 1282个真实Pair；没有scope、768行、MNLI、PAWS等额外训练数据。
- 每轮先191批Row CE，评估；再81批0.5×三组合Pair CE，再评估。
- 10轮，seed42，FP32，最大长度128，Row/Pair/Dev batch均16。
- 主干及头部2e-5、Lexical1e-3、weight decay0.01。
- Row权重[1,1,1,1.5]；Pair权重[1,1,1.25]，smoothing0.05。
- 原FB/BF/EE解码；N单例仍由模型自己预测。

每次只在Pair阶段后的结构F1刷新纪录时替换best_model.pth；不保存每轮模型，也不保存优化器。每轮两个阶段的预测和CSV都会保留。

## 改哪里

所有运行设置在config.py。默认RUN_NAME='run01_composition'，以后重复实验改RUN_NAME即可。

模型目录：D:/model/SEAMEVAL/deberta_lexical_composition/run01_composition/composition_seed_42/

结果目录：本目录outputs/run01_composition/composition_seed_42/。

epoch_summary.csv包含epoch、stage、method。stage为after_row或after_pair；每轮四行。best_summary.csv的independent与pair_aware来自同一个最佳权重，不能把两个阶段的峰值拼在一起。

每轮epoch_N/是Pair后的结果；epoch_N/after_row/是Row后的结果。diagnostics.json额外记录结构解码修好/破坏数量和金标非N行的三类排序，仅用于诊断，不改变预测。

训练结束会与历史0.930569 Lexical最佳预测逐条比较，输出修好/破坏的样本。该比较仅在预测后进行，历史预测不进入训练。

训练后右键运行evaluate_best.py，读取best_model.pth重新预测。保存了全部CNN、Lexical、原分类头、Alignment及主干权重。

## 只检查，不训练

右键运行check_setup.py：检查数据、mask隔离、梯度通路、无CNN时与原模型等价、官方评分及保存加载。默认不加载真实大权重，不进行参数更新。

真实GPU前后向检查：

    & 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_lexical_composition/check_setup.py' --gpu

不进行optimizer.step，不启动训练。显存峰值不含Adam更新状态。

2026-10-25本机检查已通过：配置/数据与旧版一致、CNN mask隔离、关闭CNN后的Row/Pair梯度及RNG等价、CNN梯度通路、完整保存重载、历史660行官方评分复现。真实NLI模型的Row16行及Pair16组GPU前后向通过，峰值约7.355 GiB（不含Adam状态）。没有执行参数更新。

只核对官方数据：训练入口加--prepare-only。

如需同代码的原Lexical对照，训练和重新评估时加--no-composition，或在config.py设置USE_COMPOSITION=False。关闭后不创建CNN，原模型参数名、计算与随机序列保持一致；不需要先跑这个对照才能运行默认CNN版本。
