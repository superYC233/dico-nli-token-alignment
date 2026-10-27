原0.938配方 + SiFT式对抗微调。正式训练由用户运行。

直接右键运行`train_original_staged_sift.py`。
训练结束自动重载最佳模型，评估Dev和Train。以后独立重载用`evaluate_original_staged_sift.py`。
旧模型、原训练入口、PAWS和replay文件均未修改。

每轮流程：

1. 原3290行、batch16、206次Row更新：`loss = 四类Row CE + 1.0 * SiFT正则`。
2. 评估Dev的independent和pair-aware。
3. 原1380个真实组、batch16组、87次Pair更新：`loss = 0.5 * Pair CE`。
4. 再评估Dev，按Pair结束后的pair-aware W-F1选最佳，只保留一个完整权重。

Pair仍比较FB/BF/EE，类别权重[1,1,1.25]、平滑0.05；Row类别权重[1,1,1,1.5]。
模型仍为NLI DeBERTa + 原CLS四分类 + Token Alignment四分类残差，全参数184001800。
同一NLI初始化重新训练，不读取0.938已训练权重续训。
Train沿用原0.938的3290行：官方3042 + scope200 + relation48；Dev660行不加入训练。
seed42、lr2e-5、10轮、float32、动态padding、非reentrant梯度检查点。
没有PAWS、新人工样本、额外N回放、NN组合或联合头。
预测不加扰动，仍按原三组合pair-aware，单例保留模型四分类argmax，不统一设成N。

SiFT做什么：

- 在embedding的原LayerNorm之前加一个无可训练参数的LayerNorm归一化，与微软SiFT的做法一致。
- 当前干净输入先产生四类分数和Row CE。
- 从缓存的归一化embedding出发，初始化小扰动，搜索一次使四类分布偏离的方向。
- 正式损失让干净/扰动输出相互约束，再与Row CE合并，只做一次优化器更新。
- 搜索delta时临时关闭模型参数梯度，结束恢复；正式反向仍更新全部模型参数。这不是冻结编码器训练。
- 与官方代码一致，扰动支路缓存输入detach；embedding参数通过干净分支接收梯度，encoder和任务头也接收扰动分支梯度。
- 本适配让三次前向复用同一dropout随机状态，屏蔽padding扰动，用独立随机数发生器产生噪声。
- 日志扣除了官方双向stop-target交叉熵中的无梯度熵常数，正则数值是KL，梯度与该官方目标一致。
- 不是完整SMART算法，不包含它的Bregman近端优化；也不是R-Drop、FGM金标对抗CE或多步PGD。

参数起点：SIFT_WEIGHT=1.0、SIFT_INIT_PERTURBATION=1e-2、SIFT_STEP_SIZE=1e-4，一次上升。
后两项参考微软实现默认值，权重1.0是本次起点，未根据Dev搜索，不保证提高F1。
小扰动是向量扰动，不把两个不同的词当同义替换，也不提供新增词义知识。

归一化必须在Pair、评估和重载时保留；只有对抗噪声限于Row训练。
因此请使用本文件配套加载器，不要用旧的原模型加载器遗漏归一化hook。
归一化没有新增权重，但改变了数值处理，不能称为与原0.938完全相同的前向。
SIFT_WEIGHT=0时完全不注册该hook，恢复同代码的普通Row→Pair目标；做此对照需换RUN_NAME。
动态padding改变dropout消耗，不保证与历史原0.938逐位复现；零权重对照提供相同省显存设置。

默认RUN_NAME：run06_original_staged_sift_seed42。

- 模型：D:/model/SEAMEVAL/deberta_bidirectional/<RUN_NAME>/best_model.pth。
- 结果：本目录outputs/<RUN_NAME>/。
- sift_batches.csv：每个Row批次的CE、SiFT原值/加权值、搜索KL、delta最大值和delta梯度范数。
- stage_summary.csv：每轮Row后和Pair后，两种读出的四行结果。
- epoch_summary.csv：只列Pair后两种读出。
- stage_changes.csv：Pair阶段修好/破坏及各类正确数变化。
- best_metrics.json、best_summary.csv：同一份最佳模型的指标。
- best/dev、best/train：重载最佳模型后的完整预测和评分。
- reload_check.json：重载前后两种预测差异，应为0。

只准备和检查路径、不训练：`train_original_staged_sift.py --prepare-only`。
CPU机制检查：`check_original_staged_sift.py`；真实CPU模型加`--real`，真实GPU单批加`--gpu`。
检查脚本不执行optimizer.step，临时完整权重保存在本项目artifacts下，完成后清理。
GPU单批检查另分配与AdamW两份动量等量的缓冲，不包含优化器step的临时内存，不代表完整训练峰值。

原论文：https://arxiv.org/abs/2006.03654
微软实现：https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/sift/sift.py
微软示例：https://github.com/microsoft/DeBERTa/tree/master/DeBERTa/sift
源代码为MIT，出处与版权信息保留在sift_regularizer.py和SIFT_LICENSE.txt。

2026-10-24已完成检查：

- prepare-only读取真实3290/660行、1380个Pair、206+87步，通过且未创建正式run06训练目录。
- 零权重与旧Row CE逐参数梯度一致；原三组合Pair梯度一致。
- 归一化后零扰动且相同dropout时正则为0，非零扰动的搜索梯度有效。
- delta搜索不污染模型grad，参数冻结状态恢复正确；非reentrant checkpoint下正式反向正常。
- 中心化KL与官方stop-target双向交叉熵的梯度一致；小模型及真实模型完整保存/重载输出一致。
- RTX 5070 Ti Laptop GPU单批16行×26 token，包含与AdamW两份动量等量的缓冲；峰值allocated约2.770 GiB、reserved约3.047 GiB。
- 该批SiFT正则约1.3874e-5，非零且有限。这是机制检查值，不是正式训练结果或建议据此调整权重。
- 四个Python文件语法检查通过。检查没有执行优化器参数更新，没有启动正式训练，临时权重已清理。
