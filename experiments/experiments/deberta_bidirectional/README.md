双向联合四分类
================

直接运行 `train_deberta_bidirectional.py`。无需重新下载模型，不加载旧实验权重续训。
后续同代码实验只改训练入口顶部的 `RUN_NAME`、`SEED` 等参数。

这版做了什么
------------

同一套DeBERTa+LoRA读取(A,B)和(B,A)，将两个CLS表征投影到128维。
联合头先比较两方向的和、绝对差、逐元素乘积，再输出F/B/E/N：

    s = [h_AB + h_BA, |h_AB - h_BA|, h_AB * h_BA]
    common, E, N = symmetric_head(s)
    direction = dot(direction_weight(s), h_AB - h_BA) / sqrt(128)
    logits = [common + direction, common - direction, E, N]
    loss = CrossEntropy(logits, 当前原行标签)

交换输入时，F/B分数交换，E/N不变。推理关闭dropout时成立。
模型仍然可能判断错误；交换一致是结构保证，不是语义准确率的保证。
若F/B最大分数完全相同，argmax的平局选择也不能作为方向正确的保证。

没有Pair CE，不交替Row和Pair阶段，不把两个二元概率相乘。
也没有使用旧Token Alignment残差分支；这版的交互发生在两方向表征之间。
采用同一NLI编码器初始化，但与最近Row版本一样，不沿用原三类NLI pooler/classifier。
它是一个模型、一套权重；单条任意输入也会自行构造反向文本，不需要另一条记录。

训练数据和参数
--------------

- 固定读取0.938实验留下的3290行训练快照：官方3042 + scope200 + relation48。
- 不生成新增强、不把N反向视图另外计成一条训练金标，原始行各计算一次CE。
- 相同无序文本对复用两次编码；没有删掉重复行或冲突标签的监督。
- 分组只用text1/text2，不用pair_id、标签、是否单例。原Train里的F/N冲突仍保留。
- DeBERTa-v3-base NLI，LoRA r16/alpha32/dropout0.1，只训练Q/V LoRA和联合头。
- seed7，lr1e-4，20轮，float32，AdamW weight_decay0.01。
- 四类权重仍为[1,1,1,1.5]；每批8个文本对，编码器输入16个视图。
- max_length128，本批裁去无效PAD尾部，长度补到8的倍数。

预测与选模
----------

每轮输出independent及原三组合pair-aware，继续使用项目官方评分器。
独立预测不读取pair_id；pair-aware只在评分阶段读取分组，仍禁止配对行输出N。
本结构已经保证两个顺序的输出交换。因此一般情况下，pair-aware额外作用主要是
排除配对行上的N，不应把这部分增益当成模型自己识别了N/非N。

沿用同轮双目标选模：

    selection_score = min(independent_f1 / 0.87, pair_aware_f1 / 0.94)

只在刷新时覆盖同一个best_model.pth，不保存每轮权重。
这两个目标数字只是选模比例，不是训练损失，也不是承诺达到的成绩。
每轮CSV都保存；训练结束自动重载最佳模型，分别评估Dev和Train。

文件位置
--------

- 模型：`D:/model/SEAMEVAL/deberta_bidirectional/<RUN_NAME>/best_model.pth`
- 记录：当前目录的`outputs/<RUN_NAME>/`
- `epoch_summary.csv`：每轮两种F1、分类正确数、非N→N和训练耗时。
- `best_summary.csv`及`best_metrics.json`：同一最佳轮次的结果。
- `best/dev/`、`best/train/`：最终最佳模型的逐行预测与混淆矩阵。
- 单独重新评估：`evaluate_deberta_bidirectional.py`。
- 只检查数据：训练入口添加`--prepare-only`。
- CPU无更新检查：`check_bidirectional.py`；加`--real`检查本机真实NLI底座。

保存的是LoRA和联合头，恢复时需要本机缓存中相同revision的底座。
不能用以前普通DeBERTa或Qwen的加载函数读取这版模型。

计算量
------

单条独立调用需要两次编码。整份数据按文本复用反向视图，避免把已有F/B、E/E
的两行各重复编码两遍；实际视图数由运行时打印。所有N也做两向编码。
动态裁剪PAD用于减少短语任务计算，但不保证不同硬件上的训练耗时。

本机检查记录（2026-10-24）
-------------------------

已运行`check_bidirectional.py --real`和训练入口的`--prepare-only`，均通过。
检查没有进行任何优化器参数更新，也没有创建正式模型目录。

- 当前3290行对应1883个不同无序文本对，每轮3766个编码视图。
- Dev660行对应381个不同无序文本对，预测时762个编码视图。
- 验证了原行标签数量不变，包括同文异标的多条监督。
- 验证了F/B交换、E/N不变、移除pair_id/金标后预测不变。
- 验证了非N行对N logit有正CE梯度，N行对N logit有负CE梯度。
- 验证了冻结范围、LoRA梯度、非零LoRA权重保存和重载。
- 已用真实缓存中的NLI DeBERTa检查单条预测与专用加载器重载。
- 已走通项目官方评分器，生成两种方法的报告和混淆矩阵。

这些是实现检查，不是训练成绩；完整训练由你启动。

已修复首次正式启动时源码哈希误指向`evaluation_functions.py`的问题，评分器实际是
`evaluation_functions/`包。现在在加载模型之前检查这些路径；只有tokenizer和数据快照、
尚无训练配置或权重的启动失败目录，可以使用原RUN_NAME直接重跑。
