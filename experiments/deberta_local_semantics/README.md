# 原0.938 + 局部语义关系监督

直接右键运行 `train_deberta_local_semantics.py`。所有正式训练由你启动；旧模型、旧代码、原数据均不修改。

**当前设置**

| 项目 | 设置 |
| --- | --- |
| 整体训练数据 | 原0.938当次3290行，哈希核对一致 |
| 基础模型 | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli |
| 全局模型 | 原CLS + Token Alignment；全参数训练，无Lexical、无LoRA |
| 新监督 | 64条既有官方Train行的局部短语关系，F/B/E/N各16 |
| 局部诊断 | 另16条Train行，只留出局部标签；原Row CE仍会看到这些输入 |
| Row阶段 | 原加权四分类CE + 0.2 × 当前批标注行的局部四分类CE |
| Pair阶段 | 原0.5 × 三组合Pair CE；不调用局部头 |
| 学习率 | 原模型2e-5；新局部头3e-4 |
| batch / epochs / 精度 | Row16、Pair16组、10轮、FP32，沿用原版 |
| 预测 | 原四分类输出及原pair-aware；单例仍由模型预测 |
| 保存 | 只保留结构F1最高的一份best_model.pth，同时记录该轮独立分 |

每轮仍为206个Row步骤 + 87个Pair步骤。每条全局Train行仅在Row阶段出现一次；64条局部标注不额外复制、不额外抽样。含局部标注的批次复用同一次DeBERTa前向。辅助头不参与正式预测，因此不增加推理计算。

这只是借鉴NS-NLI的局部关系监督思路，不是完整NS-NLI复现；没有七关系自然逻辑证明器或强化学习，不保证提升到0.95。原Pair-only阶段仍可能影响独立判断，本次先单独测新语义监督的贡献。

**文件怎么读**

- `config.py`：RUN_NAME和训练参数。后续重复实验只改RUN_NAME。
- `annotation_sources.csv`：人工选择的50个官方Train源组、短语、局部关系、理由及读法。其中40组展开为64条辅助训练标注；10组展开为16条诊断标注。
- `prepared/local_train_annotations.csv`：实际64条标注，含原文本、字符位置、局部关系与全局金标。
- `prepared/local_diagnostic_annotations.csv`：16条局部标签留出诊断。
- `prepare_data.py`：复用原3290行、从官方Train定位这些标注，不生成新训练文本。
- `data.py`：将人工短语的字符位置映射到正确的Text 1/Text 2 token。
- `model.py`：保留原模型计算；在128维token投影上增加小型局部关系头。
- `train_deberta_local_semantics.py`：先完整Row+局部监督，再完整Pair。
- `evaluate_best.py`：训练结束后重新评估唯一最佳模型。
- `check_setup.py`：数据、原模型一致性、梯度、重载、官方评分检查；不训练。

例如已有Train的 `on a blue sofa / on a sofa`：监督 `blue sofa / sofa` 为F；反向官方行监督为B。给模型指出比较的位置及其关系，不把整行标签无条件广播到所有token。

理由和读法栏用于人工审阅，不作为额外模型输入。实际新增信号是短语位置与局部关系标签。

局部标注由助手按明确读法填写，未经独立标注者仲裁。词义、角色和范围的关系不能只由程序证明，可直接查看理由栏。所有实例来自官方Train，不依赖Dev模型对错来选；Dev仅提供完整文本对的拒绝黑名单，不读取Dev标签来制定局部标签。方法方向受之前Dev分析启发，需在论文中披露。

没有增加新文本，因此不应宣称“新增样本与Dev逐词零交集”；原Train和Dev已有词汇交集保持原样。16条诊断输入仍在普通Train里，成绩只衡量局部标签留出，不能当成文本未见的泛化结果。隐藏Test没有核验。

审计中发现旧Train的 `driving down / driving` 也出现在Dev。本次已将该组从局部标注源中换掉，64+16条局部标注与Dev的规范化完整文本对交集为0；历史3290行整体输入按实验约定原样保留，不宣称原始数据本身完全无重复。

**运行**

右键先运行 `check_setup.py`，再运行 `train_deberta_local_semantics.py`。

只准备数据：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_local_semantics/prepare_data.py'
```

完整训练：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_local_semantics/train_deberta_local_semantics.py'
```

零辅助损失对照可用同一文件，不需要新文件夹：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_local_semantics/train_deberta_local_semantics.py' --run-name run02_local_weight0 --local-weight 0
```

零权重时局部头不执行，原计算路径和随机数状态保持一致。新初始化层的参数不消耗原训练随机数。

默认模型目录：`D:/model/SEAMEVAL/deberta_local_semantics/run01_local_semantics/`。

默认结果目录：`experiments/deberta_local_semantics/outputs/run01_local_semantics/`。

每轮保存Row后和Pair后的预测/混淆矩阵及局部诊断，只保存一份最佳模型。`epoch_summary.csv` 能检查局部关系有没有学好、Pair阶段是否损坏它，以及同一轮的independent和pair-aware是否共同改善。

**已完成检查**

数据/token映射、零权重的原参数/预测/Row梯度与随机数状态、一次前向复用、局部梯度到主干、Pair阶段不使用局部头、保存重载和历史660行官方评分复现均通过。真实DeBERTa的Row16行加局部损失、Pair16组前后向也通过，未执行optimizer.step。检查峰值约7.347GiB，不含正式训练才会创建的Adam状态；完整结果在 `outputs/_checks/gpu_check.json`。
