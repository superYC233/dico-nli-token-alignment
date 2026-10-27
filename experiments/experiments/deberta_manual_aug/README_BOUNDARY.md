# 修订增强：保持0.935配方，只换120行对照

右键运行同目录 `train_boundary_manual.py`。默认 `RUN_NAME='run04_boundary_120'`、`AUGMENT_MODE='manual'`。

本轮已经完成数据准备和CPU检查，没有加载预训练权重或启动训练；训练由你运行。

## 为什么缩到120行

上一批事件200行的最佳pair-aware W-F1为0.924949；相对0.935493修好14行、破坏21行。
其标签可由长度、词序和两组固定状态词全部答对。继续扩同样的模板不能保证改善。

现在先减少重复，改成72个明确列出的源对照：24个单向包含（F/B各12）、24个E、24个N。
F/E补反向后实际120行：F24、B24、E48、N24。类别比例仍为20%/20%/40%/20%。
不要求每个词汇家族都套同一模板，也不承诺120是最优规模。

## 本次具体补什么

| 对照 | 例子 | 要避免的捷径 |
|---|---|---|
| 同词数、不同谓词的包含 | frying dumplings → cooking dumplings | 只删宾语或按长度判断F/B |
| 不同长度的定义性等价 | biologists ↔ biological scientists | 更长就一定多信息 |
| 名称与保留属性的展开 | experienced biologists ↔ experienced biological scientists | 把定义展开误当范围限制 |
| 更长反而更宽的类别 | biologists → biologists or musicians | 长侧一定蕴含短侧 |
| 增加子类而不增加信息 | biologists ↔ biologists or entomologists | 一加词就改变类别 |
| 数量等价但对象可不同 | twelve flasks ↔ 1 dozen flasks；1 dozen flasks / twelve goblets=N | 数字相同就等价 |
| 同一词集合但角色交换 | doctors interviewing actors / actors interviewing doctors=N | 相同词集合就答E |
| 同样3词的词序变化仍等价 | patiently tutoring apprentices ↔ tutoring apprentices patiently | 换词序就答N |
| 事件相关而不必然互推 | submitted manuscripts / published manuscripts=N | 通常流程就是逻辑保证 |
| 类型不同但词面相近 | dental surgeons / dental clinics=N | 共享修饰词就有蕴含 |
| 上下位与独立属性组合 | certified musicians / experienced violinists=N | 只要有上下位词就一定蕴含 |
| 介词保持或改变空间关系 | beneath / under为同义空间义；beneath / above为N | 看见介词变化就统一判断 |

这些是助手手工设计、逐条附理由的合成草稿，没有独立人工标注者复核。
数据不从Dev逐条替换词汇生成；Dev错误分析指导宽泛方向，Dev文本继续仅用作拒绝黑名单。

## 数据与训练

原官方Train3042＋保留旧scope200＋新对照120＝3362行。

**不加入旧80行，也不加入上一批事件200行。**

使用原0.935模型结构和参数：同一DeBERTa NLI初始化、Token Alignment、全参数、无Lexical。
仍是每轮完整Row CE，再完整0.5×三组合Pair CE；lr=2e-5、batch16、seed42、10轮。
Row权重[1,1,1,1.5]，Pair权重[1,1,1.25]与label smoothing 0.05不变。
每轮211个Row步骤、89个Pair步骤。按Dev pair-aware W-F1选最好，只保留一个最佳模型；同时报告独立预测。
不是加载旧的Dev最佳权重续训，而是从同一NLI预训练初始化重新训练。

## 已检查的内容

- 120新增行和3362总行；反向F/B、E/E标签正确，N只作为单例。
- 新增Pair CE目标为FB12、BF12、EE24；不让全部新增方向组都朝FB一个方向。
- 新文本的完整词、Porter词干、整短语及正反向文本对与Dev交集均为0，不删除停用词。
- 旧3242行训练快照原样保留；新实例ID不与它冲突。
- 源表明确写出的F/E关系链，没有与源表N或严格F方向发生图上的矛盾。
- F/B中有28行两边同词数；有5个F源对照左侧比右侧短。
- E中有26行两边词数不同；同为3词且同词集合的情况既有E，也有N。
- 上次的表面规则由200/200降到31/120。这只是排除一条已知捷径，不证明所有捷径已消除或必然涨分。

样本质量仍须结合新Dev结果与独立诊断判断。不能把自建探针经pair-aware后全对当作独立语义能力；现有训练末尾的旧scope探针保留为辅助诊断，未经独立审核的建议标签不当正式准确率。

对疑似依赖原图／新闻上下文的金标、对职务方向的疑点，不照着Dev硬编新规则。
最高级只补确定的类别包含及“评级不等于认证”；不为追某条Dev，强行把所有最高级／绝对程度组合打成N。
没有把裸数字与对象名词的关系强制编成N；先补数量同义和明确的对象变化。

## 文件

- `boundary_examples.csv`：72个源对照，可直接打开看文本和每条标签理由。
- `prepare_boundary_data.py`：复用原过滤、反向展开和复制对照；另写入 `prepared_boundary/`。
- `prepared_boundary/manual_rows.csv`：实际新增120行。
- `prepared_boundary/training_data_manual.csv`：完整3362行。
- `prepared_boundary/design_audit.csv`：逐行词数、词集合及旧表面规则检查。
- `prepared_boundary/overlap_report.json`：隔离结果、来源、哈希与限制说明。
- `check_boundary_data.py`：CPU检查；不加载模型，不训练。

模型：`D:/model/SEAMEVAL/deberta_manual_aug/run04_boundary_120/best_model.pth`。
结果：本目录 `outputs/run04_boundary_120/`。

后续等预算复制对照不需要新目录结构或改模型代码；仍用本入口，换一个RUN_NAME并设 `AUGMENT_MODE='copy'`。
该控制会复制120行已有Train，保持新增类别、组数和训练步数预算一致。上次复制80行的run不能当这一轮的匹配对照。

新120行的词汇隔离不代表旧Train或旧scope也零交集；旧部分分别审计。隐藏Test未提供，不声称查过Test。
本任务已经使用Dev开发方法并选模，新样本词汇零交集不能使Dev重新成为独立测试。

## run04完成后的测量对照

run04最佳第1轮pair-aware W-F1为0.907866，未超过参考版0.935493。
`outputs/run04_boundary_120/measurement_review/REPORT.md` 已对齐两版完整10轮曲线、
同一200行探针的既有结果、当次b_f20实际B标签、完整训练集E比例及逐条修好/破坏。
`curves.png` 可直接看原版也存在的类别波动；没有为这次对照重跑模型。
先保留这个负结果，不修改当次数据或标签。下一步若做复制120对照，继续用本入口，只换RUN_NAME和AUGMENT_MODE。
