**两个Row CE架构对照**

按顺序右键运行：

1. train_relation_cls_row.py：run22_cls_row_seed7，DeBERTa+LoRA+CLS，只有四类Row CE。
2. train_relation_alignment_row.py：run23_alignment_row_seed7，DeBERTa+LoRA+CLS+原Token Alignment，只有四类Row CE。

这是从相同NLI初始化重新训练的两个模型，不是加载0.938权重，也不是训练后才删除某个分支。两个入口共享train_relation_row.py的全部训练参数。CLS版真正删除Alignment模块，不计算、不优化、不保存它。

**相同设置**

官方3042+旧scope200+relation48，共3290行，直接读取run06保存的快照，没有新增文本或改标签。seed7，r16，alpha32，LoRA dropout0.1，query_proj/value_proj；原编码器冻结，LoRA和保留的任务头正常训练。

固定lr=1e-4、20轮、float32、weight_decay=0.01，Row权重[1,1,1,1.5]，group batch8，Dev batch16，max_length128。没有学习率衰减、Pair CE、Pair标签平滑、词汇分支或额外监督。这样可以和已有run19_lora_r16_seed7_dropout01比较，而不把未验证的run21调度一起混入架构消融。

仍用与run19相同的GroupDataset和独立shuffle generator；每轮1910组、239步，3290行各出现一次。分组仅为了保持批次组成一致；pair_id、组大小和Pair标签不输入模型，也不计算Pair CE。N单例照常参与四分类。

两组共享的LoRA和CLS初值哈希保存在run_config.json的shared_initialization_sha256中，冻结NLI底座固定同一base_revision。两组初值和数据shuffle相同；Alignment训练中的额外dropout会消耗随机数，不承诺整条训练轨迹的dropout掩码完全一样。

**每轮与选模**

每轮仍分别报告independent与pair-aware。pair-aware只在推理阶段使用官方分组约束，没有进入训练损失。

两组与run19采用相同规则：最大化同一轮min(independent/0.87, pair_aware/0.94)，相同取早轮。只刷新一份best_model.pth，保存LoRA+当前保留的任务头。训练结束自动恢复这份权重，分别评估Dev与Train，并核对重载后的预测与当轮一致。

每组模型保存在D:/model/SEAMEVAL/deberta_manual_aug/<RUN_NAME>/，结果保存在当前目录outputs/<RUN_NAME>/。

- epoch_summary.csv：完整曲线、四类正确数、非N→N和训练耗时。
- batch_plan.csv：每批原Train行号，可比较两组数据顺序。
- best_summary.csv：选中同一轮的两种成绩。
- best/dev与best/train：最佳权重重新评估，不能混用Train与Dev成绩。
- architecture_comparison.csv：本组完成后，将已完成的run22、run23和旧run19汇总；第二组完成后可直接看该表。

若修改入口RUN_NAME，共享脚本中的CLS_RUN/ALIGNMENT_RUN也需同步修改，才能自动汇总对应的两组。修改训练超参数统一在train_relation_row.py进行；不要两组分别改。

训练结束已经自动评估。需要重新加载时用evaluate_relation_row.py，它按保存的head_type恢复正确架构，不要用旧evaluate_relation_lora.py加载CLS专用权重。

**如何解读**

- run22 vs run23：检查在同一Row CE目标下增加Alignment的作用。
- run23 vs run19：检查同一架构加入Pair CE的作用；新代码一次反向与历史两次反向数学目标等价，但浮点累加顺序可能不同。
- 两组不能直接代替0.938版本的全参数两阶段消融；适配方式、更新预算和选模规则不同。
- 这两组是定位问题的消融，不承诺F1上涨。训练由你启动。

只检查数据，不训练：两个训练入口分别加--prepare-only。CPU结构与保存检查：运行check_relation_row.py。
