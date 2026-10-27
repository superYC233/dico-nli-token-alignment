# Qwen C+r16 + 范围增强

右键运行 `train_lora_aug.py`。训练由用户执行；原 qwen_lora 代码不修改。

训练数据是原 Train 3042 + 当前 C 构造的475条反向N + 与DeBERTa完全相同的200条范围增强，
共3717条。**先做C，再加增强**，因此新增40条N不再额外反转。
每轮233步，5轮1165步；原C为每轮220步，5轮1100步，训练更新数有所增加。

训练沿用 Qwen3-0.6B-Base、LoRA r16/alpha32/dropout0.05/all-linear、lr1e-4、有效batch16、
5轮bf16、completion-only全词表SFT、包含答案和EOS监督、packing=False、seed42。
使用原版旧短Prompt，不套chat template。没有增加Pair CE、词汇分支或新的辅助损失。

每轮固定batch16一步贪婪生成Dev，报告independent与pair-aware。
pair-aware仍比较FB/BF/EE三组合和，单例保留原预测。
按pair-aware W-F1选模型，保留1个最佳checkpoint，结束后导出同一模型的best_adapter。
保存Prompt模板，独立评估时读取保存的模板，避免改Prompt后误评旧Adapter。

配置集中在 `train_lora_aug.py` 顶部：

```python
RUN_NAME = 'run02_c_scope'
ADD_SCOPE_AUGMENTATION = True
```

如要同代码复跑无增强C对照，换新RUN_NAME并将ADD_SCOPE_AUGMENTATION设为False。
这只关闭新增200条范围增强，保留C的反向N。

模型：`D:/model/SEAMEVAL/qwen_scope_aug/run02_c_scope/best_adapter`。
结果：本目录 `outputs/run02_c_scope`，每轮保存预测/四标签分数、混淆矩阵、两种解码成绩，
汇总在 `epoch_summary.csv`；`decoding_changes.csv` 记录解码修好/破坏条数。

跑完可直接运行 `evaluate_lora_aug.py` 再评估best_adapter，结果在 `best_evaluation`。
本版只有标准SFT，没有DeBERTa的Row/Pair两个训练阶段，因此不做那种阶段对比。

只准备数据、不训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe experiments/qwen_scope_aug/train_lora_aug.py --prepare-only
```

增强标签尚未独立审核；Dev参与开发和选模，不能把最高Dev分数当作独立Test证据。

`run01_c_scope`在第二轮评估时因F/E并列最高分被过严校验中断。
现已改成检查“生成标签是否属于最高分集合”，保留生成答案，不强制采用四候选的列序argmax。
`label_score_check.json`记录并列数和列序argmax差异数。默认换run02重新运行，旧模型/预测保留。

运行 `analyze_results.py` 可仅用已有CSV对比原C+r16，不加载模型、不训练。
结果保存在 `outputs/run02_c_scope/comparison_with_c_r16`，包括逐条修好/破坏、混淆矩阵、
383个pair_id整组bootstrap 20000次的区间和逐轮四候选条件置信度。
区间只覆盖固定已选模型在当前Dev上的抽样，不覆盖训练种子、选模或反复使用Dev的影响。
