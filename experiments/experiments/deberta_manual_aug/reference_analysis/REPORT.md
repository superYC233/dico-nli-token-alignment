# 0.935493版本错误复查

参考：G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_original_aug/outputs/run01_original_scope/epoch_3/predictions_with_scores.csv

同一checkpoint：独立W-F1=0.480876，pair-aware W-F1=0.935493。
结构后错误43行，涉及25组。

| 错误 | 行数 | 组数 |
|---|---:|---:|
| E_to_FB | 12 | 6 |
| FB_to_E | 20 | 10 |
| F_B_reversal | 4 | 2 |
| N_to_E | 6 | 6 |
| N_to_FB | 1 | 1 |

## 本次增强针对什么

主要错误池是E与F/B之间的混淆：32行、16组。不是所有错误都能归因于模型不懂词义。
新增训练草稿使用新词汇练习：属性限制、严格上下位、等价表达，以及共享词但属性或对象不同的N。
没有将Dev中的错误文本改标签后放入Train，也没有替换几个Dev词就保留原句训练。

## 标签与语境问题

Saudi Women/Saudis的方向、black top与big necklace的关系、senator/lawmaker的N标签等，
按字面集合解释存在疑问。其他若干新闻片段及介词关系需要原上下文。
以上均是复核线索，不是已确认的官方标注错误；本次不修改任何官方标签。
不能承诺用一般语义更一致的人工样本就能修复所有官方Dev错误。

## 开发边界

Dev错误分析参与了增强方向选择，因此本次是Dev指导的开发实验。
新增样本即使与Dev词汇零交集，也不能恢复一个从未使用过的测试集。
生成/过滤脚本只把Dev文本用作拒绝黑名单；分析文件不进入模型或训练输入。
新增标签为助手手工设计草稿，尚无独立人工标注者审核。

完整25组及逐组复核意见见error_groups_for_review.csv。
