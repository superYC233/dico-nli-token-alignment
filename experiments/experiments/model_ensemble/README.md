# 已有模型的多模型预测比较

结果入口：outputs/run01_all_existing/RESULTS.md。

- members.csv / members.json：各模型来源、已选版本、已有权重路径和单模型成绩。
- aligned_scores.npz：对齐后分数快照。此文件保存的是预测分数，不是模型权重。
- all_combinations.csv：实际计算过的全部融合方案。
- recommended_combinations.csv：代表方案的同一配方independent和pair-aware成绩。
- best_per_model_count.csv：按模型数量比较。
- selected_recipes.json：固定成员、分数来源、已有权重路径、融合方式。
- Sxx_predictions.csv：各代表方案逐行分数和预测。
- official_checks：官方评分器复核。
- conditional_group_bootstrap.csv：相对0.938的条件分组bootstrap，不校正Dev搜索选择偏差。

普通平均：每个模型四类softmax概率等权平均。几何平均：log-prob等权平均。
分层融合：N概率来自固定模型，F/B/E条件概率单独平均；实际模型数包含N门。
所有方案均不训练、不运行真实模型、不强制单例N，不用金标逐行挑模型。
每个原始运行只选择匹配其best_summary的一份四类分数，相同分数去重。
只读取官方同一660行Dev；分数按instance_id对齐并核对已有文本/标签/组号。

复算脚本：analyze_all_models.py。先更改RUN_NAME以保存新的独立结果。
1—3个成员穷举；更大组合用beam扩展，不宣称全局最优。
report_results.py复核当前快照并整理报告。

本次使用大量Dev组合搜索，结果是探索性上界，不能当作未见测试集成绩。
各成员训练所用增强数据、标注、外部资源仍须在实际使用该组合时逐项披露。
