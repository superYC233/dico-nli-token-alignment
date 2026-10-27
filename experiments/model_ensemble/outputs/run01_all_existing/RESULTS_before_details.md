# 现有模型多模型预测比较

全部结果来自已有Dev CSV的离线计算；没有训练、加载真实模型或强制单例判N。
共有97份去重后的、与官方660行逐一对齐的四类分数。每个运行只使用匹配其best_summary的一个版本。
1—3个成员穷举；4—6个方向成员采用beam扩展，不保证全局最优。分层方式额外计入N门，实际成员数见表。
模型路径存在只证明文件存在，本次没有加载权重复核。当前ModernBERT结果是盘点时已保存版本。

| 编号 | 方案 | 模型数 | independent F1 | pair-aware F1 | N正确 | FBE正确 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S01 | 普通平均：最高结构分 | 6 | 0.773545 | 0.959401 | 99 | 534 |
| S02 | 普通平均：最高独立分 | 4 | 0.888548 | 0.927560 | 81 | 532 |
| S03 | 普通平均：独立≥0.87且结构≥0.94 | 5 | 0.870753 | 0.943725 | 89 | 534 |
| S04 | 普通平均：最佳双模型结构分 | 2 | 0.507521 | 0.947468 | 99 | 526 |
| S05 | 普通平均：最佳三模型结构分 | 3 | 0.482678 | 0.954969 | 102 | 528 |
| S06 | 普通平均：官方3042行最高结构分 | 4 | 0.599192 | 0.948720 | 102 | 524 |
| S07 | 普通平均：官方3042行最高独立分 | 5 | 0.875189 | 0.914436 | 82 | 522 |
| S08 | 普通平均：不用large的最高独立分 | 3 | 0.880337 | 0.923292 | 82 | 528 |
| S09 | 普通平均：包含Qwen的最高独立分 | 6 | 0.886378 | 0.924826 | 83 | 528 |
| S10 | 普通平均：包含ModernBERT的最高独立分 | 5 | 0.883736 | 0.930859 | 83 | 532 |
| S11 | 几何平均：最高结构分 | 5 | 0.548800 | 0.960847 | 100 | 534 |
| S12 | 几何平均：两项目标同时达到 | 6 | 0.870621 | 0.941924 | 90 | 532 |
| S13 | 几何平均：官方3042行最高结构分 | 4 | 0.717115 | 0.954786 | 96 | 534 |
| S14 | 分层融合：0.938作为N门 | 6 | 0.545861 | 0.965418 | 101 | 536 |
| S15 | 分层融合：0.9306作为N门 | 6 | 0.486196 | 0.962259 | 99 | 536 |

## 各组合的成员

### S01 普通平均：最高结构分

模式：probability；相对0.938修好18行、破坏4行。

- M014（成员）：deberta_closure/run01_closure88/lexical_seed_42；训练行数=3130。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_closure\outputs\run01_closure88\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_closure\run01_closure88\lexical_seed_42\best_model.pth`
- M065（成员）：deberta_manual_aug/run12_local128_copy；训练行数=3418。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run12_local128_copy\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run12_local128_copy\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M075（成员）：deberta_manual_aug/run20_semantic64_staged_seed42；训练行数=3354。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run20_semantic64_staged_seed42\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run20_semantic64_staged_seed42\best_model.pth`
- M080（成员）：deberta_manual_aug/run26_row_sam；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run26_row_sam\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run26_row_sam\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`

### S02 普通平均：最高独立分

模式：probability；相对0.938修好14行、破坏20行。

- M008（成员）：deberta_bidirectional/run01_bidirectional_seed7；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_bidirectional\outputs\run01_bidirectional_seed7\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_bidirectional\run01_bidirectional_seed7\best_model.pth`
- M046（成员）：deberta_lexical_mixed_pair/run07_reversal_kl_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run07_reversal_kl_seed42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run07_reversal_kl_seed42\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`

### S03 普通平均：独立≥0.87且结构≥0.94

模式：probability；相对0.938修好16行、破坏12行。

- M056（成员）：deberta_manual_aug/run03_event_200；训练行数=3442。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run03_event_200\epoch_4\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run03_event_200\best_model.pth`
- M064（成员）：deberta_manual_aug/run11_local128_manual；训练行数=3418。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run11_local128_manual\epoch_6\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run11_local128_manual\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M072（成员）：deberta_manual_aug/run18_lora_r16_seed7_dropout0；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run18_lora_r16_seed7_dropout0\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run18_lora_r16_seed7_dropout0\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`

### S04 普通平均：最佳双模型结构分

模式：probability；相对0.938修好15行、破坏9行。

- M081（成员）：deberta_manual_aug/run27_row_pair_sam；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run27_row_pair_sam\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run27_row_pair_sam\best_model.pth`
- M084（成员）：deberta_original_aug/run01_original_scope；训练行数=3242。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_original_aug\outputs\run01_original_scope\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_original_aug\run01_original_scope\best_model.pth`

### S05 普通平均：最佳三模型结构分

模式：probability；相对0.938修好12行、破坏1行。

- M014（成员）：deberta_closure/run01_closure88/lexical_seed_42；训练行数=3130。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_closure\outputs\run01_closure88\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_closure\run01_closure88\lexical_seed_42\best_model.pth`
- M075（成员）：deberta_manual_aug/run20_semantic64_staged_seed42；训练行数=3354。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run20_semantic64_staged_seed42\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run20_semantic64_staged_seed42\best_model.pth`
- M080（成员）：deberta_manual_aug/run26_row_sam；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run26_row_sam\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run26_row_sam\best_model.pth`

### S06 普通平均：官方3042行最高结构分

模式：probability；相对0.938修好17行、破坏10行。

- M007（成员）：deberta_base_lexical/run01_raw_base/lexical_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_base_lexical\outputs\run01_raw_base\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_base_lexical\run01_raw_base\lexical_seed_42\best_model.pth`
- M028（成员）：deberta_lexical/run03/lexical_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical\outputs\run03\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\model\deberta_lexical\run03\lexical_seed_42\best_model.pth`
- M048（成员）：deberta_lexical_mixed_pair/run09_reason_all_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run09_reason_all_seed42\epoch_7\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run09_reason_all_seed42\best_model.pth`
- M050（成员）：deberta_lexical_reliability/run01/baseline_seed_7；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_reliability\outputs\run01\baseline_seed_7\epoch_6\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_reliability\run01\baseline_seed_7\best_model.pth`

### S07 普通平均：官方3042行最高独立分

模式：probability；相对0.938修好12行、破坏27行。

- M030（成员）：deberta_lexical/run04_weight/lexical_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical\outputs\run04_weight\lexical_seed_42\epoch_9\predictions_with_scores.csv`
  找到的权重：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\model\deberta_lexical\run04_weight\lexical_seed_42\best_model.pth`
- M045（成员）：deberta_lexical_mixed_pair/run06_oof_hard；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run06_oof_hard\epoch_7\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run06_oof_hard\best_model.pth`
- M046（成员）：deberta_lexical_mixed_pair/run07_reversal_kl_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run07_reversal_kl_seed42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run07_reversal_kl_seed42\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`
- M083（成员）：deberta_nli_xsmall/run01_xsmall_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_xsmall\outputs\run01_xsmall_mixed_pair\epoch_7\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_xsmall\run01_xsmall_mixed_pair\best_model.pth`

### S08 普通平均：不用large的最高独立分

模式：probability；相对0.938修好12行、破坏21行。

- M008（成员）：deberta_bidirectional/run01_bidirectional_seed7；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_bidirectional\outputs\run01_bidirectional_seed7\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_bidirectional\run01_bidirectional_seed7\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M083（成员）：deberta_nli_xsmall/run01_xsmall_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_xsmall\outputs\run01_xsmall_mixed_pair\epoch_7\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_xsmall\run01_xsmall_mixed_pair\best_model.pth`

### S09 普通平均：包含Qwen的最高独立分

模式：probability；相对0.938修好12行、破坏20行。

- M008（成员）：deberta_bidirectional/run01_bidirectional_seed7；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_bidirectional\outputs\run01_bidirectional_seed7\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_bidirectional\run01_bidirectional_seed7\best_model.pth`
- M051（成员）：deberta_lexical_reliability/run01/lexical_seed_7；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_reliability\outputs\run01\lexical_seed_7\epoch_1\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_reliability\run01\lexical_seed_7\best_model.pth`
- M060（成员）：deberta_manual_aug/run07_relation48_mixed；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run07_relation48_mixed\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run07_relation48_mixed\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`
- M094（成员）：qwen25_3b_reason/run01_c_reason02/C；训练行数=3517。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\qwen25_3b_reason\outputs\run01_c_reason02\C\best\predictions_with_scores.csv`
  找到的权重：`D:\LLM\train_model\DiCo-NLI\qwen2p5_3b_reason\run01_c_reason02\C\best_adapter\adapter_model.safetensors ; D:\LLM\train_model\DiCo-NLI\qwen2p5_3b_reason\run01_c_reason02\C\checkpoint-880\adapter_model.safetensors`

### S10 普通平均：包含ModernBERT的最高独立分

模式：probability；相对0.938修好14行、破坏18行。

- M008（成员）：deberta_bidirectional/run01_bidirectional_seed7；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_bidirectional\outputs\run01_bidirectional_seed7\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_bidirectional\run01_bidirectional_seed7\best_model.pth`
- M046（成员）：deberta_lexical_mixed_pair/run07_reversal_kl_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run07_reversal_kl_seed42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run07_reversal_kl_seed42\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`
- M093（成员）：modernbert_nli_base/run01_original_recipe/baseline_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\modernbert_nli_base\outputs\run01_original_recipe\baseline_seed_42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\modernbert_nli_base\run01_original_recipe\baseline_seed_42\best_model.pth`

### S11 几何平均：最高结构分

模式：log_probability；相对0.938修好20行、破坏5行。

- M017（成员）：deberta_feature_fusion/run01_feature_fusion/feature_fusion_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_feature_fusion\outputs\run01_feature_fusion\feature_fusion_seed_42\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_feature_fusion\run01_feature_fusion\feature_fusion_seed_42\best_model.pth`
- M036（成员）：deberta_lexical_aug/run02_lexical_relation_staged；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_aug\outputs\run02_lexical_relation_staged\epoch_7\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_aug\run02_lexical_relation_staged\best_model.pth`
- M050（成员）：deberta_lexical_reliability/run01/baseline_seed_7；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_reliability\outputs\run01\baseline_seed_7\epoch_6\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_reliability\run01\baseline_seed_7\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`
- M084（成员）：deberta_original_aug/run01_original_scope；训练行数=3242。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_original_aug\outputs\run01_original_scope\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_original_aug\run01_original_scope\best_model.pth`

### S12 几何平均：两项目标同时达到

模式：log_probability；相对0.938修好18行、破坏15行。

- M007（成员）：deberta_base_lexical/run01_raw_base/lexical_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_base_lexical\outputs\run01_raw_base\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_base_lexical\run01_raw_base\lexical_seed_42\best_model.pth`
- M009（成员）：deberta_bidirectional/run02_original_alignment_joint_seed42；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_bidirectional\outputs\run02_original_alignment_joint_seed42\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_bidirectional\run02_original_alignment_joint_seed42\best_model.pth`
- M018（成员）：deberta_frozen_n_head/run01_n_head；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_frozen_n_head\outputs\run01_n_head\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_frozen_n_head\run01_n_head\best_model.pth`
- M064（成员）：deberta_manual_aug/run11_local128_manual；训练行数=3418。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run11_local128_manual\epoch_6\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run11_local128_manual\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`
- M094（成员）：qwen25_3b_reason/run01_c_reason02/C；训练行数=3517。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\qwen25_3b_reason\outputs\run01_c_reason02\C\best\predictions_with_scores.csv`
  找到的权重：`D:\LLM\train_model\DiCo-NLI\qwen2p5_3b_reason\run01_c_reason02\C\best_adapter\adapter_model.safetensors ; D:\LLM\train_model\DiCo-NLI\qwen2p5_3b_reason\run01_c_reason02\C\checkpoint-880\adapter_model.safetensors`

### S13 几何平均：官方3042行最高结构分

模式：log_probability；相对0.938修好20行、破坏9行。

- M017（成员）：deberta_feature_fusion/run01_feature_fusion/feature_fusion_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_feature_fusion\outputs\run01_feature_fusion\feature_fusion_seed_42\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_feature_fusion\run01_feature_fusion\feature_fusion_seed_42\best_model.pth`
- M047（成员）：deberta_lexical_mixed_pair/run08_reason_evidence_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run08_reason_evidence_seed42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run08_reason_evidence_seed42\best_model.pth`
- M050（成员）：deberta_lexical_reliability/run01/baseline_seed_7；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_reliability\outputs\run01\baseline_seed_7\epoch_6\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_reliability\run01\baseline_seed_7\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`

### S14 分层融合：0.938作为N门

模式：hier_0938；相对0.938修好18行、破坏0行。

- M032（成员）：deberta_lexical/run06_row_parameter_anchor；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical\outputs\run06_row_parameter_anchor\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical\run06_row_parameter_anchor\best_model.pth`
- M044（成员）：deberta_lexical_mixed_pair/run05_mixed_rdrop；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run05_mixed_rdrop\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run05_mixed_rdrop\best_model.pth`
- M047（成员）：deberta_lexical_mixed_pair/run08_reason_evidence_seed42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical_mixed_pair\outputs\run08_reason_evidence_seed42\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_lexical_mixed_pair\run08_reason_evidence_seed42\best_model.pth`
- M059（N门）：deberta_manual_aug/run06_relation48_staged；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run06_relation48_staged\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run06_relation48_staged\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M084（成员）：deberta_original_aug/run01_original_scope；训练行数=3242。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_original_aug\outputs\run01_original_scope\epoch_3\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_original_aug\run01_original_scope\best_model.pth`

### S15 分层融合：0.9306作为N门

模式：hier_0930；相对0.938修好19行、破坏3行。

- M027（N门）：deberta_lexical/run02_branch_lr/lexical_seed_42；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_lexical\outputs\run02_branch_lr\lexical_seed_42\epoch_3\predictions_with_scores.csv`
  找到的权重：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\model\deberta_lexical\run02_branch_lr\lexical_seed_42\best_model.pth`
- M054（成员）：deberta_manual_aug/run01_original_manual；训练行数=3322。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run01_original_manual\epoch_2\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run01_original_manual\best_model.pth`
- M056（成员）：deberta_manual_aug/run03_event_200；训练行数=3442。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run03_event_200\epoch_4\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run03_event_200\best_model.pth`
- M071（成员）：deberta_manual_aug/run17_lora_r16_dropout01；训练行数=3290。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run17_lora_r16_dropout01\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run17_lora_r16_dropout01\best_model.pth`
- M075（成员）：deberta_manual_aug/run20_semantic64_staged_seed42；训练行数=3354。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_manual_aug\outputs\run20_semantic64_staged_seed42\best\dev\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_manual_aug\run20_semantic64_staged_seed42\best_model.pth`
- M082（成员）：deberta_nli_large/run01_large_mixed_pair；训练行数=3042。
  预测：`G:\pycode\SemEval\Task-2-DiCo-NLI-main\experiments\deberta_nli_large\outputs\run01_large_mixed_pair\epoch_10\predictions_with_scores.csv`
  找到的权重：`D:\model\SEAMEVAL\deberta_nli_large\run01_large_mixed_pair\best_model.pth`

## 解释边界

普通平均是每个模型四类softmax后等权平均；几何平均是log-prob等权平均。分层融合将N概率和F/B/E条件概率分别组合。
每行表格的独立分和结构分来自同一固定配方、同一份融合分数，没有拼接不同方案的最佳指标。
official3042表示每个成员声明使用官方3042行、未新增训练文本；部分含证据辅助标注、OOF权重或其他训练改动，不等于只用原始四分类监督。
这次进行了大量组合搜索，Dev分数有明显选择偏差，不是新测试集或交叉验证结果。条件bootstrap没有修正搜索偏差，也不包含随机种子变化。
详细成员与权重路径见selected_recipes.json，所有计算过的组合见all_combinations.csv。
