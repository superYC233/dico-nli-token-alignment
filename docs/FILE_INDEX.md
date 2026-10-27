# 文件目录索引

## 现在主要看这几个位置

| 位置 | 用途 |
|---|---|
| [项目 README](../README.md) | 任务、14阶段实验、成绩和导师讨论问题 |
| [实验分类索引](../experiments/README.md) | 按模型、数据增强、诊断、训练方法和融合分类查找实验 |
| [融合入口](../experiments/ensemble_selected/README.md) | S01／S11／S12／S14／S16的训练和预测 |
| [train.py](../experiments/deberta_original/train.py) | 原0.9248版DeBERTa Token Alignment训练入口，四分类＋Real Pair CE |
| [evaluate.py](../experiments/deberta_original/evaluate.py) | 复核原0.9248版保存的Dev预测，不加载模型 |
| [artifacts/final](../artifacts/final) | 原0.9248版预测、三个官方分数、配置和提交包；保留历史目录名 |
| [S14本次结果](../experiments/ensemble_selected/outputs/S14/run01_retrain/summary.csv) | 同一组融合分数的独立0.545861／结构0.962381 |
| [S14提交说明](../submissions/S14_track1_dev_20261027/README.md) | 英文开发阶段提交包和本地验证结果 |
| [完整实验记录](history/DICO_NLI_FINAL_EXPERIMENT_RECORD.md) | 原文保留；其中历史旧路径可在本页或 file_manifest.json 查找新位置 |
| [快速回顾](DICO_NLI_SIMPLE_REVIEW.md) | 项目发展过程的大白话记录 |
| [一页摘要](DICO_NLI_TLDR.md) | 最终配置和结果摘要 |
| [model_inventory.json](model_inventory.json) | 本地模型目录及其已有产物清单 |
| [Dev重合核对](DATA_AUGMENTATION_DEV_OVERLAP_AUDIT.md) | 原Train、人工增强和copy与Dev的重合结果 |
| [Pair目标研究](research/PAIR_OBJECTIVE_RESEARCH_2026-10-24.md) | 三组合Pair CE、阶段漂移及相关方法 |
| [GitHub更新说明](GITHUB_UPLOAD_GUIDE.md) | 上传范围和使用方法 |
| [本次移动记录](DIRECTORY_ORGANIZATION_20261027.json) | 文件新旧路径、校验值和文档修改清单 |

根目录散文件已归位：原版分析脚本在[tools/analysis](../tools/analysis/analyze_deberta_original.py)，Qwen早期Real Pair CE脚本在[Qwen2.5实验目录](../experiments/qwen2p5_3b/qwen_dico_lora_real_pair_ce.py)，闭包生成器与两批旧CSV在[deberta_closure](../experiments/deberta_closure/legacy_data/README.md)。`experiments/generative_nli.py`是多个生成式实验共用的工具模块。

## 历史实验和工具

### experiments/alignment

Token Alignment、NULL、均匀对齐和边界实验

| 原文件名 | 整理后位置 |
|---|---|
| deberta_nli_base_null_alignment.py | [deberta_nli_base_null_alignment.py](../experiments/alignment/deberta_nli_base_null_alignment.py) |
| deberta_nli_base_null_alignment_M.py | [deberta_nli_base_null_alignment_M.py](../experiments/alignment/deberta_nli_base_null_alignment_M.py) |
| deberta_nli_base_null_alignment_M_clean.py | [deberta_nli_base_null_alignment_M_clean.py](../experiments/alignment/deberta_nli_base_null_alignment_M_clean.py) |
| deberta_nli_base_null_alignment_M_internal.py | [deberta_nli_base_null_alignment_M_internal.py](../experiments/alignment/deberta_nli_base_null_alignment_M_internal.py) |
| deberta_nli_base_token_alignment.py | [deberta_nli_base_token_alignment.py](../experiments/alignment/deberta_nli_base_token_alignment.py) |
| deberta_nli_base_token_alignment_boundary.py | [deberta_nli_base_token_alignment_boundary.py](../experiments/alignment/deberta_nli_base_token_alignment_boundary.py) |
| deberta_nli_base_token_alignment_clean.py | [deberta_nli_base_token_alignment_clean.py](../experiments/alignment/deberta_nli_base_token_alignment_clean.py) |
| deberta_nli_base_token_alignment_hard_e_clean.py | [deberta_nli_base_token_alignment_hard_e_clean.py](../experiments/alignment/deberta_nli_base_token_alignment_hard_e_clean.py) |
| deberta_nli_base_token_alignment_internal.py | [deberta_nli_base_token_alignment_internal.py](../experiments/alignment/deberta_nli_base_token_alignment_internal.py) |
| deberta_nli_base_token_alignment_null_v2.py | [deberta_nli_base_token_alignment_null_v2.py](../experiments/alignment/deberta_nli_base_token_alignment_null_v2.py) |
| deberta_nli_base_uniform_alignment.py | [deberta_nli_base_uniform_alignment.py](../experiments/alignment/deberta_nli_base_uniform_alignment.py) |
| deberta_nli_large_token_alignment.py | [deberta_nli_large_token_alignment.py](../experiments/alignment/deberta_nli_large_token_alignment.py) |
| deberta_nli_large_token_alignment_null.py | [deberta_nli_large_token_alignment_null.py](../experiments/alignment/deberta_nli_large_token_alignment_null.py) |

### experiments/augmentation

人工和 train-only 增强的历史实验

| 原文件名 | 整理后位置 |
|---|---|
| deberta_nli_base_token_alignment_manual_boundary.py | [deberta_nli_base_token_alignment_manual_boundary.py](../experiments/augmentation/deberta_nli_base_token_alignment_manual_boundary.py) |
| deberta_nli_base_token_alignment_manual_boundary_clean.py | [deberta_nli_base_token_alignment_manual_boundary_clean.py](../experiments/augmentation/deberta_nli_base_token_alignment_manual_boundary_clean.py) |
| deberta_nli_base_token_alignment_trainonly_aug_clean.py | [deberta_nli_base_token_alignment_trainonly_aug_clean.py](../experiments/augmentation/deberta_nli_base_token_alignment_trainonly_aug_clean.py) |

### experiments/auxiliary

辅助任务、分层分类和多原型实验

| 原文件名 | 整理后位置 |
|---|---|
| deberta_large_type_aux.py | [deberta_large_type_aux.py](../experiments/auxiliary/deberta_large_type_aux.py) |
| deberta_nli_base_direction_aux.py | [deberta_nli_base_direction_aux.py](../experiments/auxiliary/deberta_nli_base_direction_aux.py) |
| deberta_nli_base_entailment_classifier.py | [deberta_nli_base_entailment_classifier.py](../experiments/auxiliary/deberta_nli_base_entailment_classifier.py) |
| deberta_nli_base_hierarchical.py | [deberta_nli_base_hierarchical.py](../experiments/auxiliary/deberta_nli_base_hierarchical.py) |
| deberta_nli_base_multi_prototype.py | [deberta_nli_base_multi_prototype.py](../experiments/auxiliary/deberta_nli_base_multi_prototype.py) |
| deberta_real_pair_ce_weights_type_aux.py | [deberta_real_pair_ce_weights_type_aux.py](../experiments/auxiliary/deberta_real_pair_ce_weights_type_aux.py) |

### experiments/baselines

早期 BERT / DeBERTa / Large / ModernBERT 基线

| 原文件名 | 整理后位置 |
|---|---|
| bert_baseline.py | [bert_baseline.py](../experiments/baselines/bert_baseline.py) |
| deberta_baseline.py | [deberta_baseline.py](../experiments/baselines/deberta_baseline.py) |
| deberta_large.py | [deberta_large.py](../experiments/baselines/deberta_large.py) |
| deberta_large_baseline.py | [deberta_large_baseline.py](../experiments/baselines/deberta_large_baseline.py) |
| modernbert_large_baseline.py | [modernbert_large_baseline.py](../experiments/baselines/modernbert_large_baseline.py) |

### experiments/pair_training

Pair CE、一致性、权重、平滑与训练阶段实验

| 原文件名 | 整理后位置 |
|---|---|
| deberta_large_interleaved.py | [deberta_large_interleaved.py](../experiments/pair_training/deberta_large_interleaved.py) |
| deberta_large_pair_head.py | [deberta_large_pair_head.py](../experiments/pair_training/deberta_large_pair_head.py) |
| deberta_nli_base_pair_smoothing.py | [deberta_nli_base_pair_smoothing.py](../experiments/pair_training/deberta_nli_base_pair_smoothing.py) |
| deberta_nli_base_pair_smoothing_clean.py | [deberta_nli_base_pair_smoothing_clean.py](../experiments/pair_training/deberta_nli_base_pair_smoothing_clean.py) |
| deberta_nli_base_stage_probe.py | [deberta_nli_base_stage_probe.py](../experiments/pair_training/deberta_nli_base_stage_probe.py) |
| deberta_pair_ce.py | [deberta_pair_ce.py](../experiments/pair_training/deberta_pair_ce.py) |
| deberta_pair_consistency.py | [deberta_pair_consistency.py](../experiments/pair_training/deberta_pair_consistency.py) |
| deberta_real_pair_ce.py | [deberta_real_pair_ce.py](../experiments/pair_training/deberta_real_pair_ce.py) |
| deberta_real_pair_ce_weights.py | [deberta_real_pair_ce_weights.py](../experiments/pair_training/deberta_real_pair_ce_weights.py) |
| deberta_real_pair_ce_weights_nli_base.py | [deberta_real_pair_ce_weights_nli_base.py](../experiments/pair_training/deberta_real_pair_ce_weights_nli_base.py) |
| deberta_real_pair_ce_weights_scheduler.py | [deberta_real_pair_ce_weights_scheduler.py](../experiments/pair_training/deberta_real_pair_ce_weights_scheduler.py) |

### predictions/legacy_calibration

原来散落在根目录的惩罚系数预测 CSV

| 原文件名 | 整理后位置 |
|---|---|
| dev_deberta_predictions_penalty_0.0.csv | [dev_deberta_predictions_penalty_0.0.csv](../predictions/legacy_calibration/dev_deberta_predictions_penalty_0.0.csv) |
| dev_predictions_penalty_0.0.csv | [dev_predictions_penalty_0.0.csv](../predictions/legacy_calibration/dev_predictions_penalty_0.0.csv) |
| dev_predictions_penalty_0.25.csv | [dev_predictions_penalty_0.25.csv](../predictions/legacy_calibration/dev_predictions_penalty_0.25.csv) |
| dev_predictions_penalty_0.5.csv | [dev_predictions_penalty_0.5.csv](../predictions/legacy_calibration/dev_predictions_penalty_0.5.csv) |
| dev_predictions_penalty_1.0.csv | [dev_predictions_penalty_1.0.csv](../predictions/legacy_calibration/dev_predictions_penalty_1.0.csv) |
| dev_predictions_penalty_1.5.csv | [dev_predictions_penalty_1.5.csv](../predictions/legacy_calibration/dev_predictions_penalty_1.5.csv) |
| dev_predictions_penalty_1.csv | [dev_predictions_penalty_1.csv](../predictions/legacy_calibration/dev_predictions_penalty_1.csv) |

### tools/analysis

双向 NLI 诊断工具

| 原文件名 | 整理后位置 |
|---|---|
| probe_bidirectional_nli.py | [probe_bidirectional_nli.py](../tools/analysis/probe_bidirectional_nli.py) |

### tools/data

内部划分、增强构造和边界案例挖掘工具

| 原文件名 | 整理后位置 |
|---|---|
| make_pair_split.py | [make_pair_split.py](../tools/data/make_pair_split.py) |
| make_train_only_boundary_augmentation.py | [make_train_only_boundary_augmentation.py](../tools/data/make_train_only_boundary_augmentation.py) |
| mine_boundary_pairs.py | [mine_boundary_pairs.py](../tools/data/mine_boundary_pairs.py) |

### tools/downloads

预训练模型下载工具

| 原文件名 | 整理后位置 |
|---|---|
| deberta_real_pair_ce_direction_aux.py | [download_deberta_nli_classifier.py](../tools/downloads/download_deberta_nli_classifier.py) |
| download.py | [download_deberta_base.py](../tools/downloads/download_deberta_base.py) |
| download_DeBERTa-v3-base-mnli-fever-anli.py | [download_deberta_nli_base.py](../tools/downloads/download_deberta_nli_base.py) |
| download_DeBERTa-v3-large-mnli.py | [download_deberta_nli_large.py](../tools/downloads/download_deberta_nli_large.py) |
| download_ModernBERT.py | [download_modernbert.py](../tools/downloads/download_modernbert.py) |

### tools/predict

早期 NULL 模型的 Trial 预测工具

| 原文件名 | 整理后位置 |
|---|---|
| predict_unlabeled.py | [predict_trial_null.py](../tools/predict/predict_trial_null.py) |

## 数据和产物

- `data/train`、`data/dev`、`data/trial`：官方数据，内容原样保留。
- `data/internal_split*`：从官方训练集划分的内部实验数据，不是官方 DEV。
- `data/augmentation`：已有 train-only 增强文件。
- `data/legacy_augmentation`：从旧外部实验目录复制的历史 CSV，不属于最终训练数据。
- `model`：本地权重和各实验原始产物，路径保留，Git 忽略。
- `results`：已有官方评分结果。
- `predictions/legacy_calibration`：原来散落在根目录的惩罚系数预测 CSV。
- `submissions`：按方案、赛道和日期保存提交包；旧原版包和本次S14包分开保留。

## 运行习惯

脚本通过自身位置定位项目根目录；早期入口使用`project_paths.py`适配，后期实验多在各自入口中设置路径。
历史实验保留各自原来的参数和入口；有些依赖已有 checkpoint 或外部预训练模型。
本机 D 盘预训练模型缓存路径保留。最终训练入口也支持 DICO_MODEL_CACHE 环境变量，其他机器未配置时使用 Hugging Face 默认缓存。新训练输出目录与已提交模型目录分开。
文件整理与 README 编写期间未执行训练；运行方式见根目录 README。
