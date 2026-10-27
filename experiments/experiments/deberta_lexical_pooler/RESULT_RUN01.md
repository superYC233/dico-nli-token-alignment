# run01：恢复NLI pooler后的结果

核对日期：2026-10-26。已读取训练日志、本地best_metrics.json、各轮预测及混淆矩阵；未继续训练。

## 结果

最佳独立模型是第10轮，已保存，independent W-F1为0.8523490601，同一权重pair-aware为0.8893304839。

| 同一轮状态 | 独立F1 | 同轮pair-aware | 独立正确行数 | 独立F/B/E正确 | N正确 | 非N误判N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原mixed run01 ep2 | 0.850565 | 0.873506 | 563 | 498 | 65 | 25 |
| OOF加权 ep7 | 0.852621 | 0.892869 | 562 | 486 | 76 | 37 |
| 本次pooler ep10 | 0.852349 | 0.889330 | 562 | 486 | 76 | 36 |

原mixed ep2只有预测，不代表另一个保存epoch的权重可得到同样分数。本次按independent选模，历史mixed按pair-aware选模；上表用已保存的逐轮预测比较独立最佳状态。只有一组seed结果，尚不能证明因果或稳定提升。

本次相对原mixed独立F1增加0.1784个百分点；pair-aware增加1.5824个百分点，但独立总正确数少1。W-F1还取决于各类precision/recall，不能把轻微F1上升解释为净多修好样本。

原0.938模型使用不同的数据/训练方案，不应把0.938到0.889的差值归因于pooler。

## 独立错误

| 类型 | 原mixed ep2 | pooler ep10 |
| --- | ---: | ---: |
| N漏判 | 41 | 30 |
| 非N误判N | 25 | 36 |
| N边界合计 | 66 | 66 |
| E与F/B混淆 | 26 | 29 |
| F与B互判 | 5 | 3 |
| 全部错误 | 97 | 98 |

逐条对比原mixed，本次修好33条、破坏34条：

| 金标 | 修好 | 破坏 | 正确数变化 |
| --- | ---: | ---: | ---: |
| F | 11 | 5 | +6 |
| B | 3 | 8 | -5 |
| E | 4 | 17 | -13 |
| N | 15 | 4 | +11 |

所以本次主要体现N召回与其他类别之间的交换，没有减少N边界总错误；不能宣布解决了语义泛化。

按N分数排序的诊断也不一致：ROC-AUC由0.913426变0.899053，AP由0.717922变0.750476；固定只取前90条为N，命中67变63。排序确实有变化，并非严格等价于统一阈值偏移，但目前没有一致的整体改善证据。

## 高置信错误仍然存在

真实N被判E：`Opens / begins operation`、`performs / to do`、`onto another / to another`，错误预测置信度约0.978。

真实非N被判N：`of Texas fertiliser plant blast / in Texas fertilizer plant blast`（金标E）、`from Japan tunnel / in collapsed Japan tunnel`（金标B）、`orange and white cat / ginger and white cat`（金标E），N置信度约0.997以上。

这表明两边都有非常自信的错误，不能仅用抬N权重或压N阈值解决。短语语义与本任务标注约定需区分，不能据孤立短语直接宣布金标错。

## 训练曲线与结论

Row CE从0.858下降到0.0756；独立Dev在0.806—0.852之间波动，第10轮仍刷新最佳。低训练损失与有限Dev表现支持存在泛化问题，但本次没有评估Train准确率，不能凭loss写Train已99.9%。末轮最好也不能证明再加轮次就有提升。

加载正常：日志中pooler的UNEXPECTED来自只载入编码器的第一步，随后已从同一revision单独严格加载pooler。此前CPU检查确认逐元素一致。

建议本次记为“恢复NLI pooler后基本持平，未获得明确独立能力突破”，暂不继续扫pooler参数或叠加其他改动。若研究pooler本身的作用，需要同入口、同环境、USE_NLI_POOLER=False的匹配对照；若目标直接探索新方法，可回到稳定的不带pooler混合底座再试Child-Tuning-D，避免无法分辨多个改动的贡献。

分析文件在`outputs/run01_nli_pooler_seed42/comparison_analysis/`：ranking_summary.csv和independent_errors.csv。
