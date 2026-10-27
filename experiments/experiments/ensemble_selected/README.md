# 五个融合方案

每个方案一个Python文件：先复用数据、配方和选模条件匹配的已有模型，缺少权重的成员再训练；然后重新加载成员权重预测，最后融合。只剩历史预测CSV不能代替模型权重。

| 入口 | 融合方式 | 成员和数据说明 |
|---|---|---|
| [S01.py](S01.py) | 四类概率普通平均 | [项目README第14阶段](../../README.md#14-多模型测试合并不同模型的预测) |
| [S11.py](S11.py) | 四类概率几何平均 | 同上 |
| [S12.py](S12.py) | 四类概率几何平均，包含Qwen | 同上 |
| [S14.py](S14.py) | 原0.938模型提供N概率，五个成员比较F/B/E | 同上 |
| [S16.py](S16.py) | 仅用官方3042行的分层融合 | 同上 |

通常在文件顶部修改`RUN_NAME`和本机模型保存目录，再运行所需方案。`--prepare-only`只检查数据与配方；`--evaluate-only`只加载已有权重预测。

## 结果位置

输出格式为`outputs/方案编号/RUN_NAME/`：

- `summary.csv`：融合后的independent和pair-aware成绩。
- `predictions_with_scores.csv`：融合分数与逐行预测。
- `members/成员编号/`：各成员本次预测与成绩。
- `run_config.json`：本次实际模型路径、融合成员和设置。
- `pair_aware/official_predictions.csv`：两列官方格式预测，打包后提交。

已核对的本次运行：

| 方案 | independent F1 | pair-aware F1 | 结果 |
|---|---:|---:|---|
| S14，run01_retrain | 0.545861 | 0.962381 | [summary.csv](outputs/S14/run01_retrain/summary.csv) |
| S16，run02_pair_select | 0.481308 | 0.954759 | [summary.csv](outputs/S16/run02_pair_select/summary.csv) |

S14的历史成员CSV组合曾得到0.965418；本次重载权重结果为0.962381，不能把历史值当成本次结果。本次英文Dev提交包见[提交说明](../../submissions/S14_track1_dev_20261027/README.md)。

历史组合搜索及完整成员说明见[融合报告](../model_ensemble/outputs/run01_all_existing/RESULTS.md)。训练数据说明见[data/README](../../data/README.md)；全项目实验分类见[实验索引](../README.md)。
