# DeBERTa A–E 数据消融

本目录用于A—E数据对照，基于[原0.9248版](../deberta_original/README.md)；原模型和历史预测单独保留。

## 怎么运行

1. 可选：右键运行 `check_setup.py`。只检查 A–E 数据，不加载预训练模型、不训练。
2. 正式实验：右键运行 `run_all.py`。按顺序训练 A、B、C、D、E，每组重新启动一个进程，结束后释放显存。共五组，每组 10 轮。
3. 结果：本目录 `outputs/run01/summary.csv`，每组包含独立预测和 pair-aware 两行。

只跑一组：修改 `train_deberta_abcde.py` 顶部 `MODE = 'C'`，右键运行该文件。
`run_all.py` 顶部 `MODES` 控制批量运行哪些组；它不读取单组入口的 MODE。

后续新一批实验，在 `config.py` 把 `RUN_NAME` 改为 `run02`。已有输出目录会触发报错，防止覆盖或混入旧结果。
如果某批中途停止，只将 `run_all.py` 的 MODES 改为尚未运行的组。未完成组的残留目录需人工检查，程序不会自动删除。

## 五组的唯一计划变量：训练数据

| 组 | 数据 | 行数 | Real Pair 组数 |
|---|---|---:|---:|
| A | 原始 Train | 3042 | 1282 |
| B | 原始 + 筛选后的 475 条 N 原样复制 | 3517 | 1282 |
| C | 原始 + 同一批 475 条 N 反向 | 3517 | 1282 |
| D | 原始 + 全部数据原样复制 | 6084 | 2564 |
| E | 原始 + 全部数据反向，不安全 N 改为复制 | 6084 | 2564 |

475 条 N 的选择与当前 Qwen C 一致：排除反向文本对已在 Train 中存在的 N，再去掉候选内部同向文本重复。只用 Train，不用 Dev 或模型预测选样本。筛选不等于人工证明 N 反转保标，这是实验假设。

所有新增副本的 instance_id 和 pair_id 加 `__extra`。新增 F/B/E 保留完整新 Pair，N 保持单例。原始 Train 不删除、不改标签。每组保存实际 train.csv、n_source_audit.csv 和 extra_audit.csv 便于检查。

## 沿用原 DeBERTa 的训练配方

- 主干：MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli。
- CLS 分支 + Token Alignment；128 维、temperature 0.1、ReLU、相同初始化。
- float32 全参数训练，不使用 LoRA 或 Qwen 的 SFT。
- seed 42，在加载主干和初始化分类分支之前设置。
- 每轮先完成全部 Row CE，再完成全部 Real Pair CE，各 batch 独立 optimizer.step。
- AdamW，lr=2e-5，weight_decay=0.01；无新增 scheduler、AMP、梯度裁剪。
- Row CE 权重 F/B/E/N=[1,1,1,1.5]。
- Pair CE 权重 FB/BF/EE=[1,1,1.25]，label smoothing=0.05，损失乘 0.5。
- Row batch=16 行，Pair batch=16 组，Dev batch=16，max_length=128，10 epochs。
- 原版三组合 Pair CE 不包含 N，并非之前 Qwen 的合法组合概率约束。

每组从预训练主干重新开始，不接着 epoch_7 或上一组继续训练。默认从项目原有 Hugging Face 缓存离线读取主干；沿用 project_paths.py 的缓存路径设置。

新增逐轮 Dev 评估和最佳保存：每轮保存小型 logits/预测/评分文件，仅保存当前最佳的一份大权重。以 Dev pair-aware Weighted F1 选模，同分保留更早轮次；独立预测和 pair-aware 始终报告同一轮。评估前后恢复随机状态，避免逐轮评估影响后续 shuffle/dropout。不会把全部 epoch 大权重留在磁盘。

## 输出位置

- 新权重：项目根目录 `model/deberta_abcde/run01/A_seed_42/best_model.pth`，其他组同理。
- 每组完整配置：模型目录、结果目录的 `run_config.json`。
- 最佳轮次：模型目录 `best_metrics.json`。
- 每轮曲线：本目录 `outputs/run01/A_seed_42/epoch_summary.csv`。
- 最佳轮成绩：同目录 `best_summary.csv`。
- 每轮原始 logits、独立和结构预测：同目录 `epoch_数字/predictions_with_scores.csv`。
- A–E 汇总：本目录 `outputs/run01/summary.csv`。

可选重新预测：在 `evaluate_best.py` 选择 MODE，运行后从本组 best_model.pth 严格加载，结果写入该组 best_reevaluation 目录，不训练。

## 如何比较

主比较为 C−B 和 E−D：同数据量、同标签数量、同 Row/Pair 更新次数，比较反向与复制。A 与新组均用本轮协议重跑，0.9248 仅作历史参照，不保证原分数重现。

D/E 的 Pair 训练也翻倍，这是当前“完整复制/完整反向”定义带来的训练量变化。不能把 D−A 单独解释为数据多样性作用。

N precision=1 和 SoftCons=1 在当前 Dev 的结构解码后由分组结构保证，不代表模型学到完美判断。仍需检查 N 正确数和 F/B/E 正确数。当前是单种子、反复使用过的 Dev 上的探索，不能据微小分差保证独立测试集提升。

## CPU 检查

`test_cpu.py` 不加载真实预训练大权重、不占用 GPU。检查 A–E 的数据和配对，用随机初始化的一层微型 DeBERTa 检查与原模型的初始化/前向/梯度一致性、Row/Pair 反向、评估随机状态恢复和保存加载。若历史 Qwen CSV 存在，还会逐条核对相同的结构解码与官方评分。小模型分数没有实验意义。
