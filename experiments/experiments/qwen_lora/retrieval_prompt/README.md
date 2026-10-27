# 原 C+r16 + 检索示例 Prompt

本实验不重新训练，三个组加载同一个原 C+r16 SFT Adapter。

| 方法 | Prompt 中的示例 |
| --- | --- |
| baseline | 无示例；逐字复用原来的短 Prompt |
| random | 从原始 Train 随机选 2 个不同 pair_id 的例子 |
| retrieval | 从同一候选池检索分数最高的 2 个不同 pair_id 的例子 |

示例均包含两段原文及其官方训练标签；待预测问题的标签不会进入 Prompt。
不是 SFT + alignment 融合，也不加载 token_comparison 模型。

## 最简单的运行方法

在 PyCharm 用原来的 nlpbase 环境右键运行 **run_all.py**。
它依次完成：准备示例 → 加载 Qwen 一次 → 预测三组 → 官方评分 → 汇总。
路径基于脚本位置，PyCharm 的工作目录不需要另外调整。

准备示例使用 CPU，正式 Qwen 预测需要 CUDA。不会自动训练，也不会修改 Adapter。
使用现有 transformers、torch、peft、pandas、numpy、tqdm，不需要 sentence-transformers 或 FAISS。
首次没有 BGE 缓存时，需要联网下载 BAAI/bge-small-en-v1.5。
如果下载不方便，在 config.py 把 EMBEDDING_MODEL 改为你已经下载的本地 BGE 目录。

也可以分步运行：

1. **prepare_examples.py**：先准备例子，查看 `prompt_preview.txt`。
2. **evaluate_retrieval.py**：加载原模型，跑三组预测，结束后自动汇总。
3. **summarize_results.py**：以后只想重新查看结果时运行，不重新预测。

## 默认配置

所有路径和参数都在 config.py。

```python
MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
ADAPTER_PATH = (
    'D:/LLM/train_model/DiCo-NLI/'
    'qwen3_0p6b_abc/run04_r16/C/best_adapter'
)
NUM_EXAMPLES = 2
SEED = 42
PREDICT_BATCH_SIZE = 16
```

改例子数、模型、seed 或其他协议时，请把 RUN_NAME 换成新的名称。
程序会检测已有准备数据和预测配置，避免把不同实验混在同一目录。
同配置再次运行会复用示例和向量，但正式预测会重新跑三组。

## 检索是怎样做的

示例库使用原始 Train 的 3042 行，不使用 C 中新增的反向 N 副本。
这样重复训练行不会重复占据检索名额。

BGE 按官方 Transformers 用法取 CLS 向量并进行 L2 归一化。
两段文本分别编码，按原方向计算：

```text
分数 = 0.5 × cos(问题 Text1, 示例 Text1)
     + 0.5 × cos(问题 Text2, 示例 Text2)
```

不把标签拼入向量，不根据待测金标选示例，不自动翻转示例或猜标签。
相似度分数表示检索排序，不表示两个样本具有相同关系。

随机组和检索组使用相同的清理规则：

- 排除训练库内部存在正向或反向标签冲突的源 pair_id，保留排除审计表。
- 同标签、同有序文本对重复只留一条。
- 针对每个问题，排除相同 pair_id，以及规范化后相同/反向相同文本对所在的整组。
- 两个例子来自不同源 pair_id。

规范化仅做大小写与空白统一，用于重复排除；BGE 和 Prompt 使用原文。
该规则只处理示例选择，不会消除原 Adapter 训练时已经存在的 Train/Dev 重合。
这里不进行新的训练；未来若用检索 Prompt 训练，必须排除查询自身及同源 pair_id。

## 预测与评分口径

三组都固定 batch=16、左补齐、贪婪生成一步、同一个模型和推理精度。
Prompt 超过 MAX_PROMPT_TOKENS 时直接报错，不自动截断、缩减例子或改变 batch。

三组统一使用 F/B/E/N 四候选分数 argmax 作为独立分类答案。
同时保存原始无限制生成文本 `generated_prediction`，汇总非法生成数。
合法生成与四候选读出通常一致；若出现 Class 等文本，不会静默删除该行或伪装成合法生成。
四候选概率为条件于这四个候选的 softmax，不是全词表标签概率总量。

pair-aware 复用原 evaluate_lora_pair.py：配对组选择 FB/BF/EE，单例保持原预测。
金标到评分阶段才合并。两种解码都调用项目的官方 score_files。
三组都保存独立与 pair-aware 指标，以及 N 正确数、F/B/E 正确数和非法输出数。

baseline 会与历史 C+r16 CSV 逐条比较，写入 baseline_reproduction.json。
如果 baseline 未复现，先查看不同的样本与输入/模型配置，再解释组间提升。
本实验的主要对照是本次重新运行的 baseline，不是仅引用历史分数。

## 输出

默认在 qwen_lora/outputs/retrieval_prompt/run01/：

```text
prepared/
  manifest.json                 数据与配置指纹、原 Prompt 模板
  candidate_pool.csv            实际示例库
  excluded_candidates.csv       冲突/重复排除审计
  query_exclusions.csv          每个问题排除的候选行数
  phrase_embeddings.npz         预先计算的短语向量
  embedding_metadata.json       向量缓存配置
  demonstrations.csv            每个问题选了谁、原文、标签、相似度
  prompts.csv                   三组完整 Prompt；不含 Dev 金标
  prompt_preview.txt            第一个问题的三组 Prompt，方便检查
evaluation/
  inference_config.json         原 Adapter 校验值与推理设置
  baseline_reproduction.json    本次 baseline 与历史结果的逐条对照
  baseline/                     分数、预测、两种解码的官方评分
  random/
  retrieval/
summary.csv                     六行成绩：三组 × 两种解码
comparisons.csv                 修好/破坏/改后仍错，以及 W-F1 差值
example_label_distribution.csv  两种示例选择方法的标签分布
```

已有模型和历史预测目录不会被覆盖。旧 SFT 没有见过这种 few-shot 格式，
所以首轮结果用于探索；效果不好也不能单独证明检索方法无效。
随机对照能区分“加入示例”与“按相似度选例”的效果，但示例长度和标签分布仍可能不同。

## 文件与检查

- config.py：配置。
- retrieval_utils.py：候选清理、向量、选例、Prompt。
- prepare_examples.py：准备三组输入。
- evaluate_retrieval.py：Qwen 预测与官方评分。
- summarize_results.py：汇总成绩与逐条变化。
- run_all.py：完整入口。
- test_retrieval.py：CPU 检查重复排除、方向、标签隔离和非法生成。

方法参考：
[BGE 模型卡与 CLS 用法](https://huggingface.co/BAAI/bge-small-en-v1.5)、
[检索上下文示例的论文](https://aclanthology.org/2022.deelio-1.10/)。
这些文献不能保证在 Qwen3-0.6B 与 DiCo-NLI 上提升，需以三组实际对照为准。

