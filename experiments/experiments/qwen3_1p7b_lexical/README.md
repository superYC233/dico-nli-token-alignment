# Qwen3-1.7B：C + r16 + 词汇特征分支

参考 `experiments/qwen3_1p7b` 的代码，新建独立目录。原 Qwen、DeBERTa 的代码和模型均不修改。

## 怎么运行

在 PyCharm 中直接运行 `train_lora_lexical.py`。
训练结束后直接运行本目录的 `evaluate_lora_pair.py`，它会同时加载最佳 LoRA 和词汇分支。

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments/qwen3_1p7b_lexical/train_lora_lexical.py
G:\anaconda3\envs\nlpbase\python.exe experiments/qwen3_1p7b_lexical/evaluate_lora_pair.py
```

正式训练需要 CUDA；本目录沿用已经下载的 `D:/LLM/model/Qwen3-1.7B`，无需重新下载。
`check_setup.py` 可以先检查真实数据、tokenizer、监督位置、特征和模型文件；不会加载 1.7B 大权重。

## 保持什么

- 同一个 Qwen3-1.7B，本机 tokenizer/chat template，关闭 thinking。
- C 数据：原 Train 3042 + 筛选后的反向 N 475，共 3517；Dev 660。
- LoRA：r=16、alpha=32、dropout=0.05、all-linear、bias=none。
- LoRA 学习率 1e-4，5 轮，batch=16、累积=1，bf16、非重入梯度检查点。
- completion-only 全词表 CE：只监督分类标签 + EOS，不监督 Prompt。
- 每轮 Dev 推理 batch=16，pair-aware F1 最佳选模；输出独立和结构解码两种结果。
- 从基础模型重新初始化 LoRA，不接着训练已有最佳 Adapter。

## 只增加什么

从原始 text1/text2 提取 DeBERTa 词汇版的 20 维特征：有方向的词汇覆盖、长度差异、数字、否定、去冠词后的重合等。
不读取 gold、pair_id 或 Dev 统计来构造特征。C 的反向新增行交换文本后，会重新提取正确方向的特征。

```text
20维特征 → Linear(20,32) → ReLU → Linear(32,4)
```

新增 804 个参数，使用 float32、独立学习率 1e-3，没有 dropout；最后一层初始化为零。
分支初始化用 fork_rng 隔离，不消耗 LoRA 后续的随机序列。

在分类标签位置，对完整词表分数的 F/B/E/N 四列加上修正：

```python
final_score_F = original_score_F + lexical_score_F
# B/E/N 同理，其余词表列保持原分数。
```

训练的 CE 分母仍包含全部词表。EOS 对应位置完全使用原模型分数。
预测时使用相同的四列修正，然后进行四候选 argmax 和原 pair-aware 解码；记录的完整词表生成也已经应用修正。

## 为什么改用了 Trainer 子类

原 `SFTTrainer(loss_type='chunked_nll')` 的内置分块路径直接读取 LM head 的权重，不能自动纳入新的词汇修正。
因此本目录用 `LexicalTrainer(Trainer)` 和自定义模型 forward：只提取受监督位置的隐藏向量，按小块计算完整词表 CE。
每块通过 checkpoint 在反向时重算，避免创建 `[batch, Prompt长度, 词表大小]` 的大 logits，也不返回所有层隐藏状态。
CPU 测试会把这一损失和标准全词表 CE 逐项对照，不能仅凭名字把它误认为四类 CE。

分块投影和训练器实现与原版不完全相同；零修正的数学等价和预测一致性已经测试，真实 GPU 整轮训练不承诺逐比特相同。
新增分支仍需实验才能判断是否提分。

## 参数与保存

主要设置在 `config.py`：

```python
RUN_NAME = 'run01'
USE_LEXICAL_FEATURES = True
LEXICAL_HIDDEN_SIZE = 32
LEXICAL_LEARNING_RATE = 1e-3
LOSS_CHUNK_SIZE = 8
```

- 最佳导出：`D:/LLM/train_model/DiCo-NLI/qwen3_1p7b_lexical/run01/C/best_adapter`。
- Dev 最终预测：`experiments/qwen3_1p7b_lexical/outputs/run01/C`。
- 每轮 Dev 结果：模型输出目录下 `dev_scores/epoch_*`。
- 只在 F1 刷新最佳时保存，`save_total_limit=1` 保留一个最佳 checkpoint。
- 训练结束恢复最佳后，再导出 `best_adapter`；这是同一个最佳模型的固定名称副本。
- 每份保存包含 `adapter_model.safetensors`、`adapter_config.json`、`lexical_head.pt`、tokenizer 和 `run_config.json`；不会保存一份新的 1.7B 基础权重。
- 自动恢复最佳时会同时恢复 LoRA 与词汇分支；不能只用旧评估脚本加载 Adapter 而漏掉 `lexical_head.pt`。
- 只保存模型，不支持用 checkpoint 恢复优化器续训。
- 新运行先改 `RUN_NAME`，避免覆盖旧模型。

将 `USE_LEXICAL_FEATURES=False` 且使用新 RUN_NAME，可做同代码的零修正对照。

## 已包含的检查

`test_cpu.py` 用随机微型 Qwen，不加载或训练真实 1.7B 权重：

1. 零修正时与标准全词表 CE 的损失和 LoRA 梯度一致。
2. 非零修正只作用在答案标签位置，EOS 不受修正；梯度累积的归一化和 bf16 数值有效。
3. 词汇分支和 LoRA 均有梯度，新增参数 804 个。
4. 真实 tokenizer 和 C 数据的标签/EOS 边界正确。
5. 零修正生成和旧预测函数的四类分数逐条一致。
6. 两步 Trainer 流程、最佳 F1 选模、LoRA 与小分支保存、最佳回载、全新实例重载后预测一致。

测试日志里的随机模型 F1 只用于检查流程，不代表实际模型能力。
