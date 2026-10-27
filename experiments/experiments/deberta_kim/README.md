# DeBERTa + Lexical + KIM 式局部知识

以 Lexical seed42（Dev pair-aware W-F1 0.930569）的**训练配方**为起点，从同一 NLI 主干重新训练。
不加载那个按 Dev 选出的最佳模型继续训练，不修改旧目录。

只借鉴 KIM 的局部推断知识融合，不是完整复现 KIM：
<https://aclanthology.org/P18-1224/>

## 改了哪里

保留原 CLS + Token Alignment + 20 维词汇分支。
对原文差异词查询 WordNet 的四种有向关系：同义、源词更具体、目标词更具体、反义。
通过 fast tokenizer 的字符偏移，把词对知识放到准确的子词位置；跨词 token 和截断词跳过。
用原对齐权重汇总当前 token 的词对知识，投影到 128 维，加入比较层线性输出，再经过原 ReLU。

```text
原比较 = W [h, aligned, h-aligned, h*aligned] + b
局部知识 = sum_j alignment[i,j] * relation[i,j]
新比较 = ReLU(原比较 + relation_projection(局部知识))
```

新增投影只有 512 个参数，无偏置、零初始化、隔离初始化 RNG。
反方向交换上下位通道。词对知识只来自两个原始文本，不读取任务标签或 pair_id。
同一单词拆成多个子词时，其关系附给这些子词；对齐权重仍在对方全部真实 token 上归一化。

沿用旧 WordNet 策略：过滤功能词/数字/单字母、排除共有表面词；每词性只取第一词义，向上最多 4 层。
没有词义消歧，也不会从词典硬推任务标签。多义词和标注口径差异仍可能引入噪声。
与旧全局 WordNet 分支不同：这里没有新增全局 12 维，也不修改注意力打分、池化或损失。

## 保留的训练设置

原始 Train 3042；float32；seed42；10 epochs；Row batch16；Pair batch16组；Dev batch16。
每轮先完整 Row CE 再完整 Real Pair CE，权重和平滑与原 Lexical 相同。
主干/原分类及对齐头 2e-5，原词汇分支 1e-3；新增局部投影先用比较层的 2e-5。
每轮报告独立预测和 pair-aware，按后者 W-F1 选最佳。
只保留一个最佳完整权重；原子写入临时文件后替换，因此更新时需额外一份模型的可用空间。
保存每轮预测/指标、词对关系审计、配置和资源校验信息。

## 运行

使用现有 nlpbase 环境即可（nltk、sentencepiece、transformers、torch 已在项目使用）。
WordNet 使用项目已有 `.nltk_data`，正常不需重新下载。

1. 可选：右键运行 `check_setup.py`，只做 CPU 数据和词典检查。
2. **正式训练：右键运行 `train_deberta_kim.py`。**
3. 重新预测最佳权重：运行 `evaluate_best.py`。

如果提示没有 WordNet，再运行 `prepare_wordnet.py`；只有这个入口允许下载。
如需重跑，先在 `config.py` 修改 `RUN_NAME`，已有结果不会覆盖。
命令行加 `--lexical-only` 可关闭新增知识模块，运行同代码的原 Lexical 对照。

模型：`model/deberta_kim/run01/kim_seed_42/best_model.pth`。
结果：本目录 `outputs/run01/kim_seed_42/`。
重点看 W-F1、FB_to_E_errors、E_to_FB_errors、F_B_reversal_errors、N_correct。
自动输出相对历史 Lexical 的修好/破坏样本。

## 验证范围

`test_cpu.py` 使用缓存的真实 tokenizer、WordNet 和随机微型 DeBERTa：
检查原输入一致性、有向关系及子词位置、零初始化和禁用模块的基线等价、Row/Pair 梯度、
两阶段训练、官方评分、RNG 恢复、最佳权重保存/严格重载。
不加载真实主干权重，不替用户启动正式 GPU 训练；检查通过不保证 Dev 分数提升。
