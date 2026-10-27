# Qwen 词级比较实验

当前采用的 C+r16+预测期 pair-aware 主线保留不动；本目录是独立的结构实验。

## 三个版本

- `simple`：每个 token 只参考自身，分别汇总 Text 1 / Text 2，再比较两个短语。
- `alignment`：每个 token 先参考另一短语中软匹配到的 token，再汇总比较。
- `alignment_global`：保留 alignment，再拼接最后有效 token 的投影向量，补充完整输入的信息。

simple 与 alignment 的 LoRA、投影层、比较层、分类器、参数数量、数据和训练损失一致。
`simple` 的比较输入是 `[t,t,0,t*t]`；`alignment` 的比较输入是 `[t,a,t-a,t*a]`，其中 a 是对方 token 的加权表示。
这个消融用于检查显式跨文本匹配的收益；simple 中差值块恒为零，参数数量一致不代表有效容量完全相同。

alignment_global 复用投影层取得整体向量，把分类器输入由 8d 扩到 9d。
因此它比 alignment 多一些分类头参数；这是加入整体信息的结构实验，不是参数量完全匹配的对照。
训练数据、r=16、学习率、batch、轮数、四分类 CE 和 pair-aware 都不变。

输入沿用旧短 Prompt，通过字符偏移定位两段正文，指令和补齐位置不参加短语汇总。
alignment_global 另取 attention_mask 指示的最后有效位置，此位置可以看到两段正文；它不是补齐位置，也没有拼入金标答案。
Qwen 仍然是因果模型：第一段 token 不能在主干中看到第二段；显式双向比较发生在新增模块内。
这里没有把 Qwen 改成双向编码器。

## 文件

| 文件 | 作用 |
| --- | --- |
| `pair_data.py` | 定位两段文本、构造掩码、右补齐 |
| `pair_model.py` | Qwen + LoRA + 比较头，以及完整保存/加载 |
| `pair_metrics.py` | 复用已有结构解码和官方评分 |
| `comparison_trainer.py` | 每轮评估、保存和恢复最佳完整模型 |
| `train_token_comparison.py` | 训练入口 |
| `evaluate_token_comparison.py` | 单独预测入口 |

## 运行

直接右键运行 `train_token_comparison.py`，默认 `HEAD_TYPE='alignment_global'`。
做对照时，仅把 `HEAD_TYPE` 改成 `'alignment'` 或 `'simple'` 后再运行，各版本保存到不同目录。
所有版本都从基础 Qwen 重新训练 LoRA，不加载旧主线的 Adapter。

训练目标为四类 CE，LoRA 和比较头一起更新；不再训练生成标签或 EOS，不叠加 Pair CE/R-Drop。
因此和旧 SFT 主线的差异包含读出与监督目标；要判断对齐模块本身，应比较 simple 与 alignment。
这是已有词级对齐思路的实验实现，不能仅因添加该模块就视为新方法。

每轮用固定 batch=16 计算 Dev 分数，随后复用原 pair-aware 和官方评分。
按 `eval_pair_f1` 保留最佳 checkpoint，结束后导出 `best_model/`。
这个目录同时包含 `adapter/`、`comparison_head.pt`、tokenizer 和 Prompt/结构配置。
不能仅用旧版 `PeftModel.from_pretrained` 预测，也不能只拷贝 Adapter。

默认模型目录：

```text
D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_token_comparison/run01/alignment/best_model
D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_token_comparison/run01/simple/best_model
D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_token_comparison/run01/alignment_global/best_model
```

默认结果目录为 `qwen_lora/outputs/token_comparison/run01/{版本}/`。
训练后可以右键运行 `evaluate_token_comparison.py`，它会读取最佳完整模型重新预测，默认也是 alignment_global；评估其他版本时同步修改该文件的 `HEAD_TYPE`。
读取已保存的 Prompt 模板，避免训练与预测模板不一致。

需要同时报告独立 W-F1、pair-aware W-F1、N 正确数和 F/B/E 正确数。
主线参考：pair-aware W-F1=0.899136759，N=80/106，F/B/E=514/554。
