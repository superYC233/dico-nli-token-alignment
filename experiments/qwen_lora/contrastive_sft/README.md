# 原 C+r16 SFT + 文本对级监督对比学习

## 运行

1. 用原来的 nlpbase 环境，右键运行 **train_lora_contrastive.py**。
2. 完成后右键运行 **evaluate_lora_contrastive.py**。

训练入口沿用 train_lora_abc.py 的布局，主要参数都在该文件中。
脚本从基础 Qwen 重新初始化 LoRA，不接着训练旧 Adapter。
不会修改主线脚本、旧 Adapter 或旧预测目录。

```python
mode = 'C'
seed = 42
contrastive_weight = 0.1
temperature = 0.1
projection_dim = 128
```

这三个新增参数是第一轮探索设置，不是论文最优值或已经验证的设置。

模型目录：

```text
D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_contrastive/run01/C/best_adapter
```

预测结果目录：qwen_lora/outputs/qwen3_0p6b_contrastive/run01/C/。
原主线参考：pair-aware W-F1=0.899136759，N 正确 80，F/B/E 正确 514。

## 保持原样的部分

- 原 C 训练集，3042 + 475 = 3517 行。
- 旧短 Prompt，不加入检索例子。
- LoRA r=16、alpha=32、dropout=0.05、all-linear。
- lr=1e-4、5 epochs、训练 batch=16、梯度累积=1、bf16。
- 全词表 completion-only SFT，包括标签与 EOS；packing=False。
- 每轮用原 BestF1Callback、固定预测 batch=16、pair-aware 选最佳模型。
- 标准标签生成与原 pair-aware 推理，投影层不参与预测。

loss_type='chunked_nll' 明确保留当前 TRL 的分块全词表 SFT，避免物化所有位置的词表分数。
remove_unused_columns=False 只用于把同源编号交给 collator；编号不会传入 Qwen。

## 新增的训练目标

```text
loss = 原 SFT loss + 0.1 × supervised_contrastive_loss
```

在同一次 SFT 前向里，从首个监督标签之前的位置取隐藏向量。
位置 t-1 负责预测位置 t 的 F/B/E/N；不会取已经读到正确标签的隐藏向量。
通过一个两层小投影头，将该向量变成对比特征。

同一批次中：不同源、同标签的文本对作正例；不同标签作负例。
F 和 B 是两个类别，不能因为文本互换就当作正例。
同源、同标签的反向/重复样本既不作正例，也不进入该锚点的分母。
源分组按规范化后无序文本对确定，因此 C 中 pair_id 带 __swap 的 N 也能和原行归为同源。
这一步只增加分组编号，不改标签、不改行数、不改随机采样或训练类比例。

没有跨源同类正例的锚点不计算对比项；整批没有正例时，对比项为可求导的零。
不额外做第二次 Qwen 前向，也不堆叠 R-Drop、Pair CE、合法组合约束或检索 Prompt。

## 日志怎么看

- sft_loss：原全词表 SFT 损失。
- contrastive_loss：未加权的监督对比损失。
- weighted_contrastive_loss：乘以 0.1 后实际加进总损失的部分。
- contrastive_valid_fraction：本批有跨源同类正例的样本比例。

若 valid_fraction 长期为零，对比项没有实际信号，需要先检查批次与分组。
本实现保持 batch=16；不能把梯度累积后的样本数当作对比负例池大小。
Dev 的 eval_loss 仍只计算 SFT；选模型依旧看 eval_pair_f1。
权重设为 0 时完全走原 SFT compute_loss，可以作为同代码的零权重对照；另换输出目录。

## 保存和预测

checkpoint 与 best_adapter 中保存标准 PEFT Adapter、tokenizer，另存：

- contrastive_head.pt：小投影层参数，保证最佳轮次能完整恢复。
- contrastive_config.json：温度、权重、投影维度、标签 token ID 与 Prompt。

原来的 PeftModel.from_pretrained 可以直接加载这个 Adapter 用于预测，不需要加载投影层。
新预测入口额外检查当前 Prompt 与保存的模板是否相同。
训练使用 save_only_model=True，没有保存完整优化器续训状态，不宣称可以精确续训。

## 方法边界

这是受 [PairSCL](https://github.com/THU-BPM/PairSCL) 启发的监督对比目标改编，
不是原论文完整复现。原模型使用交叉注意力与分类 CE；本实验保留生成式 SFT，
取答案前的有序文本对表示，使用标准 SupCon 的正例平均对数概率。
原论文公式、模型、批次规模和我们的四类设置均不同，不能照搬论文提升幅度。

N 包含 SIMI/REL/OPPO，类内可能不均一；监督对比损失是否有益需要实测，
不能仅凭损失下降认定边界改善。最终同时报告 W-F1、N 与 F/B/E 正确数。
此次只使用官方 Train 标签，不加入 Dev 标签或外部细粒度标注。

## 文件

| 文件 | 用途 |
| --- | --- |
| train_lora_contrastive.py | 训练入口，参数布局沿用你的 train_lora_abc |
| contrastive_trainer.py | 保留 SFT 前向，加入辅助损失，保存/恢复投影层 |
| contrastive_data.py | 增加同源编号，避免靠重复副本构造正例 |
| evaluate_lora_contrastive.py | 复用原预测与 pair-aware |
| test_contrastive.py | CPU 检查损失、梯度、同源分组与无正例情况 |

