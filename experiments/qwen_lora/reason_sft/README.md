# Qwen3-0.6B：原C+r16 + 文字理由辅助SFT

右键运行本目录的 `train_lora_reason.py`。以后只改 `config.py` 中的 `RUN_NAME` 和参数，不需要再建实验代码目录。

## 数据和任务

标签任务完全沿用旧C构造：官方Train3042行 + 筛选反向N475行，共 **3517行**。直接调用原 `data_variants.build_train_dataset`，没有另外新增标签样本。

理由来自上一轮已经完成的Train理由CSV。仅使用 `annotation_status=supported` 的说明：原Train1756行，加上其中可复用到旧C反向N的284行，共 **2040行**。其余1477行仍训练标签，跳过解释任务。反向样本会交换理由中明确指向first/second phrase或Text 1/2的表述；原文中事件的first/second顺序不改。

这些理由是助手撰写的草稿，未经过独立人工复核。复用历史C反向N是保持基线数据的一项设计，不将其描述为新增的官方反向标签保证。

两个任务使用不同Prompt：

```text
标签任务：原短Prompt + Text1、Text2 → 标签 + EOS
解释任务：比较语义的Prompt + Text1、Text2 → reason + EOS
```

解释Prompt不提供金标。标签Prompt逐字符复用原版，不把reason放进标签输入。Dev只运行标签任务，不生成或读取理由。

## 训练目标

```python
loss = label_loss + 0.2 * reason_loss
```

label_loss在本批所有标签任务的监督token上取平均，reason_loss在本批可用解释任务的监督token上独立取平均；两者均包含各自EOS。Prompt和padding不计损失。几十个理由token不会直接改变标签损失的分母。

每步仍取16条原C标签样本，再给其中有可用理由的样本单独做解释前向；合并两个损失后只更新一次。每轮220个优化步、5轮1100步，标签行每轮一次、可用理由每轮一次。梯度累积固定为1。本版本没有Pair CE或依据词分类头。

额外解释前向会增加计算和激活显存。沿用本机TRL的chunked_nll，只对监督位置计算分块词表CE，减少完整词表logits的峰值内存；这不改变CE的数学定义。

固定设置：

- Qwen3-0.6B-Base，从基础模型重新初始化LoRA。
- r=16、alpha=32、dropout=0.05、all-linear。
- lr=1e-4、batch=16、5 epochs、bf16。
- 预测batch固定16。
- 每轮同时报告independent与pair-aware，按pair-aware选最佳。
- 只保留最佳checkpoint，并在结束时导出同一最佳权重的best_adapter。

## 运行

训练：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/qwen_lora/reason_sft/train_lora_reason.py
```

默认RUN_NAME为 `run01_reason02`。

- Adapter：`D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_reason/run01_reason02/C/best_adapter`
- 结果：`experiments/qwen_lora/outputs/qwen3_0p6b_reason/run01_reason02/C/`
- `best_summary.csv`：所选同一checkpoint的两种读出。
- `epoch_summary.csv`：每轮两项分数、N与F/B/E正确数、非法原始生成数。
- `run_config.json`：实际Prompt、训练设置、数据与代码哈希。

训练结束后，可右键运行 `evaluate_lora_reason.py`，重新加载best_adapter评估Dev。它只读取模型配置和Dev，不读取理由CSV或Train。

原标签SFT的同代码零权重对照：

```powershell
G:/anaconda3/envs/nlpbase/python.exe G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/qwen_lora/reason_sft/train_lora_reason.py --baseline
```

该对照保存为 `run01_reason02_baseline`，reason权重为0且不做解释前向，保持同样标签数据、Prompt、batch和优化步数。比较它与理由版，才能更清楚地判断新增理由任务的贡献。历史0.899137仅作为既有单种子参考。

## 预测口径

使用原标签Prompt，生成一步并读取F/B/E/N四个候选分数，取四类argmax作为independent；再按原FB/BF/EE三组合相加解码得到pair-aware。单例保留模型结果，不强制N。

另存未经候选限制的 `generated_prediction`、`illegal_generated` 及 `raw_generation_accuracy`。历史C+r16的原始生成均合法时，候选argmax与旧预测一致；若新模型产生非法token，四类候选分数评估与原始生成不等价，需一起报告。

## 已准备的数据和检查

`../outputs/qwen3_0p6b_reason/prepared_C/`（相对qwen_lora目录的输出位置为 `outputs/qwen3_0p6b_reason/prepared_C/`）中包含：

- `train_C_with_reasons.csv`：3517行标签样本与理由来源。
- `reason_task_rows.csv`：2040行实际解释目标。
- `token_audit.csv`：每行两项任务的输入长度与监督token数。
- `data_manifest.json`：来源和Prompt快照。
- `setup_checks.json`：检查结果。
- `real_forward_checks.json`：本机真实0.6B模型的CPU前向检查。

仅更新准备文件可运行 `train_lora_reason.py --prepare-only`。检查可运行 `check_setup.py`。

已检查：原C3517行和475反向N、标签Prompt与历史版本逐字符相同、全部答案/EOS掩码、解释目标掩码、Dev无解释任务、历史pair-aware逐行一致。随机小Qwen3+LoRA上验证了分别归一化、零权重等价、解释梯度到LoRA、无理由批次和Trainer评估路径。真实本地0.6B模型的标签及解释CPU前向均为有限损失。

上述检查没有创建优化器或更新参数，不是训练后的F1结果。本次也没有声称能超过0.899137。

理由作为额外监督的背景可参考[Distilling Step-by-Step，ACL 2023](https://aclanthology.org/2023.findings-acl.507/)。这里是针对原Qwen C配方的适配，不是论文实现的完整复现。
