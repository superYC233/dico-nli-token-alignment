# Qwen LoRA 文件分类

## 当前采用的主线

**Qwen3-0.6B-Base + C 训练集 + LoRA r=16 / alpha=32 + 旧短 Prompt + 预测时 pair-aware。**

按本次整理要求，继续把这套方案作为主线。R-Drop 等其他实验保留作对照。

| 项目 | 当前主线 |
| --- | --- |
| 训练入口 | `train_lora_abc.py` |
| 预测和评分入口 | `evaluate_lora_pair.py` |
| 数据构造 | `data_variants.py` 中的 `build_train_dataset(..., mode='C')` |
| Prompt | `data_utils.py` 中的旧短 Prompt |
| 训练目标 | 标准全词表 SFT，completion-only，包含 EOS；不加 Pair CE |
| 训练配置 | seed=42，lr=1e-4，5 epochs，batch=16，梯度累积=1 |
| 选模 | 每轮 Dev pair-aware W-F1，保留最佳 Adapter |
| 预测配置 | batch=16，先独立预测，再按 pair_id 在 FB/BF/EE 中联合选答案 |
| 已有最佳 Adapter | `D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_abc/run04_r16/C/best_adapter` |
| 已有结果 | `outputs/qwen3_0p6b_abc/run04_r16/C/` |
| 独立预测 W-F1 | 0.814028059 |
| pair-aware W-F1 | **0.899136759** |
| N 正确数 / 成对行正确数 | 80/106；514/554 |

这里的 pair-aware 是**预测后的答案调整**，不表示训练时使用了 Pair 损失。
主线 C 由你当前的简洁构造函数生成，3042 条原始数据加 475 条反向 N，共 3517 条。
旧消融脚本中的 C 曾使用 472 条来源，两者不能视为完全相同的数据版本。

### 平时只需要看这些根目录文件

| 文件 | 用途 |
| --- | --- |
| `train_lora_abc.py` | 主线训练，默认 C+r16 |
| `evaluate_lora_pair.py` | 加载已有主线 Adapter，预测并比较 independent / pair-aware |
| `data_utils.py` | 数据读取、旧短 Prompt、SFT 数据、tokenizer |
| `data_variants.py` | 你自己实现的 A/B/C 数据构造 |
| `best_f1_callback.py` | 每轮计算 pair-aware F1，供 Trainer 选模 |
| `compare_decoding.py` | 公共函数 `evaluate_with_scores`，导出四候选分数 |
| `baseline.py` | 公共预测和官方评分函数 |
| `analyze_decoding.py` | 公共结构解码、指标分析；部分 3B 脚本也引用它，因此保留原位置 |
| `experiment_paths.py` | 子目录脚本的路径兼容工具，一般不用修改 |

`evaluate_lora_pair.py` 仍默认加载上面的 run04 主线模型。
今后重新训练时，`train_lora_abc.py` 保存到新的
`D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_main/run01/C/`，避免覆盖已有主线和 run06 长 Prompt 记录。
如果要评估新训练的模型，再修改预测入口的 `adapter_path` 和结果目录。

## 历史实验与工具

| 子目录 | 内容 | 文件 |
| --- | --- | --- |
| `analysis/` | 数据和错误分析 | `check_data.py`、`check_train.py`、`analyze_errors.py`、`analyze_train.py`、`diagnose_pairs.py` |
| `ablation/` | 早期 A–E 数据消融 | `train_n_ablation.py`、`evaluate_abcde_pair_aware.py` |
| `pair_training/` | 训练期 Pair 损失对照 | `train_pair.py`、`train_real_pair_ce.py`、`train_row4ce_pairce.py`、`train_bcde_real_pair_ce.py`、`evaluate_epochs.py`、`evaluate_pair_epochs.py` |
| `sixclass/` | 六类监督 | `prepare_sixclass_data.py`、`sixclass_data_utils.py`、`train_lora_sixclass.py`、`evaluate_lora_sixclass.py`、`analyze_sixclass.py` |
| `binary/` | 双向二分类实验 | `binary_data_utils.py`、`train_lora_binary.py`、`evaluate_lora_binary.py` |
| `rdrop/` | R-Drop 实验 | `train_lora_rdrop.py` |
| `teacher/` | 教师分数导出，尚未蒸馏训练 | `export_teacher_scores.py` |
| `token_comparison/` | 新的结构实验：简单比较头 / token 对齐比较头 | `train_token_comparison.py`、`evaluate_token_comparison.py`，其余模块见该目录 README |
| `retrieval_prompt/` | 原 C+r16 的检索示例 Prompt 实验，不重新训练 | 右键运行 `run_all.py`，比较无示例 / 随机 2 例 / 检索 2 例 |
| `contrastive_sft/` | 原 C+r16 SFT + 文本对级监督对比损失 | `train_lora_contrastive.py` 训练，`evaluate_lora_contrastive.py` 预测 |
| `legacy/` | 最初的训练、重采样和 Prompt 备份 | `train_lora.py`、`evaluate_lora.py`、`train_hard.py`、`train_random.py`、`build_prompt_original.py` |
| `outputs/` | 所有已有预测、评分与分析文件 | 保持原目录结构 |

### 几个容易混淆的版本

| 版本 | 训练时做什么 | 与主线关系 |
| --- | --- | --- |
| 主线 C+r16 | 全词表 SFT | 当前采用 |
| `train_pair.py` | SFT + 合法组合概率质量约束 | 训练实验，不等于预测期 pair-aware |
| `train_real_pair_ce.py` | SFT + 三组合真实 Pair CE | 训练实验，只比较 FB/BF/EE 的相对分数 |
| `train_row4ce_pairce.py` | 四类 CE + 三组合 Pair CE | 同时改变了单条样本目标；不是单独的四类 CE 对照 |
| 六类 | F/B/E + SIMI/REL/OPPO，预测时把后三类合并为 N | 单独实验 |
| 二分类 | 两个方向分别判断是否蕴含，再组合四类概率 | 单独实验 |
| R-Drop | 两次 SFT 的平均值 + 答案位置四类分布的对称 KL | 单独实验；最佳 W-F1=0.900674933，但 N 正确数 76，比主线少 4 条 |

### 已有参考结果

| 实验 | pair-aware W-F1 | N 正确数 |
| --- | ---: | ---: |
| 当前采用：C+r16，旧短 Prompt | **0.899137** | **80** |
| 六类监督 | 0.898809 | 78 |
| 长 Prompt：run06_paper_prompt_checked | 0.897178 | 本表未统计 |
| R-Drop | 0.900675 | 76 |

这些是已经反复用于开发的 Dev 上的单次运行记录，不能据此认定微小分差具有稳定性。

## 如何运行

主线仍可以直接在 PyCharm 右键运行根目录的训练或预测入口。
分类后的历史脚本也可直接右键运行；路径工具会让它们继续读取原来的公共代码和 outputs。

从项目根目录运行的例子：

```powershell
G:\anaconda3\envs\nlpbase\python.exe experiments\qwen_lora\evaluate_lora_pair.py
G:\anaconda3\envs\nlpbase\python.exe experiments\qwen_lora\analysis\check_data.py
```

历史脚本保留着当时的默认模型路径和配置，运行前仍需核对；本次归类没有把它们重新训练。
整理前的 Python 源码备份在 `legacy/source_before_organization.zip`，文件迁移清单在 `organization_manifest.json`。
