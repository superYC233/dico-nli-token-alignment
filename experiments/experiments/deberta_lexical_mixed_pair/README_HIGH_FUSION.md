# 历史高分版本的等权概率融合

运行 `analyze_high_fusion.py`，配置在 `config_high_fusion.py`，输出在 `outputs/fusion02_high_pair_models/`。

本次读取四个历史最佳轮次的Dev原始四类分数，按instance_id对齐；不训练、不重新选择epoch。每个模型先做四类softmax，再等权平均。平均概率的argmax是independent；其对数分数接原三组合pair-aware。单例仍取模型预测，不强制N。

| 成员/方案 | independent W-F1 | pair-aware W-F1 | pair-aware F/B/E正确 | N正确 |
| --- | ---: | ---: | ---: | ---: |
| relation_0938：relation48两阶段epoch2 | 0.547177 | 0.938145 | 518 | 101 |
| scope_0935：scope200两阶段epoch3 | 0.480876 | 0.935493 | 518 | 99 |
| lexical_0930：Lexical run02 epoch3 | 0.468191 | 0.930569 | 516 | 98 |
| original_0924：原版epoch7教师导出 | 0.528004 | 0.924794 | 512 | 98 |
| prob_top2：前两个等权平均 | 0.411614 | **0.939879** | 520 | 100 |
| prob_high3：前三个等权平均 | 0.445081 | 0.936712 | 518 | 100 |
| prob_high4：四个等权平均 | 0.457446 | 0.935238 | 518 | 99 |

以上来自实际执行及官方评分器复核；不存在0.95的结果。最高双模型相对0.938145修好9行、破坏8行，另2行换成另一错误标签，净多对1行。F/B/E多对2行，N少对1行。独立F1下降，所以不能称为单行判断提升。

383个source pair_id成组bootstrap，共5000次。最高双模型pair-aware差值为+0.1734个百分点，95%区间[-1.5043,+1.8268]，包含0。该区间条件于当前模型，不覆盖训练随机性或Dev选模偏差；这次比较属于Dev探索。

成员权重状态：0.938、0.9355、0.9306的best_model.pth存在，best_metrics.json分别确认epoch2、epoch3、epoch3。原0.9248的配置路径epoch_7.pth不存在，本次四模型方案只使用其历史教师导出分数，因此不能声称已具备完整四模型的新输入推理流程。脚本只读取已有预测，没有重新加载权重。

0.938成员是官方3042 + scope200 + relation48，**没有Lexical**；0.9355成员是官方3042 + scope200，也没有Lexical；0.9306才包含Lexical。融合继承这些历史人工增强的数据来源，不能当作纯官方3042训练的实验。

可查看：`summary.csv`、`fixed_broken.csv`、`paired_bootstrap.csv`、`manifest.json`、`*_predictions.csv`及`official_checks/`。CSV中的logit列对融合方案存储的是平均概率的对数，用于保持原pair-aware的相加形式。

## 补算Qwen C+r16

运行 `analyze_high_fusion.py --with-qwen`，输出为 `outputs/fusion03_high_pair_qwen/`，配置在 `config_high_qwen_fusion.py`。沿用原Qwen3-0.6B C+r16的历史预测，不重新训练、也没有生成理由。

| 等权概率平均成员 | independent | pair-aware | 结构后F/B/E正确 | N正确 |
| --- | ---: | ---: | ---: | ---: |
| 0.938 + Qwen | 0.804710 | 0.927206 | 522 | 90 |
| 0.9355 + Qwen | 0.790303 | 0.932186 | 524 | 91 |
| 0.938 + 0.9355 + Qwen | 0.718260 | 0.930799 | 518 | 96 |
| 前三高分DeBERTa + Qwen | 0.536905 | 0.936650 | 520 | 98 |
| 四个高分DeBERTa + Qwen | 0.531459 | 0.936712 | 518 | 100 |

这些方案均经官方评分器复核，都没有超过原0.938145或高分双模型0.939879。0.938 + Qwen相对0.938单模型，结构后F/B/E多对4行、N少对11行，总计净少对7行。

作为错误互补诊断：Qwen自身在0.938版的41条结构后错误里答对16条，但在0.938版原本正确的619条里错41条。两者共同错误25条。这只是事后金标诊断，不是可用于推理的选模型规则，也不能证明Qwen生成理由的质量。

高分双模型的independent为0.411614，原0.938单模型为0.547177。双模型把非N误判N从297行增加到371行；因此其0.939879是结构后分数的小幅提升，不能视为单行能力改善。
