# 原0.9306单模型：先检查已有强教师的Train分布

说明：`python`指已安装项目依赖的解释器。示例中的`model/`、`pretrained_models/`和`model_cache/`为通用目录，实际保存与缓存位置以配置项为准。

**现在先运行 [export_teacher_scores.py](export_teacher_scores.py)。本目录当前只提供导出和诊断，不训练学生。** 不用重跑S16、重训五个教师，也不用再跑source_aux。

```powershell
python 'experiments/deberta_lexical_distill/export_teacher_scores.py'
```

## 这一步在做什么

```text
官方Train3042行
  → 依次加载M017、M030、M046、M082、M087现成权重
  → 每个模型只预测一遍Train，保存原始分数
  → 每个模型在F/B/E内归一化，五模型等权平均
  → 为Train真实双行组构造FB/BF/EE概率
  → 输出T=1/2/4的分布、教师错误和置信度
```

每次只把一个教师放到GPU，完成后释放。不加载M027，不采用S16的N门，也不为478条N生成蒸馏目标。原始模型只读，程序没有optimizer或训练调用。

五位教师全部使用官方3042行文本，分别是特征拼接版、N权重1.8的词汇版、反向KL版、large混合版、冻结底6层Row版。权重路径固定取自S16实际采用的[清单](../ensemble_selected/outputs/S16/run02_pair_select/run_config.json)，不重新搜索或选择最佳轮。保存时的选模规则和最佳epoch会如实记录，不能因为清单写pair-aware就把实际independent选模改写成pair-aware。

使用已保存的tokenizer、backbone配置及完整任务权重，不依赖已经删除的原预训练快照。M017使用自己的特征拼接模型，不能用普通Lexical模型替代。M082保持原saved bf16推理方式，其余成员沿用各自原设置。

## 输出在哪里

默认 `outputs/run01/teacher_train/`：

- `teacher_check.json`：五个实际权重路径、配方及保存轮次。
- `members/Mxxx/train_logits.csv`：每位教师在3042行Train上的原始四类logits；N列仅保留原始记录，不进入蒸馏目标。
- `fbe_row_targets.csv`：2564条真实F/B/E行的三类条件概率。
- `pair_train_targets.csv`：1282个真实配对组，含正反行instance_id、金标组合、三组合分数及T=1/2/4软概率。
- `teacher_pair_errors.csv`：教师在Train上选错的组，不偷偷用金标覆盖其目标。
- `teacher_diagnostics.csv`：各教师及融合教师的条件三类／组合准确率、最大概率、熵、金标概率和高置信错误。
- `completed.json`：输入与权重指纹、错误数量、成员分歧组数及非OOF说明。

**跑完先发最后的Train诊断表。** 要看教师错误是否合理、不同成员是否提供不同软偏好，以及升温后的分布。T=1接近one-hot不能直接判定“没有信息”；升温让概率变软也不等于一定能提高Dev分数。本次诊断不是Dev成绩，更不是蒸馏后的成绩。

成员分数按训练文件、权重SHA-256、配置、tokenizer、模型代码和导出代码核对后缓存复用。中断后运行同一入口即可复用匹配缓存；`--refresh`强制重新预测。模型从不被复制或覆盖。

只核对路径与Train数据、不加载大权重：

```powershell
python 'experiments/deberta_lexical_distill/export_teacher_scores.py' --prepare-only
```

GPU不可用时默认明确停止；可显式传`--device cpu`做CPU推理，但large教师会慢。权重若缺失会列出路径并停止，不会自动重训。

## 数学口径

先固定每成员T=1的F/B/E softmax，再等权平均得到每行q。对正反行0、1：

```text
uFB = log q0(F) + log q1(B)
uBF = log q0(B) + log q1(F)
uEE = log q0(E) + log q1(E)
teacher_T = softmax([uFB,uBF,uEE] / T)
```

温度加在最终三组合分数上，不在每个成员融合前分别升温，因而T=1/2/4不改变教师argmax。保存正反行ID是为了后续与学生PairDataset正确对齐，不能只假定组内行序。

N门因子对同一组的三个候选相同，条件组合选择不需要N门。这使目标保留S16的F/B/E优势，而不要求学生学习它的N偏置。

## 后续才做什么

诊断确认后，再增加一个学生训练入口：恢复原0.9306的Row/Pair配方，以官方金标和教师组合分布训练一个DeBERTa＋Alignment＋Lexical学生。最终预测只加载该学生。当前不启动这一步，也没有承诺一定达到0.94。

教师都在这些Train样本上训练过，这些目标是in-sample，不是OOF；教师和S16成员选择也受Dev开发影响，需要披露。导出过程不读取Dev样本／标签或历史Dev预测CSV生成训练目标，不向Train加入Dev行，不新增文本、不使用单例强制N。

[check_setup.py](check_setup.py)只用CPU小随机模型和模拟分数检查方向、温度、N排除、三种架构重载及缓存；真实教师仅核对文件存在和配置，不做真实GPU预测或正式训练。
