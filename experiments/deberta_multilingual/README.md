# 其他语言：单模型Alignment＋Lexical

基于不加增强、不做集成的英文 **0.9306 Lexical版**，为官方Track 2／3／4提供训练、评估和预测。每个赛道独立微调一个模型，预测时只加载这一个模型，不合并其他成员分数。英文原实验不受影响。

## 先运行哪个

本地的官方多语言CSV已经下载并核对好。先运行[download_model.py](download_model.py)下载模型，再运行[train_deberta_multilingual.py](train_deberta_multilingual.py)训练。两者都可以在IDE中直接运行。

[config.py](config.py)默认：

```python
TRACK = 2       # 2西班牙语，3巴斯克语，4混合语言
RUN_NAME = 'run01'
```

先跑西班牙语保持`TRACK=2`；训练巴斯克语改为`3`，混合赛道改为`4`。不同赛道的模型和结果自动分目录；同一赛道改配置重跑时，再修改`RUN_NAME`。

## 模型和数据

多语言初始化使用[MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)。它的多语言预训练覆盖Basque，后续NLI训练语言包含西班牙语和跨语言样本；其27种NLI训练语言列表不含Basque，所以巴斯克语能力仍需用官方Track 3验证，不能保证达到英文分数。

下载脚本只取一份Safetensors权重和Tokenizer，不下载重复的BIN／ONNX版本，约需580MB下载空间。训练明确使用float32，不照搬原checkpoint的float16存储设置。

| TRACK | 输入 | 官方Train | 官方Dev | 训练用反向Pair数 |
|---|---|---:|---:|---:|
| 2 | 西班牙语→西班牙语 | 3042 | 660 | 1282 |
| 3 | 巴斯克语→巴斯克语 | 3042 | 660 | 1282 |
| 4 | 英／西／巴之间的6种混合语言顺序 | 18252 | 3960 | 7692 |

数据全部来自[官方final_data](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/final_data)。没有自行翻译、没有scope／relation／反向N增强，也没有把其他赛道的Train追加进当前赛道。首次使用时运行[download_data.py](download_data.py)准备官方CSV；文件仍按[data说明](../../data/README.md)放在项目的`data/train`和`data/dev`中。

## 沿用哪些设置，改了什么

恢复的是英文历史`run02_branch_lr`的实际保存配置，而不是后来修改过的英文config：

| 项目 | 设置 |
|---|---|
| 架构 | CLS＋Token Alignment＋20维Lexical修正分支 |
| 训练 | 10轮；每轮先完整Row CE，再完整Real Pair CE |
| 学习率 | 主干、CLS、Alignment为2e-5；Lexical为1e-3 |
| batch／长度 | Row、Pair、Dev均16；max_length=128 |
| Row类别权重F/B/E/N | 1／1／1／1.5 |
| Pair候选权重FB/BF/EE | 1／1／1.25 |
| Pair损失系数／标签平滑 | 0.5／0.05 |
| 精度／seed | float32／42 |
| 选模 | Dev pair-aware Weighted F1；同分保留较早轮 |
| 保存 | 只保留一份最佳完整模型，包含主干、对齐和词汇分支 |

必要的多语言适配有三项：

- 用多语言mDeBERTa替换英文NLI主干，模型计算公式保持原样。
- Lexical按每侧语言使用冠词、否定和数字词表。西班牙语`a`是介词，不当作英文冠词删除；巴斯克语不硬删单词的`a/ak`词尾。数字特征兼容常见的小数逗号写法。这些是粗略词汇线索，不作词义标注或自动改标签。
- 混合赛道同一`pair_id`含多个语言视图，不能整组当成两行。代码为`en-es/original`寻找真实互换的`es-en/flipped`，核对文本和语言；由participant的ID和文本完成配对，不读取Dev金标或参考反向链接来决定预测。

预测仍报告同一权重的independent和pair-aware。pair-aware只比较FB／BF／EE，未配对行保留模型自己的四类预测，不统一判N。

## 文件与运行方式

| 文件 | 用途 |
|---|---|
| [download_model.py](download_model.py) | 下载共用的多语言初始化 |
| [download_data.py](download_data.py) | 下载官方三个赛道CSV，保留原文件与来源hash |
| [config.py](config.py) | TRACK、RUN_NAME、模型路径和训练参数 |
| [train_deberta_multilingual.py](train_deberta_multilingual.py) | 训练当前赛道并保存最佳完整权重 |
| [evaluate_best.py](evaluate_best.py) | 重载最佳权重，预测对应赛道Dev并评分 |
| [predict.py](predict.py) | 预测无标签输入，导出官方两列CSV |
| [check_setup.py](check_setup.py) | 数据分组、语言特征、微型CPU模型和评分格式检查 |

也可在项目根目录运行：

```powershell
python experiments/deberta_multilingual/download_data.py
python experiments/deberta_multilingual/download_model.py
python experiments/deberta_multilingual/train_deberta_multilingual.py --track 2
python experiments/deberta_multilingual/evaluate_best.py --track 2
```

训练另外两个赛道，使用同一个入口：

```powershell
python experiments/deberta_multilingual/train_deberta_multilingual.py --track 3
python experiments/deberta_multilingual/train_deberta_multilingual.py --track 4
```

下载脚本和训练使用同一份`MODEL_PATH`。下载、缓存及微调权重的保存位置在config中配置，模型类不依赖特定计算机的目录结构。

结果位于本目录的`outputs/track编号/RUN_NAME/lexical_seed_42/`，其中：

- `epoch_summary.csv`记录所有轮次的两种成绩。
- `best_summary.csv`是被选中那一轮的两种成绩。
- `epoch_编号/pair_aware/official_predictions.csv`是该轮的官方字段预测。
- `best_reevaluation/`是运行evaluate_best后重新加载权重得到的预测。

无标签预测示例：

```powershell
python experiments/deberta_multilingual/predict.py --track 3 --input data/dev/dico_nli_dev_track3_participant_labeled.csv
```

输入需有`instance_id,pair_id,text1_lang,text2_lang,text1,text2`。即使传入公开Dev文件，predict也会先去掉标签和参考链接；要计算Dev成绩请使用evaluate_best。不同赛道的模型不能交叉当作同一个保存模型使用。

## 已完成的检查

已用真实官方Train／Dev核对三个赛道的反向关系，与reference中的所有反向链接一致；还检查了去掉／篡改标签不改变解码、单例保留模型判断、无标签预测两列格式，以及微型CPU模型前向。检查入口不会下载大权重或训练。

`outputs/setup_check`下的合成评分测试使用人工指定的logits，出现1.0只说明评分代码和格式正确，不是模型成绩。正式训练尚未运行，当前没有多语言Dev实验分数。

完整项目发展过程见[项目README](../../README.md)，原英文版本见[Lexical目录](../deberta_lexical/README.md)。
