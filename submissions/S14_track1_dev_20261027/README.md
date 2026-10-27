# S14 英文开发阶段提交包

准备日期：2026-10-27。本目录没有执行线上提交，最后由你上传。

## 上传哪个文件

上传 **[submission_S14_track1_dev.zip](submission_S14_track1_dev.zip)**，不用上传模型权重。

压缩包根目录只有 `track1_predictions.csv`，没有外层文件夹。CSV包含660行预测，仅有 `instance_id,label` 两列，使用官方完整标签。

这是本次S14重新加载模型后生成的 **pair-aware融合预测**，不是某个成员的预测，也不是历史CSV搜索得到的0.965418版本。

## 这次核对的结果

| 指标 | 本地官方评分器结果 |
|---|---:|
| Weighted F1 | 0.9623808537672834 |
| SoftCons | 1.0 |
| HardCons | 0.9638989169675091 |

已检查：660个ID与官方Dev模板完全一致、无重复或缺失ID、标签合法、压缩包可读取、打包后的预测与原预测一致。只按官方模板调整了行顺序，没有修改预测标签。金标仅用于本地评分。

原始预测：[S14本次pair-aware输出](../../experiments/ensemble_selected/outputs/S14/run01_retrain/pair_aware/official_predictions.csv)。验证结果见 [local_validation/scores.json](local_validation/scores.json)，文件哈希和成员信息见 [submission_manifest.json](submission_manifest.json)。

## 怎么提交

1. 打开[官方CodaBench开发阶段](https://www.codabench.org/competitions/18038/)，登录你的账号。
2. 如果尚未参加，先报名并阅读、接受页面上的规则。
3. 进入开发阶段的提交入口，提交英文Track 1预测，选择本目录的 `submission_S14_track1_dev.zip`。
4. 如果页面要求填写名称，可填 `S14 hierarchical ensemble - Track 1 Dev`。
5. 提交后等待评分完成，再查看自己的成绩和Results排行榜；如果需要手动选择公开成绩，按页面提示操作。

按钮名称、提交次数和是否需要手动公开，以当前平台页面为准。本次网页工具未能读取CodaBench提交页面；包内文件名沿用项目此前正确包的 `track1_predictions.csv` 命名。官方通用评分器的默认名是 `predictions.csv`，而平台可另外配置文件名。如果页面提供的当前示例包规定了不同名字，应以当前示例为准，不要混用两个名字或同时放入两个CSV。

目前这是 **Dev开发阶段成绩**，不是隐藏Test最终成绩。[官方参与流程](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI#how-to-participate)说明开发阶段用于实验与格式测试，正式排名另按评估阶段和系统论文要求进行。

## 需要填写方法说明时

可用以下英文简述：

> S14 is a six-member DeBERTa-v3-base ensemble initialized from MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli. Members use the official 3,042 training rows, with selected members additionally using 200 synthetic scope examples and 48 synthetic relation examples. One member also uses rationale/evidence annotations on 100 training groups. A designated member provides the N probability, while five members equally average their conditional F/B/E probabilities. Pair-aware decoding restricts two-row groups to FB/BF/EE; singleton predictions remain model-based. Checkpoints and ensemble selection were developed using Dev results. Augmentation design was informed by Dev error analysis; Dev examples were not appended as training rows. No hidden test labels were used.

这段说明包含人工增强和Dev参与开发的事实，不应改写成“只使用官方3042行”或“完全没有参考Dev”。本地审计发现官方Train与Dev有7个规范化重复文本对；scope200、relation48新增部分均无重复Dev整对，scope200有两个相同单侧短语。完整依据见[文本重合核对报告](../../docs/DATA_AUGMENTATION_DEV_OVERLAP_AUDIT.md)。
