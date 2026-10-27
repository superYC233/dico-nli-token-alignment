# 原始DeBERTa-v3-base + 无增强Lexical

本目录对应无增强的 **0.930569 Lexical版**，不是含scope等增强的0.938版。仅把初始化从 `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` 换成 `microsoft/deberta-v3-base`。

训练入口：右键运行 `train_deberta_base_lexical.py`。

## 保持一致

- 官方Train3042行、Dev660行；F881/B881/E802/N478，真实Pair1282组。没有scope、人工768行或复制768行。
- 原模型直接复用：DeBERTa + CLS分类 + Token Alignment + 20维Lexical分支。
- 全参数微调，不冻结、不用LoRA，不加载本项目旧checkpoint续训。
- 原模型lr=2e-5，Lexical分支lr=1e-3，weight_decay=0.01。
- 10轮、FP32、最大长度128、Row batch16、Pair batch16、Dev batch16，seed42。
- Row权重[1,1,1,1.5]；Pair权重[1,1,1.25]，Pair label smoothing=0.05。
- 每轮先完整四分类Row CE，再完整0.5×三组合Pair CE；每轮191+81=272次更新。
- 原FB/BF/EE结构解码；N单例保持模型自己的argmax。
- 每轮同时报告独立与结构成绩，按Dev pair-aware F1保存一份最佳完整权重。最佳权重的两项成绩保存在同一张best_summary.csv中。

原始base的分类任务头从头初始化；微软模型中的预训练任务头不作为DiCo分类头使用。正式训练只加载原始微软主干，没有NLI适配器。

## 本地加载

模型已经在 `D:/BaiduNetdiskDownload/PretrainedModel/models/models--microsoft--deberta-v3-base/`。

代码读取缓存refs/main，定位本地snapshot后加载，避免模型ID触发远程adapter_config查询。主snapshot权重为pytorch_model.bin，所以设置 `use_safetensors=False`；不需要重新下载。

本机原始分词器出现了Mistral regex提示，检查脚本已实际确认全部3042行Train和660行Dev的分词编码与旧NLI版完全相同，因此本实验没有额外修改分词规则。加载报告中的lm_predictions/mask_predictions为原预训练任务头，新的四分类和Lexical头由本项目创建。

## 输出

`config.py` 默认 `RUN_NAME='run01_raw_base'`，复验改RUN_NAME即可。

模型：`D:/model/SEAMEVAL/deberta_base_lexical/run01_raw_base/lexical_seed_42/best_model.pth`。

结果：本目录 `outputs/run01_raw_base/lexical_seed_42/`。

每轮保存预测、分数和混淆矩阵，不保存每轮模型。只有结构F1刷新纪录才替换唯一最佳模型。训练结束后输出相对历史0.930569版的修好/破坏明细。

重新预测：右键 `evaluate_best.py`，加载保存的最佳完整权重。

## 检查

只检查官方数据、分组和特征，不训练：

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_base_lexical/train_deberta_base_lexical.py' --prepare-only
```

运行 `check_raw_base.py`：检查参数/特征/数据与0.930569运行配置一致、两个分词器在全部Train/Dev上的编码一致、Row/Pair循环梯度和随机状态与旧代码一致。加 `--gpu` 会加载原始base并检查真实Row16行和Pair16组前后向，不执行优化器更新，不训练。

## 附件里的复制对照

人工768行的最佳结构F1=0.908421，复制768行=0.910746，差0.2325个百分点；本次人工数据没有显示出更高的Dev结构峰值。两者都低于旧0.938145。这里先放下增强，单独验证初始化选择。
