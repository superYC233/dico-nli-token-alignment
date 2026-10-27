**DeBERTa LoRA：普通mixed训练的适配方式对照**

右键运行train_relation_lora.py。当前默认RUN_NAME='run21_lora_r16_seed7_dropout01_linear_fast'、USE_LORA=True、SEED=7。代码继续放在原deberta_manual_aug目录，旧模型及结果保留。正式训练由你启动。

本次主要对照run19_lora_r16_seed7_dropout01：沿用同一seed7、r=16、dropout=0.1及20轮，学习率从1e-4逐步降至1e-5。run19的固定1e-4所选同一轮Dev为independent 0.839668、pair-aware 0.898373。run20是固定1e-5，不能当作学习率衰减实验。run17是seed42的单次最佳0.858277/0.913362；0.938145仍是不同训练安排下的最高结构分参考。

**这一版做什么**

- 使用原NLI DeBERTa，从同一初始化重新训练，不加载0.938或run13已训练权重。
- 使用原3290行快照：官方3042＋旧scope200＋relation48，没有新增文本。
- 只在编码器query_proj、value_proj加入LoRA：r=16、alpha=32、dropout=0.1、bias=none。
- 原编码器权重全部冻结；LoRA A/B正常训练，CLS四分类头及整个Token Alignment正常训练。
- 使用PEFT的inject_adapter_in_model，保持原DeBERTa的forward和参数路径。注入在CPU的fork_rng里完成，初始化会恢复CPU随机数状态。启用LoRA dropout后，它在训练中会额外消耗随机数；训练数据shuffle仍由独立generator控制。
- LoRA B初始为0，因此初始输出与同一个普通模型一致。已有编码器/任务头dropout沿用原配置；本次只对LoRA新增支路启用0.1的dropout，预测时关闭。

PEFT官方注入说明：https://huggingface.co/docs/peft/developer_guides/low_level_api 。

**训练条件**

普通mixed：每批随机8组，最多16行。所有行做四类Row CE，配对行另做三组合Pair CE，N单例不做Pair CE。每轮3290行各出现一次、1380个可逆组、530个N单例、239步，20轮，float32。

学习率初始1e-4，每次optimizer.step之后调用LinearLR.step，经过20×239=4780步降到1e-5；没有warmup。每轮记录首步、末步和下一步学习率，run_config.json保存完整调度配置。weight_decay=0.01，Row权重[1,1,1,1.5]，Pair权重[1,1,1.25]，Pair系数0.5和平滑0.05。

实际训练改为一次forward、一次backward：loss=row_loss+0.5*pair_loss；没有pair时只用row_loss。与之前分别求梯度再相加的数学目标相同，浮点累加顺序可能不同，不承诺逐字节相同。没有投影、KL、SWA或新词汇分支。这样减少重复反向的计算开销，实际提速需要正式运行计时。

可训练参数和逐项冻结状态保存在parameters.csv。训练期间会打印三组参数量，原编码器应全部为False、LoRA及任务头为True。

初版run14使用r=8、lr=2e-5、10轮；run15改为lr=1e-4、20轮；run16再改为r=16、alpha=32。run17/19启用LoRA dropout=0.1，在seed42/7上收益方向不同，因此不能声称稳定改善或过拟合已解决。当前不再加rank或轮次，检查学习率衰减。原编码器和任务头本来已有dropout。

**选模与文件**

与run13相同，按同轮的min(independent/0.87, pair_aware/0.94)最大选择，相同取早轮，只刷新一份最佳权重。目标数字只用于选模，不参与训练损失。

- 模型：D:/model/SEAMEVAL/deberta_manual_aug/run21_lora_r16_seed7_dropout01_linear_fast/best_model.pth。
- 结果：outputs/run21_lora_r16_seed7_dropout01_linear_fast。
- epoch_summary.csv及selection_history.csv：完整逐轮同一模型的两种成绩。
- best_metrics.json及best_summary.csv：所选同一轮。
- best/dev、best/train：恢复最佳权重后分别预测，Train分数不能当Dev分数。
- batch_plan.csv：每批使用的原Train行号，便于核对与mixed批次顺序。
- comparison_with_mixed.csv：仍与run13_gradient_mixed所选模型比较，不能把这个文件当作调度消融。调度对照需与run19所选同一轮的两项成绩比较。
- reload_check.json：训练末尾重新加载最佳权重后，应与当轮预测差异0。

best_model.pth只保存可训练参数：LoRA A/B＋全部分类/对齐头，不重复保存冻结的NLI底座。恢复还需要同目录run_config.json、tokenizer以及本地原NLI模型缓存。不要用旧的完整state_dict加载脚本或PeftModel.from_pretrained直接加载这个文件。

训练结束会自动恢复最佳参数、评估Dev和Train；也可单独右键evaluate_relation_lora.py，默认恢复run21_lora_r16_seed7_dropout01_linear_fast。换RUN_NAME后同步修改该评估文件顶部MODEL_DIR。

若以后需要同代码的全参数复验，可改USE_LORA=False并更换RUN_NAME。它会保留普通mixed训练，但保存文件将较大；本次先不重复训练已有对照。

**检查，不训练**

```powershell
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/train_relation_lora.py' --prepare-only
& 'G:/anaconda3/envs/nlpbase/python.exe' -X utf8 'G:/pycode/SemEval/Task-2-DiCo-NLI-main/experiments/deberta_manual_aug/check_relation_lora.py'
```

自检使用随机初始化的小DeBERTa，验证初始输出与RNG不变、冻结范围、LoRA梯度、保存非零adapter及所有任务头后精确恢复。不会训练或下载模型。

加--real会额外读取已缓存的真实NLI DeBERTa，在CPU对原Train中的一个可逆组检查前向及两项梯度；不创建优化器，不更新参数，不读取Dev。
