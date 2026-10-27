# ModernBERT-base-nli：复用0.9248配方

1. 运行download_model.py，下载到D:/LLM/model/ModernBERT-base-nli。
2. 运行train_modernbert.py，默认关闭Lexical，使用官方Train3042行和Dev660行。
3. 训练后运行evaluate_best.py，加载best_model.pth重新评估。

正式模型为tasksource/ModernBERT-base-nli。作者说明其已在MNLI、ANLI、SICK、WANLI等任务微调：
https://huggingface.co/tasksource/ModernBERT-base-nli

本实验从0.9248版的模块化代码复制：原CLS四分类头 + Token Alignment，
每轮先全部Row CE，再0.5×三组合Pair CE；预测采用原pair-aware，单例由模型判断。
Row权重[1,1,1,1.5]、Pair权重[1,1,1.25]、Pair smoothing0.05，
AdamW lr2e-5 / weight decay0.01，batch16，10轮，float32，全参数训练。
不加人工数据、不反转N、不加载我们旧的DiCo权重、不附加reason。

改动是编码器及对应tokenizer换成ModernBERT。其SDPA后端使用PyTorch，不依赖FlashAttention安装。
预训练三分类头不复用，保留0.9248的自建四分类头；这也意味着不使用原模型的mean-pooling分类头。
不能承诺比0.9248更高。仓库已有ModernBERT-large旧实验，这次是base和指定旧配方。

每轮报告同一模型的independent及pair-aware，按pair-aware选最佳，只保留一个best_model.pth。
输出模型位于D:/model/SEAMEVAL/modernbert_nli_base/run01_original_recipe/baseline_seed_42。
CSV位于本目录outputs，保留logit_F/B/E/N，便于后续按instance_id与已有模型融合。

后续Lexical可直接运行train_modernbert.py --lexical；结果在lexical_seed_42子目录，不覆盖baseline。
参数都在config.py，复制的模型/数据/评估文件均在本目录，原实验代码不修改。
check_setup.py只运行CPU随机小模型和数据检查，不运行真实模型训练。
