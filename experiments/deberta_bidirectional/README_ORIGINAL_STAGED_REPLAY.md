原0.938模型：先Row，再同批四分类CE + Pair CE + N回放。

当前定位：这是已有Row CE + 0.5×Pair CE的调度/曝光变体，不是新的训练目标，暂不建议优先重跑。
已有`deberta_joint_pair/train_deberta_joint_pair.py`和`deberta_manual_aug/train_relation_manual.py`的mixed模式，
都已实现配对两行的四类CE、N行四类CE与Pair CE同一步更新。
本文件的差别仅是保留前置完整Row阶段，并在后续87批中完整、按比例回放N。
这些差别尚未被证明能突破已有结果，不能把“同时要求单行与Pair正确”作为新方法的依据。

若以后专门比较该调度，可右键运行`train_original_staged_replay.py`。正式训练由用户运行。
旧的0.938、PAWS及其他训练文件、模型和结果均保留。

每轮流程：

1. 原3290行训练四分类CE，206次更新。
2. 评估同一模型的independent和pair-aware。
3. 第二阶段87次更新。每批16个真实Pair对应32行，按Train类别比例分配约6—7条N；最后一批较小。
4. 一次前向得到本批所有行的四类logits；全部行做四分类CE，配对行额外做三组合Pair CE。
5. `loss = row_loss + 0.5 * pair_loss`，一次反向、一次优化器更新。
6. 再次评估同一模型，记录第二阶段修好/破坏及非N→N数量。

第二阶段直接监督当前Pair两行的真实F/B/E标签；N也计算同一四分类CE。
N从原Train的530行单例取，每轮打乱、各使用一次。不反转N，不引入NN组合，不使用PAWS。
每轮第二阶段完整覆盖2760条配对行和530条N，所以每条原Train行每轮总共参与两次四分类CE。
Pair目标仍是FB/BF/EE，1380组各一次，系数0.5、类别权重[1,1,1.25]、平滑0.05保持原设置。

模型直接使用原DebertaDiCoModel：NLI DeBERTa + CLS四分类 + Token Alignment四分类残差。
不使用联合头、LoRA、词汇分支、SWA或额外保持损失。全部184001800参数训练。
读取0.938的3290行Train快照，从相同NLI初始化重新训练，不读取旧的0.938权重续训。
seed42、lr2e-5、10轮、float32、max_length128、Row类别权重[1,1,1,1.5]。

与旧版的比较边界：

- 原版每轮206+87=293步、6050条编码行；新版仍293步，但6580条编码行，增加530条N。
- 第二阶段增加系数1的同批四分类CE。它不同于旧版系数0.1的随机Row辅助，也不同于只遍历一次Train的普通mixed。
- 每批最多39行，第二阶段一次合批前向。训练dropout和shuffle轨迹会与旧版不同，不承诺逐位复现。
- 原三组合Pair CE不约束N相对分数，允许其漂移，但不必然把F/B/E整体压低。本版本以直接四类监督干预这一现象，不保证Dev两项都提高。

选模按用户的同轮目标：

`selection_score = min(independent_F1 / 0.87, pair_aware_F1 / 0.94)`

只在第二阶段结束后的该分数刷新时保存；相同保留较早轮。只保留一个完整best_model.pth。
0.87/0.94是比较标尺，不是保证达到的成绩。targets_met表示所选同一轮是否同时达标。
这与旧版只按pair-aware选模的协议不同，不能把不同轮次的峰值拼成一个成绩。

显存修改：`DYNAMIC_PADDING=True`，按本批最长有效token裁去右侧padding，正文和标签不变。
Row、第二阶段、Dev/Train评估均使用该方式。另启用编码器梯度检查点，反向时重算部分激活以节省显存。
仍使用float32、每批16组Pair、206+87次更新、原损失系数；未通过减小有效batch修改训练配方。
动态padding会改变训练dropout随机数消耗；不保证与旧版逐位相同。检查点有重算开销，实际速度取决于padding节省量。

默认RUN_NAME：`run05_original_staged_replay_seed42_mem`，保留上一次启动/显存不足的记录。

- 模型：`D:/model/SEAMEVAL/deberta_bidirectional/<RUN_NAME>/best_model.pth`
- 结果：本目录`outputs/<RUN_NAME>/`
- `stage_summary.csv`：每轮两个阶段、两种读出的四行结果。
- `epoch_summary.csv`：只列第二阶段结束后的两种读出。
- `stage_changes.csv`：第二阶段相对Row修好/破坏、各类正确数、非N→N变化。
- `pair_replay_exposure.csv`：第二阶段每批原Train行号、Pair两行ID和N的ID，便于核对是否重复/遗漏。
- `selection_history.csv`、`best_metrics.json`：同轮选模及是否达到目标。
- `best/dev`、`best/train`：重新加载所选权重后的两套评估，不可混淆。
- `reload_check.json`：重载前后预测差异，应为0。

训练结束自动评估。以后独立重载运行`evaluate_original_staged_replay.py`。
下次实验只改训练入口RUN_NAME；评估入口会导入同一RUN_NAME，也支持--model-dir指定已有模型目录。

只检查分词/数据/路径、不加载模型或训练：训练入口加`--prepare-only`。
CPU机制检查入口为`check_original_staged_replay.py`；加`--real`检查真实NLI模型前向/反向及完整权重重载，无优化器参数更新。

2026-10-24已完成检查：

- `--prepare-only`真实3290/660行数据、分词、源文件路径通过，没有创建正式run05目录。
- 原Row阶段逐参数梯度与旧实现一致。
- 真实Train第二阶段87批：1380个Pair、530条N、3290行各一次，无重复或遗漏，最大39行/批。
- 验证Pair CE对N列没有直接梯度，而同批四类CE对非N行和N正例提供方向正确的N列梯度。
- 小模型验证一次反向的梯度等于四类CE + 0.5×三组合CE，保存/重载一致。
- 缓存中的真实184001800参数原模型，在CPU对4组Pair+1条N前向/反向通过；完整encoder权重保存并严格重载后输出一致。
- 三个新Python文件语法检查通过。没有运行正式训练或优化器参数更新，临时检查权重已清理。

2026-10-24显存修改检查：

- 真实Train/Dev最长有效长度分别26/21，原固定128的大部分位置为padding。
- 动态padding及梯度检查点的CPU输出/梯度检查通过；曝光与更新次数仍为原设置。
- RTX 5070 Ti Laptop GPU上检查39行×26 token的一批前向/反向，另外分配与AdamW两份动量等量的显存缓冲。
- 该单批峰值allocated约2.794 GiB，reserved约3.223 GiB；未执行optimizer.step，也未启动正式训练。
- 同一批输入在eval模式下，padding到128与使用26长度的四类logits最大差为0。
- 上述为单批检查，不包含真实优化器step的临时内存，不应当作完整训练峰值。
