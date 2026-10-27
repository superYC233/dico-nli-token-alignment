# DiCo-NLI：论文阅读与自己编写 Qwen LoRA 的路线

整理日期：2026-10-17。

这份说明用于自己学习和编写一个新的 Qwen 实验。现有 DeBERTa 最终方案仍保留：DeBERTa-v3-base NLI + Token Alignment + Real Pair CE + Pair-aware inference，官方 DEV Weighted F1 为 0.9247935854。

本次仅核对文件、分词器、硬件与文献；没有下载模型、安装依赖、加载 Qwen 大模型权重或启动训练。这里的 Qwen 配置是待验证的起点，不是新的实验结果，也不保证超过 DeBERTa。

## 一、先把任务说准确

自然语言推理一般称为 NLI，文本蕴含也常称为 RTE。我们的任务更具体：判断两个短语之间的信息包含关系，并区分方向。

“A 能推出 B”表示：在通常语义解释下，满足 A 的情况也满足 B。它不等于两个短语看起来很相似。

| A 能推出 B | B 能推出 A | 标签 | 直观例子（A / B） |
|---|---|---|---|
| 能 | 不能 | F：FORWARD_ENTAILMENT | train station / station |
| 不能 | 能 | B：BACKWARD_ENTAILMENT | station / train station |
| 能 | 能 | E：EQUIVALENCE | sofa / couch |
| 不能 | 不能 | N：NEGATIVE_OTHER | red car / blue car |

这些例子用于解释，不会自动加入训练数据。具体短语的判断仍需考虑词义和任务标注规范，不能根据长短或词语包含直接定标签。

特别注意：N 不只表示完全无关。两个短语可能语义相关，但都不能推出对方。也不能把普通 NLI 的 neutral 直接当成我们的 N：例如 animal / cat 在普通单向 NLI 中可以是 neutral，在这里却是 B。

上面的两个“能否推出”是理解标签的办法。第一版 Qwen 仍然直接输出 F/B/E/N，不拆成两个二分类模型。

## 二、最相关的四篇方法论文

### 1. Adding Semantics to Data-Driven Paraphrasing

Ellie Pavlick 等，ACL-IJCNLP 2015。

- [论文页面](https://aclanthology.org/P15-1146/)
- [PDF](https://aclanthology.org/P15-1146.pdf)
- 配套资源论文：[PPDB 2.0](https://aclanthology.org/P15-2070/)

**为什么先看它：** 研究对象就是短语对，区分等价、方向性蕴含、排斥以及其他关系，与我们的 E/F/B 边界非常接近。

**具体做法：** 给短语对收集细粒度关系标注，结合词汇、分布、句法路径和双语翻译等特征学习关系分类器，再把预测关系作为推理系统的知识。

**能学到什么：** “相似”不等于“等价”；要研究哪些证据支持同义，哪些证据支持信息包含，哪些只是相关。不同信号擅长判断的关系可能不同。

**阅读重点：** 关系定义、标注表格、特征分类、错误分析。先不纠结所有特征公式。

**用于本项目的限制：** PPDB 的大量标签是自动赋予的，需要检查质量、方向和任务定义。论文与本任务的标签不完全一致，不能直接整库导入，也不能据此保证性能提升。

### 2. Modeling Semantic Containment and Exclusion in Natural Language Inference

Bill MacCartney、Christopher D. Manning，COLING 2008。

- [论文页面](https://aclanthology.org/C08-1066/)
- [PDF](https://aclanthology.org/C08-1066.pdf)

**为什么相关：** 它直接讨论等价、正向蕴含、反向蕴含、排斥和独立等关系。虽然年代早，但问题定义与我们高度相关。

**具体做法：** 将文本之间的变化拆成局部编辑，预测局部语义关系，根据语法和上下文传播关系，再组合出整体判断。

**能学到什么：** 词之间的关系需要放进上下文解释。比如 cat 到 animal 的概念扩大，在否定或某些量词环境里，蕴含方向可能改变。

**阅读重点：** 关系表、局部变化如何影响整体、上下文如何改变方向。七种关系不必一开始全部背下来。

**用于本项目的限制：** 当前短语数据并不都含复杂句法或量词，完整解析器与逻辑系统未必适合直接加入。先理解机制。

### 3. Enhanced LSTM for Natural Language Inference（ESIM）

Qian Chen 等，ACL 2017。

- [论文页面](https://aclanthology.org/P17-1152/)
- [PDF](https://aclanthology.org/P17-1152.pdf)

**为什么相关：** 它能帮你把学过的 BiLSTM、Attention 与当前 Token Alignment 联系起来。

**具体做法：** 编码文本，进行软对齐，拼接原表示、对齐表示、差和逐元素乘积，再进行推理组合、平均/最大池化与分类。

**能学到什么：** 我们现有 Token Alignment 分支与这类“对齐后比较”的方法有明确联系。可以对照图 1 和第 3 节，把每个组件对应到自己的代码。

**用于本项目的限制：** ESIM 的句子级三分类与我们的短语四分类不同。DeBERTa 本身已经提供上下文表示；不能据此断言再加 BiLSTM 就会更好。局部 Conv 等历史尝试也不需要直接重跑。

### 4. Neuro-symbolic Natural Logic with Introspective Revision for Natural Language Inference

Yufei Feng 等，TACL 2022。

- [论文页面](https://aclanthology.org/2022.tacl-1.14/)
- [PDF](https://aclanthology.org/2022.tacl-1.14.pdf)
- [作者代码 NS-NLI](https://github.com/feng-yufei/NS-NLI)

**为什么相关：** 它明确建模局部自然逻辑关系，再组合成整体结论，研究“预测对了但理由错了”的问题。

**具体做法：** 神经网络预测局部关系，符号程序执行关系组合；通过策略梯度与中间步骤修订学习推理路径，并借助外部知识减少虚假推理。

**能学到什么：** 可以研究有明确关系含义的中间表示，而不只用一个相似度分数概括所有差异。

**用于本项目的限制：** 它主要评估普通三分类 NLI，不是即插即用的 DiCo-NLI 模型。其训练中还利用了普通 NLI 数据中等价较少的假设，这不适合直接搬到 E 占比很高的本任务。暂不复现整套强化学习系统。

### 怎么读，才不容易看完又忘

按 1 → 3 → 2 → 4 的顺序。前两篇先读摘要、方法图、关系表和错误分析，后两篇逐步深入。

每篇只先回答五个问题：

1. 输入是什么，输出是什么？
2. 方向如何表示？等价如何表示？
3. 方法依靠什么监督或知识？
4. 哪一部分与我们当前代码相同或不同？
5. 如果借鉴，只改变哪个变量，如何验证？

不要直接比较论文在其他数据集上的分数与我们的 0.9248。这几篇是根据问题和方法相关性选择的，不是当下排行榜排名。

## 三、Qwen 实验先明确目标

这次由你自己编写和解释整个流程：

```text
官方 TRAIN 的 A、B 和金标
    ↓
固定的任务说明 + 当前 A/B
    ↓
Qwen2.5-3B-Instruct + 可训练的 LoRA
    ↓
学习输出一个字母：F / B / E / N
    ↓
在官方 DEV 上评估，再加入 Pair 方法做对照
```

LoRA 冻结原模型权重，在选定线性层上训练小矩阵。可以理解为让模型学习一个低秩权重更新，而不是替换成一个普通四分类输出层。参考 [LoRA 原论文](https://arxiv.org/abs/2106.09685) 与 [PEFT 官方入门](https://huggingface.co/docs/peft/en/quicktour)。

原来 DeBERTa 的 Token Alignment 不会自动出现在 Qwen 中。第一版也不额外加入它，先建立可解释、能保存和恢复的 Qwen LoRA 基线。

历史聊天中原 Qwen + Real Pair CE 有过 0.913483，未超过 DeBERTa；那是历史运行，不是这次重新核验的结果。现在的价值是自己掌握流程，并通过明确消融判断后续变化。

## 四、已经核对的本机条件

2026-10-17 的只读检查结果：

- GPU：RTX 5070 Ti Laptop GPU，总显存约 12GB。
- 模型目录：`D:\BaiduNetdiskDownload\PretrainedModel\models\Qwen2.5-3B-Instruct`，模型分片和分词器文件均存在；本次只实际加载了分词器，没有校验权重能否完整加载。
- Python：使用 `G:\anaconda3\envs\nlpbase\python.exe`。
- 已安装 torch 2.11.0+cu128、transformers 5.17.0、peft 0.21.2、trl 1.13.0、datasets 4.8.5、accelerate 1.15.0。本次没有升级或安装包。
- F/B/E/N 在本地分词器中均是单 token，分别是 37/33/36/45；编写程序时仍动态获取，不硬编码这些编号。
- 已确认示例的训练对话与推理 prompt 前缀完全相同。

先尝试普通 BF16 LoRA、短输入、梯度检查点、小 batch。实际能否装下仍要用一个训练 batch 核对。若显存不足，再考虑 QLoRA；量化主要减少资源需求，不能保证准确率更高。

## 五、分七步自己写

建议新建你自己的 `qwen_dico_my_lora.py`。前期一个文件分成清楚的函数即可，不急着拆很多模块。下面是编写顺序，不要求一次写完。

### 第 1 步：读数据、定义标签、生成 messages

你自己实现：

```python
def load_data():
    ...

def build_messages(text1, text2, answer=None):
    ...
```

官方 TRAIN 为 3042 行：F 881、B 881、E 802、N 478。DEV 为 660 行。保持官方 TRAIN 原样，不加样本、不删除、不用 DEV 生成训练消息或挑选示例。

每条训练消息包含 system 的规则、user 的两个短语、assistant 的金标字母。预测时不放 assistant 金标。instance_id 与 pair_id 用于关联和评估，不作为模型判断标签的输入。

### 第 2 步：分词，并检查哪些 token 参与损失

你自己实现：

```python
def encode_example(messages, tokenizer):
    ...

def collate_batch(examples):
    ...
```

使用 Qwen 的 `apply_chat_template`。训练消息带正确答案；推理消息只带问题和 assistant 起始标记。参考 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。

损失只监督 assistant 的答案及结束标记；system/user 和 padding 对应的 labels 置为 -100。模型负责 causal shift，不要再把 labels 手动错移一次。

必须打印一条样本的“参与监督的文本”，确认不是把整段问题当答案训练。必须检查截断后仍保留完整问题和答案，不因最大长度把答案截掉。

本地 Qwen2.5 模板没有 Jinja generation 块。因此不能只照抄 `assistant_only_loss=True` 就假定得到正确掩码。学习版建议先自己构造 labels。若以后使用 TRL，可选 prompt/completion 格式并核对 completion mask。官方说明见 [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer)。

如果希望 `apply_chat_template` 返回 token 列表，明确传 `return_dict=False`；本机版本的默认返回形式容易让初学者把字典长度误当 token 数。

### 第 3 步：加载模型与挂载 LoRA

你自己实现：

```python
def build_model():
    ...
```

使用 AutoModelForCausalLM、AutoTokenizer、LoraConfig、get_peft_model；任务类型用 CAUSAL_LM。先打印可训练参数，确认原权重冻结、LoRA 参数参与训练。PEFT 官方入门已给出这一基本流程。

下面是一个待验证的起始配置，不是最佳参数：

| 项目 | 起始选择 |
|---|---|
| 权重精度 | BF16 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| LoRA 位置 | q_proj、k_proj、v_proj、o_proj |
| 微批大小 / 梯度累积 | 1 / 16 |
| 最大输入长度 | 先检查完整数据长度，再从 256 起步；不足时调大 |
| 学习率 | 1e-4 |
| 正式训练轮数 | 先设 3，每轮评估 |
| 随机种子 | 42 |
| 显存设置 | gradient checkpointing 开启、训练时 use_cache=False |

这些参数先固定下来。不要为了追一次分数，同时改提示词、学习率、rank 和损失。

### 第 4 步：先检查一个 batch，再短跑约 20 次更新

你自己实现普通的前向、loss、反向传播和优化器更新，或在已经理解这些步骤之后接入 Trainer。

先确认：loss 有限、监督位置正确、LoRA 梯度非空、padding 被忽略。梯度累积每 16 个微批才更新一次参数；单个微批仍然计算梯度。

这个短跑只检查实现是否正确，不作为最终性能。正式比较从相同基础模型和新的 LoRA 初始化开始，避免无意多训练了这 20 步。

### 第 5 步：完成普通 LoRA 四分类基线

你自己实现：

```python
def train_one_epoch(model, loader, optimizer):
    ...

def predict_label_scores(model, tokenizer, data):
    ...

def evaluate(gold_labels, scores):
    ...
```

训练采用只监督答案的语言模型交叉熵。这是对整个词表学习正确答案，不等同于只把 F/B/E/N 四个 logits 送进交叉熵；后者可以是另外一个实验，第一版不要混用。

正式评价时，从答案首位置取 F/B/E/N 四个 token 的分数并做确定性分类。先用单条输入把位置核对清楚，批量时取每条问题的最后有效 token 对应的 next-token logits，不能误取 padding 位置。

这样能稳定输出四种合法标签；同时检查几条自由生成的输出，确认模型确实学会按格式作答。训练每轮评估 DEV，保存最佳 adapter、分词器、配置和带 instance_id 的预测。

先看 Raw Weighted F1、各类召回和错误数量，不加入 N 权重、检索案例、辅助任务或新数据。

### 第 6 步：给同一个 checkpoint 加 Pair-aware 解码

这一步不训练，仅比较同一份原始分数在解码前后的结果。

设同一真实反向 Pair 的第一条为 (A,B)，第二条为 (B,A)，两条的四分类分数为 s1、s2：

```text
FB = s1[F] + s2[B]
BF = s1[B] + s2[F]
EE = s1[E] + s2[E]
```

对符合官方反向 Pair 协议的组选择最高分组合。核对两条文本确实交换，而不是随意把相似样本配起来；不依靠 DEV 标签寻找组合。

单条样本继续按四分类分数预测，绝不因为没有配对就判 N。记录 Raw 与 Pair-aware 两套结果，SoftCons=1 也不能代替语义准确率。

### 第 7 步：最后才添加 Real Pair CE

你自己实现：

```python
def build_real_pairs(training):
    ...

def compute_pair_loss(scores1, scores2, pair_targets):
    ...
```

官方 TRAIN 中有 1282 组真实正反 Pair。检查标签组合符合 FB/BF/EE，再用上面三个联合分数计算交叉熵；目标索引由当前两条样本的真实标签确定，不能假定第一条一定是 F。

直观理解：

```text
普通训练：这一条 (A,B) 应该回答 F。
Pair CE：这两条问题的答案组合应该是 (F,B)。
```

Pair CE 改变的是训练目标，不是给 Qwen 新增一个神秘的 Pair 网络，也不是仅仅改变 CSV 排序。

可以在包含真实 Pair 和其他样本的批次中同时计算行级损失与 Pair 损失；其他样本继续贡献行级损失。使用 `loss = row_loss + 0.5 * pair_loss` 作为待测起点。没有 Pair 的批次跳过 Pair 项，不能产生空集合 NaN。

与普通 LoRA 对照时，让两版使用同一批次安排和更新预算，只有 Pair 项开关不同。不同时增加 EE/N 权重或 label smoothing。

第一轮学习实验并不要求照搬历史“先整轮行级、再整轮 Pair”的训练调度。若以后比较两种调度，需要单独做实验，清楚记录训练更新次数。

## 六、结果怎么比，才知道多出来的东西是否有效

| 编号 | 训练 | 解码 | 回答的问题 |
|---|---|---|---|
| A | 未微调 Qwen | Raw | 原模型起点如何？ |
| B | 普通 LoRA SFT | Raw | 当前数据微调带来什么变化？ |
| B 的另一种解码 | 同一个 B checkpoint | Pair-aware | 只改变解码有什么作用？ |
| C | 同样 LoRA，加 Real Pair CE | Raw 与 Pair-aware | 联合训练本身有没有帮助？ |

每一行记录 Weighted F1、F/B→E、E→F/B、F↔B、N 漏判以及非N→N。不要只记录最好看的一个分数。

确认改善后，再补随机种子验证。DEV 用于选模，经过很多次选模后的 DEV 分数不能当作独立 TEST 泛化成绩。与旧 DeBERTa 比较时明确：两者预训练经历、结构与训练目标不同，不能只归因为模型大小。

保存 adapter 后，要用相同基础模型、相同 prompt 和保存的 adapter 重新加载，核对预测一致。保存目录写完整配置并避免覆盖旧运行；不要只保存一张分数截图。

## 七、现在就能自己写的第一段

新建自己的 Python 文件，先写这部分。它只读取 TRAIN 并打印一条训练消息，不加载 Qwen、不训练、不读取 DEV。

```python
from pathlib import Path
import json
import pandas as pd

PROJECT_DIR = Path(r"G:\pycode\SemEval\Task-2-DiCo-NLI-main")

LABEL_TO_ANSWER = {
    "FORWARD_ENTAILMENT": "F",
    "BACKWARD_ENTAILMENT": "B",
    "EQUIVALENCE": "E",
    "NEGATIVE_OTHER": "N",
}

SYSTEM_PROMPT = (
    "Classify the semantic relation between phrases A and B. "
    "F: A entails B, but B does not entail A. "
    "B: B entails A, but A does not entail B. "
    "E: A and B entail each other. "
    "N: neither phrase entails the other, even if they are related. "
    "Output exactly one letter: F, B, E, or N."
)


def build_messages(text1, text2, answer=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"A: {text1}\nB: {text2}"},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


if __name__ == "__main__":
    training = pd.read_csv(
        PROJECT_DIR / "data/train/dico_nli_train_track1_participant_labeled.csv"
    )
    assert {"text1", "text2", "label"} <= set(training.columns)
    assert set(training["label"]) <= set(LABEL_TO_ANSWER)
    assert training[["text1", "text2", "label"]].notna().all().all()

    print("训练行数：", len(training))
    print(training["label"].value_counts())

    row = training.iloc[0]
    messages = build_messages(
        row["text1"], row["text2"], LABEL_TO_ANSWER[row["label"]]
    )
    print(json.dumps(messages, ensure_ascii=False, indent=2))
```

能解释这段数据为什么对应一个答案后，下一步写分词与 loss mask。之后再写 LoRA 加载和训练。每一步都应有能打印、能核对的中间结果。

