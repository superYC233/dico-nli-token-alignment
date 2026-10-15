
# 文件迁入子目录后，从项目根目录定位数据和依赖。
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, configure_legacy_workdir
configure_legacy_workdir()

import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
from pathlib import Path

# todo 0. 固定随机种子，设置本次实验目录
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 上一版最佳：
# type_aux = 0.3
# NEGATIVE 权重 = 1.0
# best epoch = 6
# Weighted F1 = 0.8998
# RUN_DIR = Path(
#     f"./model/deberta_large_type_aux_w0p3_seed_{SEED}"
# )

# type_aux = 0.3
# auxiliary loss 中 NEGATIVE 权重 = 1.2
RUN_DIR = Path(
    f"./model/deberta_large_type_aux_w0p3_nweight1p2_seed_{SEED}"
)

# epoch=6
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 准备数据集
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')

# 使用已经经过多个 NLI 数据集训练的 DeBERTa-v3-large
model_name = (
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)

cache_path = r"D:\BaiduNetdiskDownload\PretrainedModel\models"

my_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path
)

my_model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_path,
    dtype=torch.float32
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# todo 1. 查看并处理数据集
def checkAndprocess_data():
    # print(train_data.head())

    # instance_id', 'pair_id', 'text1_lang', 'text2_lang', 'text1', 'text2', 'label'
    # instance_id → 当前这一条样本的唯一编号        pair_id → 这一组语义 pair 的编号
    # A B关系 和 B A关系属于同一个pair_ID，即使label不同
    # text1_lang → text1 的语言                 text2_lang  → text2 的语言
    # text1 → 第一个短语 text2 → 第二个短语        label → text1 和 text2 的语义关系
    # print(train_data.columns)

    '''
    FORWARD_ENTAILMENT     881    前向蕴含 A =>  B
    BACKWARD_ENTAILMENT    881    反向蕴含 B =>  A
    EQUIVALENCE            802    语义等价 A <=> B
    NEGATIVE_OTHER         478    不属于前三种关系的其他情况
    '''
    # print(train_data['label'].value_counts())

    # 查看同一个pair_id的数据
    first_pair_id = train_data.iloc[0]['pair_id']
    same_pair_data = train_data[train_data['pair_id'] == first_pair_id][
        ['pair_id', 'text1', 'text2', 'label']
    ]
    '''
                    pair_id                    text1                    text2                       label
      dico_phrasis_0000001             Conservative opposition    Conservatives              FORWARD_ENTAILMENT
      dico_phrasis_0000001            Conservatives               Conservative opposition    BACKWARD_ENTAILMENT
    '''
    # print(same_pair_data.to_string(index=False))

    # print(train_data[train_data['label'] == 'EQUIVALENCE'][['text1','text2','label']].head(5).to_string())
    # print(train_data[train_data['label'] == 'NEGATIVE_OTHER'][['text1','text2','label']].head(5).to_string())

    '''
    FORWARD_ENTAILMENT     190
    BACKWARD_ENTAILMENT    190
    EQUIVALENCE            174
    NEGATIVE_OTHER         106
    '''
    # print(dev_data['label'].value_counts())

    # 将标签转化为数字表示
    label_to_id = {
        'FORWARD_ENTAILMENT': 0,
        'BACKWARD_ENTAILMENT': 1,
        'EQUIVALENCE': 2,
        'NEGATIVE_OTHER': 3
    }

    train_data['label_id'] = train_data['label'].map(label_to_id)
    dev_data['label_id'] = dev_data['label'].map(label_to_id)
    # print(train_data.head())

    # train_data['pair_id'].value_counts() 每个 pair_id 在训练集中出现了几次
    # .value_counts() 出现 1/2 次的 pair 有多少个
    print(train_data['pair_id'].value_counts().value_counts())
    print(dev_data['pair_id'].value_counts().value_counts())

    pair_counts = dev_data['pair_id'].value_counts()

    single_pair_ids = pair_counts[pair_counts == 1].index

    # 所有pair_counts == 1的全是NEGATIVE_OTHER
    # 成对出现的pair没有NEGATIVE_OTHER
    print(
        dev_data[
            dev_data['pair_id'].isin(single_pair_ids)
        ]['label'].value_counts()
    )


# todo 2. 自定义短语对数据集
class DiCoDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text1 = row['text1']
        text2 = row['text2']
        label = row['label_id']

        encoded = self.tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label


# todo 2.1 真实 pair_id 数据集：同一个 pair_id 的两条真实数据组成一组
'''
作用:
    把同一个pair_id 的两条真实数据绑定成一个 Pair样本
pair_id = 001 
1. (A,B) label=0 FORWARD 
2. (B,A) label = 1 BACKWARD
经过PairDataset合并:
    ((A,B),(B,A), pair_label=0)
Pair三分类: 0 -> FB      1 -> BF      2 -> EE 
'''


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 统计每个pair_id 在数据集中出现了多少次
        # 如:
        # pair_id  count
        # 001       2
        # 002       1
        pair_counts = df['pair_id'].value_counts()

        # 2. 只保留出现2次的pair_id
        # pair_ids = [001, 002, 004]
        pair_ids = pair_counts[pair_counts == 2].index

        # 用于保存 最后整理好的所有pair
        self.pairs = []

        # 3. 按 pair_id 分组
        # 每次 group 就是同一个 pair_id 的两条真实数据
        '''
        例如：
        pair_id = 1
        group =
           pair_id    text    label
        0        1     AAA        0
        1        1     BBB        1

        pair_id = 3
        group =
           pair_id    text    label
        3        3     DDD        0
        4        3     EEE        1
        '''
        for pair_id, group in df[
            df['pair_id'].isin(pair_ids)
        ].groupby('pair_id'):
            # 重设索引, 因为现在group的索引来自df
            group = group.reset_index(drop=True)
            row1 = group.iloc[0]
            row2 = group.iloc[1]

            label1 = int(row1['label_id'])
            label2 = int(row2['label_id'])

            # 4. 根据两条真实数据的 label
            # 构造整个 Pair 的三分类正确答案
            # Pair 三分类：
            # 0 = FB
            # 1 = BF
            # 2 = EE
            if label1 == 0 and label2 == 1:
                pair_label = 0
            elif label1 == 1 and label2 == 0:
                pair_label = 1
            else:
                pair_label = 2

            self.pairs.append(
                (row1['text1'],
                 row1['text2'],
                 row2['text1'],
                 row2['text2'],
                 pair_label)
            )

    '''
     一个 Dataset 样本现在不是“一条数据” 而是“一个完整 Pair”
     训练集中：
     2564 条可逆关系数据
     两条组成一个 Pair
     2564 / 2 = 1282 个 Pair
    '''

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # 1. 取出第 index 个完整 Pair
        '''
        self.pairs[index] =
        (
            "a video game",
            "game",
            "game",
            "a video game",
            0
        )
        所以：
        text1_1 = "a video game"
        text1_2 = "game"

        text2_1 = "game"
        text2_2 = "a video game"
        pair_label = 0  -> FB
        '''
        text1_1, text1_2, text2_1, text2_2, pair_label = self.pairs[index]

        # 2. tokenizer
        # 2.1 tokenizer 第一条真实数据
        encoded1 = self.tokenizer(
            text1_1,
            text1_2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 2.2 tokenizer 第二条真实数据
        encoded2 = self.tokenizer(
            text2_1,
            text2_2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 第一条输入
        # [1,128] -> [128]
        input_ids1 = encoded1['input_ids'].squeeze(0)
        attention_mask1 = encoded1['attention_mask'].squeeze(0)

        # 第二条输入
        # [1,128] -> [128]
        input_ids2 = encoded2['input_ids'].squeeze(0)
        attention_mask2 = encoded2['attention_mask'].squeeze(0)

        # Pair 三分类正确答案
        pair_label = torch.tensor(
            pair_label,
            dtype=torch.long
        )

        # 9. 一个 PairDataset 样本最终返回：
        # 第一条真实数据的输入
        # 第二条真实数据的输入
        # 整个 Pair 的正确答案
        #
        # 如果 batch_size=8：
        # input_ids1       -> [8,128]
        # attention_mask1  -> [8,128]
        # input_ids2       -> [8,128]
        # attention_mask2  -> [8,128]
        # pair_label       -> [8    ]
        return (
            input_ids1,
            attention_mask1,
            input_ids2,
            attention_mask2,
            pair_label
        )


# todo 3. 创建开发集的 DataLoader
def create_dev_dataloader(batch_size=8):
    # 1. 创建 dev_dataset
    dev_dataset = DiCoDataset(dev_data, my_tokenizer, max_length=128)
    # 2. 创建 dev_loader
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # 3. 返回两个 loader
    return dev_dataloader


# todo 3.2 创建 四分类 + 真实pair的 两个Dataloader
def create_my_dataloaders(batch_size=8):
    # 1. 普通四分类数据
    # 3042条训练数据全部参加
    # 包括：
    # F / B / E / NEGATIVE
    row_dataset = DiCoDataset(train_data, my_tokenizer, max_length=128)
    row_dataloader = DataLoader(row_dataset, batch_size=batch_size, shuffle=True)

    # 2. Pair 数据
    # 只包含1282个真实 pair_id
    # 每个 pair_id 只出现一次
    pair_dataset = PairDataset(train_data, my_tokenizer, max_length=128)
    pair_dataloader = DataLoader(pair_dataset, batch_size=4, shuffle=True)

    print(f"普通四分类样本数：{len(row_dataset)}")
    print(f"真实 Pair 数量：{len(pair_dataset)}")

    return row_dataloader, pair_dataloader


# todo 4. 定义 Deberta 四分类模型
class debertaLargeDiCoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 保存 Deberta 模型
        self.deberta_model = my_model
        # 2. 定义 Dropout
        self.dropout = nn.Dropout(0.2)
        # 3. 定义四分类层
        self.output = nn.Linear(self.deberta_model.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        # 1. 将三个输入传给 Deberta
        deberta_output = self.deberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 2. 取出 [CLS] 位置的向量
        last_hidden_state = deberta_output.last_hidden_state
        cls_feature = last_hidden_state[:, 0, :]
        # 3. 经过 Dropout
        cls_feature = self.dropout(cls_feature)
        # 4. 经过分类层并返回
        output = self.output(cls_feature)

        return output
# todo 4.1 辅助任务：把原来的4分类，再粗分成3类
# 原任务：
#   0 -> FORWARD
#   1 -> BACKWARD
#   2 -> EQUIVALENCE
#   3 -> NEGATIVE
#
# 辅助任务：
#   0 -> 单向蕴含（FORWARD / BACKWARD）
#   1 -> 等价（EQUIVALENCE）
#   2 -> 其他关系（NEGATIVE）
#
# 作用：
#   原来的四分类继续负责区分 F / B / E / N
#   这个辅助损失额外强调：
#   “单向蕴含”和“等价”和“其他关系”之间的边界
def relation_type_loss(logits, labels):

    # logits形状：
    # [batch_size, 4]
    #
    # 每一行4个分数分别表示：
    # [F分数, B分数, E分数, N分数]

    # 转成float，保证后面的loss计算稳定
    logits = logits.float()

    # 1. 把 F 和 B 合并成“单向蕴含”这一类
    # 原来:
    # [F, B, E, N]
    # 现在辅助任务想变成：
    # [单向蕴含, E, N]
    # F 和 B 都属于“单向蕴含”
    # 所以把 F 和 B 两个logit合并
    # 注意:
    # logits不是概率，不能直接 F + B
    # logsumexp(F, B) 相当于在概率空间里：
    # P(单向蕴含) = P(F) + P(B)
    directional_score = torch.logsumexp(
        logits[:, :2],   # 取出F和B两列
        dim=1
    )

    # 2. 构造辅助三分类的logits
    # type_logits形状：
    # [batch_size, 3]
    # 第0列：单向蕴含(F/B合并)
    # 第1列：EQUIVALENCE
    # 第2列：NEGATIVE
    type_logits = torch.stack(
        [
            directional_score,
            logits[:, 2],
            logits[:, 3]
        ],
        dim=1
    )

    # 3. 把原来的4分类标签映射成辅助3分类标签
    # 原标签：
    # F -> 0
    # B -> 1
    # E -> 2
    # N -> 3
    # 辅助标签：
    # F -> 0  单向蕴含
    # B -> 0  单向蕴含
    # E -> 1  等价
    # N -> 2  其他
    #
    # 所以映射关系是：
    # [0, 0, 1, 2]
    label_mapping = labels.new_tensor(
        [0, 0, 1, 2]
    )

    # 例如：
    # labels = [0, 2, 3]
    #         = [F, E, N]
    # 映射后：
    # type_labels = [0, 1, 2]
    #             = [单向蕴含, 等价, 其他]
    type_labels = label_mapping[labels]

    # ==========================================
    # 4. 计算辅助三分类损失
    # ==========================================
    #
    # 让模型额外学习：
    # 这条样本到底属于
    # “单向蕴含 / 等价 / 其他关系”中的哪一类

    # 加强Negative
    type_weights = logits.new_tensor(
        [1.0, 1.0, 1.2]
    )

    return torch.nn.functional.cross_entropy(
        type_logits,
        type_labels,
        weight=type_weights
    )

# todo 5. 四分类 CE + 真实 pair_id 进行 Pair CE
def train_model_real_pair_ce():
    # 1. 获取两个 DataLoader
    row_dataloader, pair_dataloader = (
        create_my_dataloaders(batch_size=8)
    )

    # 2. 创建模型
    deberta_model = debertaLargeDiCoModel().to(device)

    # 3. 损失函数
    class_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device
    )

    classification_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Pair CE 权重
    pair_weights = torch.tensor(
        [1.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device
    )

    pair_criterion = nn.CrossEntropyLoss(
        weight=pair_weights
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=5e-6,
        weight_decay=0.01
    )

    epochs = 10

    pair_weight = 0.5

    for epoch in range(epochs):
        deberta_model.train()

        total_classification_loss = 0.0
        total_pair_loss = 0.0

        # 第一部分：
        # 所有3042条真实数据正常计算四分类 CE
        for input_ids, attention_mask, label in tqdm(row_dataloader, desc='四分类训练'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            # 在前向之前清除上一批的梯度
            optimizer.zero_grad(set_to_none=True)

            # 混合精度：前向计算和损失放在这里
            # 能用低精度算的地方，尽量使用 bfloat16
            with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
            ):
                logits = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 原来的4分类任务：
                # 继续学习 F / B / E / N
                ce_loss = classification_criterion(
                    logits,
                    label
                )

                # 辅助3分类任务：
                # 再学习 单向蕴含 / 等价 / 其他
                type_loss = relation_type_loss(
                    logits,
                    label
                )

                # 最终Row阶段损失：
                # 4分类是主任务
                # 3分类辅助任务占0.3
                classification_loss = (
                        ce_loss
                        + 0.3 * type_loss
                )

            classification_loss.backward()
            optimizer.step()

            total_classification_loss += classification_loss.item()

        # 第二部分：
        # 每个真实 pair_id 只计算一次 Pair CE
        for (input_ids1, attention_mask1,
             input_ids2, attention_mask2, pair_label) in tqdm(pair_dataloader, desc='Pair训练'):
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            pair_label = pair_label.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 自动混合精度
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits1 = deberta_model(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1
                )

                logits2 = deberta_model(
                    input_ids=input_ids2,
                    attention_mask=attention_mask2
                )

                score_fb = logits1[:, 0] + logits2[:, 1]
                score_bf = logits1[:, 1] + logits2[:, 0]
                score_ee = logits1[:, 2] + logits2[:, 2]

                pair_logits = torch.stack(
                    [score_fb, score_bf, score_ee],
                    dim=1
                )

                pair_loss = pair_criterion(pair_logits, pair_label)
                loss = pair_weight * pair_loss

            loss.backward()
            optimizer.step()

            total_pair_loss += pair_loss.item()

            # # 两条真实数据分别经过同一个模型
            # # logits1: [batch, 4]
            # # logits2: [batch, 4]
            # logits1 = deberta_model(
            #     input_ids=input_ids1,
            #     attention_mask=attention_mask1
            # )
            #
            # logits2 = deberta_model(
            #     input_ids=input_ids2,
            #     attention_mask=attention_mask2
            # )
            #
            # # 三种合法组合：
            # score_fb = logits1[:, 0] + logits2[:, 1]
            # score_bf = logits1[:, 1] + logits2[:, 0]
            # score_ee = logits1[:, 2] + logits2[:, 2]
            #
            # pair_logits = torch.stack(
            #     [
            #         score_fb,
            #         score_bf,
            #         score_ee
            #     ],
            #     dim=1
            # )
            #
            # # Pair 三分类 CE
            # pair_loss = pair_criterion(pair_logits, pair_label)
            #
            # loss = pair_weight * pair_loss
            #
            # optimizer.zero_grad()
            #
            # loss.backward()
            #
            # optimizer.step()
            #
            # total_pair_loss += pair_loss.item()

        avg_classification = total_classification_loss / len(row_dataloader)
        avg_pair = total_pair_loss / len(pair_dataloader)

        print(
            f"\n第 {epoch + 1} 轮\n"
            f"四分类 CE：{avg_classification:.6f}\n"
            f"Pair CE：{avg_pair:.6f}\n"
            f"乘权重后的 Pair CE："
            f"{pair_weight * avg_pair:.6f}"
        )

        # 保存模型
        torch.save(
            deberta_model.state_dict(),
            RUN_DIR / f"epoch_{epoch + 1}.pth"
        )


# todo 6. 评估模型 + pair-aware 后处理
def evaluate_pair_aware_model():
    # 1. 获取开发集
    dev_dataloader = create_dev_dataloader()

    # 2. 创建模型
    deberta_model = debertaLargeDiCoModel().to(device)

    best_score = -1.0
    best_epoch = None
    best_preds = None
    best_labels = None

    # ==========================================
    # 3. 自动寻找当前目录下实际存在的模型
    # ==========================================
    #
    # 例如目录中只有：
    # epoch_6.pth
    #
    # model_paths =
    # [epoch_6.pth]
    #
    # 如果有：
    # epoch_1.pth
    # epoch_2.pth
    # epoch_5.pth
    #
    # 就全部拿出来评估
    model_paths = list(
        RUN_DIR.glob("epoch_*.pth")
    )

    # 一个模型都没有
    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"{RUN_DIR} 中没有找到 epoch_*.pth"
        )

    # ==========================================
    # 4. 从文件名中提取 epoch
    # ==========================================
    #
    # epoch_6.pth
    # ↓
    # stem = "epoch_6"
    # ↓
    # epoch = 6
    epochs = []

    for model_path in model_paths:
        epoch = int(
            model_path.stem.split("_")[1]
        )

        epochs.append(epoch)

    # 按epoch从小到大排序
    epochs = sorted(epochs)

    print(
        f"\n找到的模型 epoch：{epochs}"
    )

    # ==========================================
    # 5. 依次评估所有实际存在的模型
    # ==========================================
    for epoch in epochs:

        deberta_model.load_state_dict(
            torch.load(
                RUN_DIR / f"epoch_{epoch}.pth",
                map_location=device,
                weights_only=True
            )
        )

        deberta_model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, label in tqdm(
                    dev_dataloader
            ):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)

                # [batch_size, 4]
                deberta_output = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # pair-aware需要保存原始4分类logits
                all_logits.append(
                    deberta_output.cpu()
                )

                all_labels.append(
                    label.cpu()
                )

        # [660, 4]
        all_logits = torch.cat(
            all_logits,
            dim=0
        )

        # [660]
        all_labels = torch.cat(
            all_labels,
            dim=0
        )

        # pair-aware预测
        final_preds = pair_aware(
            all_logits,
            dev_data
        )

        # Weighted F1
        score = f1_score(
            all_labels,
            final_preds,
            average="weighted",
            zero_division=0
        )

        print(
            f'第{epoch}轮: '
            f'Pair-aware Weighted F1 = '
            f'{score:.4f}'
        )

        # 保存当前最佳模型的信息
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_preds = final_preds.clone()
            best_labels = all_labels.clone()

    # ==========================================
    # 6. 删除其他模型，只保留最佳epoch
    # ==========================================
    for epoch in epochs:

        if epoch != best_epoch:

            model_path = (
                RUN_DIR
                / f"epoch_{epoch}.pth"
            )

            if model_path.exists():
                model_path.unlink()

    # ==========================================
    # 7. 打印最终结果
    # ==========================================
    print(
        f'\n最佳 epoch：{best_epoch}'
        f'\n最佳 Pair-aware Weighted F1：'
        f'{best_score:.4f}'
    )

    assert best_preds is not None

    # 各类别结果
    print(
        classification_report(
            best_labels,
            best_preds,
            labels=[0, 1, 2, 3],
            target_names=[
                "FORWARD_ENTAILMENT",
                "BACKWARD_ENTAILMENT",
                "EQUIVALENCE",
                "NEGATIVE_OTHER"
            ],
            digits=4,
            zero_division=0
        )
    )

    # 混淆矩阵
    names = [
        "FORWARD",
        "BACKWARD",
        "EQUIVALENCE",
        "NEGATIVE"
    ]

    matrix = confusion_matrix(
        best_labels,
        best_preds,
        labels=[0, 1, 2, 3]
    )

    print(
        pd.DataFrame(
            matrix,
            index=[
                f"真实_{name}"
                for name in names
            ],
            columns=[
                f"预测_{name}"
                for name in names
            ]
        ).to_string()
    )

    return (
        best_epoch,
        best_score,
        best_preds
    )


# todo 7. 导出开发集预测结果
def submission_output(best_preds):
    # 把数字编号转换回官方标签名称
    id_to_label = {
        0: "FORWARD_ENTAILMENT",
        1: "BACKWARD_ENTAILMENT",
        2: "EQUIVALENCE",
        3: "NEGATIVE_OTHER"
    }

    # 每条样本的编号，对应一个预测标签
    predictions = pd.DataFrame({
        "instance_id": dev_data["instance_id"].tolist(),
        "label": [id_to_label[i] for i in best_preds.tolist()]
    })

    # 保存文件，不保存 pandas 的行号
    predictions.to_csv(
        RUN_DIR / "dev_predictions_pair_aware.csv",
        index=False
    )


# todo 8. pair-aware 后处理 => 解决方向一致性问题
def pair_aware(all_logits, data):
    # 1. 先得到普通预测
    final_preds = torch.argmax(all_logits, dim=1)

    # 2. pair_aware 处理
    # 同一个pair_id的两条样本一起判断
    '''
       pair_id   text1              text2              label
    0    100    A dog is running   An animal runs        0
    1    100    An animal runs     A dog is running      1
    2    101    A man is eating    A person eats         0
    3    101    A person eats      A man is eating       1
    '''
    for pair_id, group in data.groupby('pair_id'):
        # 只处理有2条数据的pair
        if len(group) != 2:
            continue

        # 找到这一对样本的位置
        # idx1,idx2 = [0,1]
        idx1, idx2 = group.index.tolist()

        # 两条样本各自的4类分数
        logits1 = all_logits[idx1]
        logits2 = all_logits[idx2]

        # 三种合法组合
        # forward + backward
        score_fb = logits1[0] + logits2[1]

        # backward + forward
        score_bf = logits1[1] + logits2[0]

        # equivalence + equivalence
        score_ee = logits1[2] + logits2[2]

        '''
        [
          13,   # forward + backward
          8,   # backward + forward
          2    # equivalence + equivalence
        ]
        '''
        scores = torch.stack([score_fb, score_bf, score_ee])

        # 找到总分最高的组合
        best_pair = torch.argmax(scores).item()

        # 强制修改结果
        if best_pair == 0:
            final_preds[idx1] = 0
            final_preds[idx2] = 1
        elif best_pair == 1:
            final_preds[idx1] = 1
            final_preds[idx2] = 0
        else:
            final_preds[idx1] = 2
            final_preds[idx2] = 2

    return final_preds


# todo 9. 分析 pair-aware 的错误样本
def analyze_pair_aware_errors(best_preds):
    # 1. 复制一份 dev_data，避免修改原数据
    error_data = dev_data.reset_index(drop=True).copy()

    # 2. 加入模型预测结果
    error_data['pred_id'] = best_preds.tolist()

    id_to_label = {
        0: "FORWARD_ENTAILMENT",
        1: "BACKWARD_ENTAILMENT",
        2: "EQUIVALENCE",
        3: "NEGATIVE_OTHER"
    }

    # 把预测数字转换成标签名称
    error_data['pred_label'] = error_data['pred_id'].map(id_to_label)

    # 3. FORWARD / BACKWARD 被错预测成 EQUIVALENCE
    fb_to_e = error_data[
        (error_data['label_id'].isin([0, 1])) &
        (error_data['pred_id'] == 2)
        ]

    print('\n========== FORWARD/BACKWARD -> EQUIVALENCE ==========')
    print(f'数量：{len(fb_to_e)}')

    print(
        fb_to_e[
            ['text1', 'text2', 'label', 'pred_label']
        ].to_string(index=False)
    )

    # 4. EQUIVALENCE 被错预测成 FORWARD / BACKWARD
    e_to_fb = error_data[
        (error_data['label_id'] == 2) &
        (error_data['pred_id'].isin([0, 1]))
        ]

    print('\n========== EQUIVALENCE -> FORWARD/BACKWARD ==========')
    print(f'数量：{len(e_to_fb)}')

    print(
        e_to_fb[
            ['text1', 'text2', 'label', 'pred_label']
        ].to_string(index=False)
    )

    # 5. NEGATIVE_OTHER 被错预测成 EQUIVALENCE
    n_to_e = error_data[
        (error_data['label_id'] == 3) &
        (error_data['pred_id'] == 2)
        ]

    print('\n========== NEGATIVE_OTHER -> EQUIVALENCE ==========')
    print(f'数量：{len(n_to_e)}')

    print(
        n_to_e[
            ['text1', 'text2', 'label', 'pred_label']
        ].to_string(index=False)
    )


# todo 10. 对比新旧模型的预测结果
def compare_predictions():
    project_dir = (PROJECT_ROOT)

    # 1. 读取标准答案
    gold = pd.read_csv(
        project_dir / "data/dev/dico_nli_dev_track1_reference.csv"
    )[["instance_id", "label"]]

    gold = gold.rename(columns={"label": "gold_label"})

    # 旧模型：N weight = 1.5
    old_preds = pd.read_csv(
        project_dir
        / "model/deberta_real_pair_ce_w0p5_nweight1p5_seed_42/dev_predictions_pair_aware.csv"
    ).rename(columns={"label": "old_pred"})

    # 新模型：N weight = 1.5 + EE weight = 1.25
    new_preds = pd.read_csv(
        RUN_DIR / "dev_predictions_pair_aware.csv"
    ).rename(columns={"label": "new_pred"})

    # 4. 检查样本编号，避免漏掉样本或重复合并
    for name, data in [
        ("标准答案", gold),
        ("旧预测", old_preds),
        ("新预测", new_preds)
    ]:
        if data["instance_id"].isna().any():
            raise ValueError(f"{name}存在空 instance_id")

        if data["instance_id"].duplicated().any():
            raise ValueError(f"{name}存在重复 instance_id")

    gold_ids = set(gold["instance_id"])

    if set(old_preds["instance_id"]) != gold_ids:
        raise ValueError("旧预测的样本编号与标准答案不一致")

    if set(new_preds["instance_id"]) != gold_ids:
        raise ValueError("新预测的样本编号与标准答案不一致")

    # 5. 按编号合并，CSV 中的排列顺序不同也能正确对齐
    comparison = gold.merge(
        old_preds[["instance_id", "old_pred"]],
        on="instance_id",
        validate="one_to_one"
    )

    comparison = comparison.merge(
        new_preds[["instance_id", "new_pred"]],
        on="instance_id",
        validate="one_to_one"
    )

    # 加入短语，方便人工分析
    comparison = comparison.merge(
        dev_data[["instance_id", "pair_id", "text1", "text2"]],
        on="instance_id",
        how="left",
        validate="one_to_one"
    )

    # 6. 分别判断新旧模型是否预测正确
    comparison["old_correct"] = (
            comparison["old_pred"] == comparison["gold_label"]
    )

    comparison["new_correct"] = (
            comparison["new_pred"] == comparison["gold_label"]
    )

    # 旧模型错、新模型对
    improved = comparison[
        (~comparison["old_correct"])
        & comparison["new_correct"]
        ]

    # 旧模型对、新模型错
    regressed = comparison[
        comparison["old_correct"]
        & (~comparison["new_correct"])
        ]

    # 7. 打印具体样本
    columns = [
        "pair_id",
        "text1",
        "text2",
        "gold_label",
        "old_pred",
        "new_pred"
    ]

    print("\n===== 旧模型错，新模型对 =====")
    print(
        f"{len(improved)} 条样本，"
        f"涉及 {improved['pair_id'].nunique()} 个 pair_id"
    )
    print(improved[columns].to_string(index=False))

    print("\n===== 旧模型对，新模型错 =====")
    print(
        f"{len(regressed)} 条样本，"
        f"涉及 {regressed['pair_id'].nunique()} 个 pair_id"
    )
    print(regressed[columns].to_string(index=False))

    # 8. 按真实类别统计收益和损失
    summary = pd.DataFrame({
        "新修正条数": improved["gold_label"].value_counts(),
        "新增错误条数": regressed["gold_label"].value_counts()
    }).fillna(0).astype(int)

    summary["净增加正确数"] = (
            summary["新修正条数"] - summary["新增错误条数"]
    )

    print("\n===== 按真实类别汇总 =====")
    print(summary.to_string())

    # 9. 保存对比结果
    output_dir = (
            project_dir
            / "results/deberta_real_pair_ce_w0p5_seed_42"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    improved.to_csv(
        output_dir / "improved_examples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    regressed.to_csv(
        output_dir / "regressed_examples.csv",
        index=False,
        encoding="utf-8-sig"
    )
# todo 9.1 按完整 pair 分析错误
def analyze_pair_errors(best_preds):
    # 复制开发集
    error_data = dev_data.reset_index(drop=True).copy()

    # 加入预测结果
    error_data['pred_id'] = best_preds.tolist()

    pair_errors = []

    # 按 pair_id 分组
    for pair_id, group in error_data.groupby('pair_id'):

        # 只分析真正成对的数据
        if len(group) != 2:
            continue

        group = group.reset_index(drop=True)

        row1 = group.iloc[0]
        row2 = group.iloc[1]

        # =========================
        # 1. 得到真实 Pair 类型
        # =========================
        true1 = row1['label_id']
        true2 = row2['label_id']

        if true1 == 0 and true2 == 1:
            true_pair = 'FB'
        elif true1 == 1 and true2 == 0:
            true_pair = 'BF'
        elif true1 == 2 and true2 == 2:
            true_pair = 'EE'
        else:
            true_pair = 'OTHER'

        # =========================
        # 2. 得到预测 Pair 类型
        # =========================
        pred1 = row1['pred_id']
        pred2 = row2['pred_id']

        if pred1 == 0 and pred2 == 1:
            pred_pair = 'FB'
        elif pred1 == 1 and pred2 == 0:
            pred_pair = 'BF'
        elif pred1 == 2 and pred2 == 2:
            pred_pair = 'EE'
        else:
            pred_pair = 'OTHER'

        # =========================
        # 3. 只保存预测错的 pair
        # =========================
        if true_pair != pred_pair:
            pair_errors.append({
                'pair_id': pair_id,
                'text1': row1['text1'],
                'text2': row1['text2'],
                'true_pair': true_pair,
                'pred_pair': pred_pair
            })

    pair_errors = pd.DataFrame(pair_errors)

    print('\n========== Pair级错误 ==========')
    print(f'错误 Pair 数量：{len(pair_errors)}')

    print(pair_errors.to_string(index=False))

    # =========================
    # 4. 汇总错误类型
    # =========================
    print('\n========== Pair错误类型统计 ==========')

    print(
        pair_errors.groupby(
            ['true_pair', 'pred_pair']
        ).size()
    )

# todo 9.2 分析错误 Pair 的三个分数
def analyze_pair_scores(best_epoch, best_preds):
    # 1. 加载开发集
    dev_dataloader = create_dev_dataloader()

    # 2. 加载最佳模型
    deberta_model = debertaLargeDiCoModel().to(device)

    deberta_model.load_state_dict(
        torch.load(
            RUN_DIR / f"epoch_{best_epoch}.pth",
            map_location=device,
            weights_only=True
        )
    )

    deberta_model.eval()

    # 3. 获取整个 dev 的四分类 logits
    all_logits = []

    with torch.no_grad():
        for input_ids, attention_mask, label in tqdm(dev_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = deberta_model(
                input_ids,
                attention_mask
            )

            all_logits.append(logits.cpu())

    # [660, 4]
    all_logits = torch.cat(all_logits, dim=0)

    # 4. 准备数据
    data = dev_data.reset_index(drop=True).copy()
    data['pred_id'] = best_preds.tolist()

    error_scores = []

    # 5. 按 pair_id 分析
    for pair_id, group in data.groupby('pair_id'):

        # 只分析完整 Pair
        if len(group) != 2:
            continue

        idx1, idx2 = group.index.tolist()

        row1 = group.iloc[0]
        row2 = group.iloc[1]

        logits1 = all_logits[idx1]
        logits2 = all_logits[idx2]

        # =========================
        # 三种 Pair 分数
        # =========================
        score_fb = (
            logits1[0] + logits2[1]
        ).item()

        score_bf = (
            logits1[1] + logits2[0]
        ).item()

        score_ee = (
            logits1[2] + logits2[2]
        ).item()

        scores = {
            'FB': score_fb,
            'BF': score_bf,
            'EE': score_ee
        }

        # =========================
        # 真实 Pair
        # =========================
        true1 = int(row1['label_id'])
        true2 = int(row2['label_id'])

        if true1 == 0 and true2 == 1:
            true_pair = 'FB'

        elif true1 == 1 and true2 == 0:
            true_pair = 'BF'

        elif true1 == 2 and true2 == 2:
            true_pair = 'EE'

        else:
            continue

        # =========================
        # 预测 Pair
        # =========================
        pred1 = int(row1['pred_id'])
        pred2 = int(row2['pred_id'])

        if pred1 == 0 and pred2 == 1:
            pred_pair = 'FB'

        elif pred1 == 1 and pred2 == 0:
            pred_pair = 'BF'

        elif pred1 == 2 and pred2 == 2:
            pred_pair = 'EE'

        else:
            continue

        # 只分析预测错误的 Pair
        if true_pair == pred_pair:
            continue

        # 正确类别的分数
        true_score = scores[true_pair]

        # 错误预测类别的分数
        pred_score = scores[pred_pair]

        # 错误类别比正确类别高多少
        margin = pred_score - true_score

        error_scores.append({
            'pair_id': pair_id,
            'text1': row1['text1'],
            'text2': row1['text2'],
            'true_pair': true_pair,
            'pred_pair': pred_pair,
            'score_FB': score_fb,
            'score_BF': score_bf,
            'score_EE': score_ee,
            'margin': margin
        })

    # 6. 转成 DataFrame
    error_scores = pd.DataFrame(error_scores)

    # margin 小 -> 模型只是稍微选错
    # margin 大 -> 模型非常确定地选错
    error_scores = error_scores.sort_values(
        'margin',
        ascending=True
    )

    print('\n========== 错误 Pair 分数分析 ==========')

    print(
        error_scores.to_string(
            index=False,
            float_format=lambda x: f'{x:.4f}'
        )
    )

    print('\n========== 不同错误类型的平均 margin ==========')

    print(
        error_scores.groupby(
            ['true_pair', 'pred_pair']
        )['margin'].agg(
            ['count', 'mean', 'min', 'max']
        )
    )

# todo 9.3 检查错误主要来自 Pair 还是 NEGATIVE
def analyze_pair_vs_negative_errors(best_preds):

    # 1. 准备dev数据
    data = dev_data.reset_index(drop=True).copy()

    # 加入最终预测结果
    data['pred_id'] = best_preds.tolist()

    # 是否预测正确
    data['correct'] = (
        data['label_id'] == data['pred_id']
    )

    # ============================================
    # 2. 区分完整Pair和单独样本
    # ============================================
    pair_counts = data['pair_id'].value_counts()

    # 出现2次 = 完整Pair
    paired_ids = pair_counts[
        pair_counts == 2
    ].index

    # 出现1次 = singleton
    singleton_ids = pair_counts[
        pair_counts == 1
    ].index

    paired_data = data[
        data['pair_id'].isin(paired_ids)
    ]

    singleton_data = data[
        data['pair_id'].isin(singleton_ids)
    ]

    # ============================================
    # 3. Pair部分错了多少条样本
    # ============================================
    pair_wrong_rows = (
        ~paired_data['correct']
    ).sum()

    # 有多少个完整Pair至少错了一条
    wrong_pair_count = (
        paired_data[
            ~paired_data['correct']
        ]['pair_id']
        .nunique()
    )

    # ============================================
    # 4. NEGATIVE / singleton错了多少条
    # ============================================
    negative_wrong_rows = (
        ~singleton_data['correct']
    ).sum()

    # ============================================
    # 5. 总错误
    # ============================================
    total_wrong = (
        ~data['correct']
    ).sum()

    print('\n========== 错误来源分析 ==========')

    print(
        f'总错误样本数：{total_wrong}'
    )

    print(
        f'\nPair部分错误样本数：'
        f'{pair_wrong_rows}'
    )

    print(
        f'错误Pair数量：'
        f'{wrong_pair_count}'
        f' / {len(paired_ids)}'
    )

    print(
        f'\nNEGATIVE/singleton错误样本数：'
        f'{negative_wrong_rows}'
    )

    # ============================================
    # 6. 看谁贡献的错误更多
    # ============================================
    if total_wrong > 0:

        pair_ratio = (
            pair_wrong_rows
            / total_wrong
            * 100
        )

        negative_ratio = (
            negative_wrong_rows
            / total_wrong
            * 100
        )

        print(
            f'\nPair错误占总错误：'
            f'{pair_ratio:.2f}%'
        )

        print(
            f'NEGATIVE错误占总错误：'
            f'{negative_ratio:.2f}%'
        )

    # ============================================
    # 7. NEGATIVE具体被错成什么
    # ============================================
    negative_errors = singleton_data[
        ~singleton_data['correct']
    ]

    print(
        '\n========== NEGATIVE具体错成什么 =========='
    )

    print(
        negative_errors['pred_id']
        .value_counts()
        .sort_index()
    )

# todo n. 测试
if __name__ == '__main__':
    checkAndprocess_data()

    train_model_real_pair_ce()
    # 找出 pair-aware F1 最高的一轮
    best_epoch, best_score, best_preds = evaluate_pair_aware_model()

    # 保存这一轮的开发集预测
    submission_output(best_preds)

    print(f"最佳轮次：{best_epoch}")
    print(f"最佳 Pair-aware F1：{best_score:.4f}")

    # analyze_pair_aware_errors(best_preds)

    # analyze_pair_errors(best_preds)
    # compare_predictions()

    analyze_pair_vs_negative_errors(
        best_preds
    )