
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

# 在 NLI-base 最佳版基础上，
# 仅给 Pair 三分类 CE 加入 label_smoothing=0.05
#
# 其他配置保持不变：
# 四分类权重：[1.0, 1.0, 1.0, 1.5]
# Pair CE 权重：[1.0, 1.0, 1.25]
# pair_weight = 0.5
# pair-aware 推理保持不变
#
# 当前最佳结果：
# best epoch = 8
# Pair-aware Weighted F1 = 0.9160
# RUN_DIR = Path(
#     r"G:\pycode\SemEval\Task-2-DiCo-NLI-main\model"
# ) / f"deberta_nli_base_pair_smoothing_0p05_seed_{SEED}"

# Pair CE 标签平滑 0.05 + EE 权重由 1.25 降至 1.0
# 对照版本：标签平滑 0.05、EE 权重 1.25，最佳 F1=0.916043
# 当前结果:
# best epoch = 5
# Pair-aware Weighted F1 = 0.9148
RUN_DIR = (PROJECT_ROOT / 'model') / f"deberta_nli_base_pair_smoothing_clean_seed_{SEED}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

train_data = pd.read_csv("./data/internal_split_clean/train.csv")
dev_data = pd.read_csv("./data/internal_split_clean/valid.csv")

# 准备经过 NLI 数据训练的 DeBERTa-v3-base
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# 模型下载、缓存到 D 盘
cache_path = r"D:\BaiduNetdiskDownload\PretrainedModel\models"

my_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path
)

# 加载 backbone，后面继续使用自己的四分类头
my_model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_path,
    dtype=torch.float32,
    use_safetensors=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    def __init__(self, df, Deberta_tokenizer, max_length):
        self.df = df
        self.Deberta_tokenizer = Deberta_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text1 = row['text1']
        text2 = row['text2']
        label = row['label_id']

        encoded = self.Deberta_tokenizer(
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
        # 如果 batch_size=16：
        # input_ids1       -> [16,128]
        # attention_mask1  -> [16,128]
        # input_ids2       -> [16,128]
        # attention_mask2  -> [16,128]
        # pair_label       -> [16]
        return (
            input_ids1,
            attention_mask1,
            input_ids2,
            attention_mask2,
            pair_label
        )


# todo 3. 创建开发集的 DataLoader
def create_dev_dataloader(batch_size=16):
    # 1. 创建 dev_dataset
    dev_dataset = DiCoDataset(dev_data, my_tokenizer, max_length=128)
    # 2. 创建 dev_loader
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # 3. 返回两个 loader
    return dev_dataloader


# todo 3.2 创建 四分类 + 真实pair的 两个Dataloader
def create_my_dataloaders(batch_size=16):
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
    pair_dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)

    print(f"普通四分类样本数：{len(row_dataset)}")
    print(f"真实 Pair 数量：{len(pair_dataset)}")

    return row_dataloader, pair_dataloader


# todo 4. 定义 Deberta 四分类模型
class debertaDiCoModel(nn.Module):
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


# todo 5. 四分类 CE + 真实 pair_id 进行 Pair CE
def train_model_real_pair_ce():
    # 1. 获取两个 DataLoader
    row_dataloader, pair_dataloader = (
        create_my_dataloaders(batch_size=16)
    )

    # 2. 创建模型
    deberta_model = debertaDiCoModel().to(device)

    # 3. 损失函数
    # 四分类损失：适当增加真实 NEGATIVE 样本的权重
    class_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.5],
        dtype=torch.float32,
        device=device
    )

    classification_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Pair CE 权重
    pair_weights = torch.tensor(
        [1.0, 1.0, 1.25],
        dtype=torch.float32,
        device=device
    )

    # Pair 三分类：FB / BF / EE
    # 轻微软化训练目标
    pair_criterion = nn.CrossEntropyLoss(
        weight=pair_weights,
        label_smoothing=0.05
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=2e-5,
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

            logits = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
            classification_loss = classification_criterion(logits, label)

            total_classification_loss += classification_loss.item()

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

        # 第二部分：
        # 每个真实 pair_id 只计算一次 Pair CE
        for (input_ids1, attention_mask1,
             input_ids2, attention_mask2, pair_label) in tqdm(pair_dataloader, desc='Pair训练'):
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            pair_label = pair_label.to(device)

            # 两条真实数据分别经过同一个模型
            # logits1: [batch, 4]
            # logits2: [batch, 4]
            logits1 = deberta_model(
                input_ids=input_ids1,
                attention_mask=attention_mask1
            )

            logits2 = deberta_model(
                input_ids=input_ids2,
                attention_mask=attention_mask2
            )

            # 三种合法组合：
            score_fb = logits1[:, 0] + logits2[:, 1]
            score_bf = logits1[:, 1] + logits2[:, 0]
            score_ee = logits1[:, 2] + logits2[:, 2]

            pair_logits = torch.stack(
                [
                    score_fb,
                    score_bf,
                    score_ee
                ],
                dim=1
            )

            # Pair 三分类 CE
            pair_loss = pair_criterion(pair_logits, pair_label)

            loss = pair_weight * pair_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_pair_loss += pair_loss.item()
            # 当前 epoch 平均 loss
            # ======================================================

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


# # todo 6. 评估模型 + pair-aware 后处理
# def evaluate_pair_aware_model():
#     # 1. 获取开发集
#     dev_dataloader = create_dev_dataloader()
#
#     # 创建模型
#     deberta_model = debertaDiCoModel().to(device)
#
#     best_score = -1.0
#     best_epoch = None
#     best_preds = None
#     best_labels = None
#
#     # 2. 依次评估 10 个 epoch
#     for epoch in range(1,11):
#         # 加载当前 epoch 的模型参数
#         deberta_model.load_state_dict(
#             torch.load(
#                 RUN_DIR / f"epoch_{epoch}.pth",
#                 map_location=device,
#                 weights_only=True
#             )
#         )
#
#         deberta_model.eval()
#
#         # 3. 收集整个 dev 的 logits 和 label
#         all_logits = []
#         all_labels = []
#
#         with torch.no_grad():
#             for input_ids, attention_mask, label in tqdm(dev_dataloader):
#                 input_ids = input_ids.to(device)
#                 attention_mask = attention_mask.to(device)
#                 label = label.to(device)
#
#                 # 模型输出
#                 # [batch_size, 4]
#                 deberta_output = deberta_model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask
#                 )
#
#                 # 这里不能直接存 argmax 后的结果
#                 # pair-aware 需要使用每个类别的4个原始分数
#                 all_logits.append(deberta_output.cpu())
#                 all_labels.append(label.cpu())
#
#         # 4. 拼接所有 batch
#         # [660, 4]
#         all_logits = torch.cat(all_logits, dim=0)
#
#         # [660]
#         all_labels = torch.cat(all_labels, dim=0)
#
#         # 7. 计算pair-aware 后的 Weighted F1
#         final_preds = pair_aware(all_logits, dev_data)
#
#         score = f1_score(all_labels, final_preds, average="weighted", zero_division=0)
#
#         print(f'第{epoch}轮: '
#               f'Pair-aware Weighted F1 = {score:.4f}'
#         )
#
#         # 8. 保存最好的一轮
#         if score > best_score:
#             best_score = score
#             best_epoch = epoch
#
#             best_preds = final_preds.clone()
#             best_labels = all_labels.clone()
#
#     # 9. 打印最终结果
#     print(f'\n 最佳 epoch:{best_epoch}'
#           f'\n 最佳 Pair-aware Weighted F1 = {best_score:.4f}')
#
#
#     assert best_preds is not None
#
#     # 10. 查看 pair-aware 后各类别表现
#     print(classification_report(
#         best_labels,
#         best_preds,
#         labels=[0, 1, 2, 3],
#         target_names=[
#             "FORWARD_ENTAILMENT",
#             "BACKWARD_ENTAILMENT",
#             "EQUIVALENCE",
#             "NEGATIVE_OTHER"
#         ],
#         digits=4,
#         zero_division=0
#     ))
#
#     # 11. 混淆矩阵
#     names = [
#         "FORWARD",
#         "BACKWARD",
#         "EQUIVALENCE",
#         "NEGATIVE"
#     ]
#
#     matrix = confusion_matrix(
#         best_labels,
#         best_preds,
#         labels=[0, 1, 2, 3]
#     )
#
#     print(pd.DataFrame(
#         matrix,
#         index=[f"真实_{name}" for name in names],
#         columns=[f"预测_{name}" for name in names]
#     ).to_string())
#
#     return best_epoch, best_score, best_preds
# todo 6. 评估模型 + pair-aware 后处理
def evaluate_pair_aware_model():
    # 1. 获取开发集
    dev_dataloader = create_dev_dataloader()

    # 2. 创建模型
    deberta_model = debertaDiCoModel().to(device)

    best_score = -1.0
    best_epoch = None
    best_preds = None
    best_labels = None

    # ==========================================
    # 3. 自动寻找当前目录下实际存在的模型
    # ==========================================
    # 例如：
    # epoch_1.pth
    # epoch_2.pth
    # epoch_6.pth
    #
    # 就自动找到：
    # [1, 2, 6]
    model_paths = list(
        RUN_DIR.glob("epoch_*.pth")
    )

    # 如果一个模型都没有
    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"{RUN_DIR} 中没有找到 epoch_*.pth"
        )

    # ==========================================
    # 4. 从文件名中提取 epoch
    # ==========================================
    epochs = []

    for model_path in model_paths:
        # epoch_6.pth
        # -> epoch_6
        # -> 6
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

        # 加载当前epoch模型
        deberta_model.load_state_dict(
            torch.load(
                RUN_DIR / f"epoch_{epoch}.pth",
                map_location=device,
                weights_only=True
            )
        )

        deberta_model.eval()

        # 保存整个dev的logits和真实标签
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, label in tqdm(
                    dev_dataloader
            ):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)

                # 模型输出
                # [batch_size, 4]
                deberta_output = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # pair-aware需要原始4分类logits
                all_logits.append(
                    deberta_output.cpu()
                )

                all_labels.append(
                    label.cpu()
                )

        # ==========================================
        # 6. 拼接所有batch
        # ==========================================

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

        # ==========================================
        # 7. pair-aware 后处理
        # ==========================================
        final_preds = pair_aware(
            all_logits,
            dev_data
        )

        # 计算 Weighted F1
        score = f1_score(
            all_labels,
            final_preds,
            average="weighted",
            zero_division=0
        )

        print(
            f"第{epoch}轮: "
            f"Pair-aware Weighted F1 = "
            f"{score:.4f}"
        )

        # ==========================================
        # 8. 保存目前最好的epoch
        # ==========================================
        if score > best_score:
            best_score = score
            best_epoch = epoch

            best_preds = final_preds.clone()
            best_labels = all_labels.clone()

    # ==========================================
    # 9. 删除其他模型，只保留最佳epoch
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
    # 10. 打印最佳结果
    # ==========================================
    print(
        f"\n最佳 epoch：{best_epoch}"
        f"\n最佳 Pair-aware Weighted F1："
        f"{best_score:.4f}"
    )

    assert best_preds is not None

    # ==========================================
    # 11. 查看每个类别的表现
    # ==========================================
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

    # ==========================================
    # 12. 混淆矩阵
    # ==========================================
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



# todo 10. 对比新旧模型的预测结果
def compare_predictions():
    project_dir = (PROJECT_ROOT)

    # 1. 读取标准答案
    gold = pd.read_csv(
        project_dir / "data/dev/dico_nli_dev_track1_reference.csv"
    )[["instance_id", "label"]]

    gold = gold.rename(columns={"label": "gold_label"})

    # 对照：NLI-base，N weight = 1.5，F1 = 0.9133
    old_preds = pd.read_csv(
        project_dir
        / "model"
        / f"deberta_nli_base_pair_smoothing_0p05_seed_{SEED}"
        / "dev_predictions_pair_aware.csv"
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
    # 本次对比结果存入新实验目录
    output_dir = RUN_DIR / "comparison_vs_best"

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

# todo 11. 深度分析当前最佳模型
def analyze_best_model():
    """
    分析内容：
    1. Raw预测 vs Pair-aware预测
    2. Pair-aware修正/新增错误数量
    3. paired / singleton 分开表现
    4. 277个完整Pair的预测情况和置信度margin
    5. NEGATIVE_OTHER错误分析
    6. EQUIVALENCE 与 F/B 边界分析
    """

    # ==========================================
    # 1. 加载最佳模型
    # ==========================================
    best_epoch = 8

    model_path = RUN_DIR / f"epoch_{best_epoch}.pth"

    deberta_model = debertaDiCoModel().to(device)

    deberta_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    deberta_model.eval()

    dev_dataloader = create_dev_dataloader()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for input_ids, attention_mask, label in tqdm(
                dev_dataloader,
                desc="分析最佳模型"
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_logits.append(logits.cpu())
            all_labels.append(label)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 普通argmax预测
    raw_preds = torch.argmax(
        all_logits,
        dim=1
    )

    # Pair-aware预测
    pair_preds = pair_aware(
        all_logits.clone(),
        dev_data
    )

    # ==========================================
    # 2. Raw vs Pair-aware
    # ==========================================

    raw_f1 = f1_score(
        all_labels,
        raw_preds,
        average="weighted",
        zero_division=0
    )

    pair_f1 = f1_score(
        all_labels,
        pair_preds,
        average="weighted",
        zero_division=0
    )

    print("\n========== Raw vs Pair-aware ==========")

    print(
        f"Raw Weighted F1：{raw_f1:.6f}"
    )

    print(
        f"Pair-aware Weighted F1：{pair_f1:.6f}"
    )

    print(
        f"Pair-aware 提升：{pair_f1 - raw_f1:+.6f}"
    )

    # ==========================================
    # 3. Pair-aware到底修正/破坏了多少
    # ==========================================

    raw_correct = (
        raw_preds == all_labels
    )

    pair_correct = (
        pair_preds == all_labels
    )

    # 原来错，Pair后变对
    fixed_mask = (
        (~raw_correct)
        & pair_correct
    )

    # 原来对，Pair后变错
    broken_mask = (
        raw_correct
        & (~pair_correct)
    )

    # Pair确实修改了预测
    changed_mask = (
        raw_preds != pair_preds
    )

    print("\n========== Pair-aware贡献 ==========")

    print(
        f"总共修改预测："
        f"{changed_mask.sum().item()} 条"
    )

    print(
        f"Raw错 -> Pair对："
        f"{fixed_mask.sum().item()} 条"
    )

    print(
        f"Raw对 -> Pair错："
        f"{broken_mask.sum().item()} 条"
    )

    print(
        f"净增加正确："
        f"{fixed_mask.sum().item() - broken_mask.sum().item()} 条"
    )

    # ==========================================
    # 准备详细DataFrame
    # ==========================================

    analysis_df = dev_data.reset_index(
        drop=True
    ).copy()

    analysis_df["gold_id"] = (
        all_labels.numpy()
    )

    analysis_df["raw_pred_id"] = (
        raw_preds.numpy()
    )

    analysis_df["pair_pred_id"] = (
        pair_preds.numpy()
    )

    id_to_label = {
        0: "FORWARD_ENTAILMENT",
        1: "BACKWARD_ENTAILMENT",
        2: "EQUIVALENCE",
        3: "NEGATIVE_OTHER"
    }

    analysis_df["raw_pred"] = (
        analysis_df["raw_pred_id"]
        .map(id_to_label)
    )

    analysis_df["pair_pred"] = (
        analysis_df["pair_pred_id"]
        .map(id_to_label)
    )

    analysis_df["raw_correct"] = (
        raw_correct.numpy()
    )

    analysis_df["pair_correct"] = (
        pair_correct.numpy()
    )

    # ==========================================
    # 4. paired / singleton 分开分析
    # ==========================================

    pair_counts = (
        analysis_df["pair_id"]
        .value_counts()
    )

    paired_ids = (
        pair_counts[
            pair_counts == 2
        ].index
    )

    paired_mask = (
        analysis_df["pair_id"]
        .isin(paired_ids)
        .values
    )

    singleton_mask = ~paired_mask

    print("\n========== Paired / Singleton ==========")

    for name, mask in [
        ("Paired rows", paired_mask),
        ("Singleton rows", singleton_mask)
    ]:

        gold = all_labels[mask]
        pred = pair_preds[mask]

        score = f1_score(
            gold,
            pred,
            average="weighted",
            zero_division=0
        )

        accuracy = (
            (gold == pred)
            .float()
            .mean()
            .item()
        )

        print(
            f"{name}："
            f"{len(gold)} 条，"
            f"Weighted F1={score:.4f}，"
            f"Accuracy={accuracy:.4f}"
        )

    # ==========================================
    # 5. 277个完整Pair单独分析
    # ==========================================

    pair_results = []

    for pair_id, group in (
        analysis_df.groupby("pair_id")
    ):

        if len(group) != 2:
            continue

        idx1, idx2 = (
            group.index.tolist()
        )

        logits1 = all_logits[idx1]
        logits2 = all_logits[idx2]

        # 三种合法Pair分数
        score_fb = (
            logits1[0]
            + logits2[1]
        )

        score_bf = (
            logits1[1]
            + logits2[0]
        )

        score_ee = (
            logits1[2]
            + logits2[2]
        )

        scores = torch.stack([
            score_fb,
            score_bf,
            score_ee
        ])

        # Pair预测
        pred_pair = (
            torch.argmax(scores)
            .item()
        )

        # Top1和Top2之间差多少
        sorted_scores = torch.sort(
            scores,
            descending=True
        ).values

        pair_margin = (
            sorted_scores[0]
            - sorted_scores[1]
        ).item()

        gold1 = int(
            analysis_df.loc[
                idx1,
                "gold_id"
            ]
        )

        gold2 = int(
            analysis_df.loc[
                idx2,
                "gold_id"
            ]
        )

        if gold1 == 0 and gold2 == 1:
            gold_pair = 0

        elif gold1 == 1 and gold2 == 0:
            gold_pair = 1

        elif gold1 == 2 and gold2 == 2:
            gold_pair = 2

        else:
            # 如果数据里出现非预期组合
            # 不纳入Pair结构分析
            continue

        pair_results.append({
            "pair_id": pair_id,

            "text1": analysis_df.loc[
                idx1, "text1"
            ],

            "text2": analysis_df.loc[
                idx1, "text2"
            ],

            "gold_pair": gold_pair,
            "pred_pair": pred_pair,

            "score_fb": score_fb.item(),
            "score_bf": score_bf.item(),
            "score_ee": score_ee.item(),

            "margin": pair_margin,

            "correct": (
                pred_pair == gold_pair
            )
        })

    pair_df = pd.DataFrame(
        pair_results
    )

    pair_name = {
        0: "FB",
        1: "BF",
        2: "EE"
    }

    pair_df["gold_pair_name"] = (
        pair_df["gold_pair"]
        .map(pair_name)
    )

    pair_df["pred_pair_name"] = (
        pair_df["pred_pair"]
        .map(pair_name)
    )

    print("\n========== 完整Pair分析 ==========")

    print(
        f"完整Pair数量：{len(pair_df)}"
    )

    print(
        f"Pair判断正确："
        f"{pair_df['correct'].sum()}"
    )

    print(
        f"Pair判断错误："
        f"{(~pair_df['correct']).sum()}"
    )

    print("\n各类Pair：")

    print(
        pd.crosstab(
            pair_df["gold_pair_name"],
            pair_df["pred_pair_name"]
        )
    )

    print(
        "\n正确Pair平均 margin："
        f"{pair_df[pair_df['correct']]['margin'].mean():.4f}"
    )

    print(
        "错误Pair平均 margin："
        f"{pair_df[~pair_df['correct']]['margin'].mean():.4f}"
    )

    # ==========================================
    # 6. NEGATIVE_OTHER分析
    # ==========================================

    # N自己的logit
    n_logits = all_logits[:, 3]

    # F/B/E里最高的logit
    relation_logits = (
        all_logits[:, :3]
        .max(dim=1)
        .values
    )

    # > 0：更支持N
    # < 0：更支持F/B/E
    n_margin = (
        n_logits
        - relation_logits
    )

    analysis_df["n_margin"] = (
        n_margin.numpy()
    )

    negative_df = analysis_df[
        analysis_df["gold_id"] == 3
    ].copy()

    print("\n========== NEGATIVE_OTHER ==========")

    print(
        f"N总数：{len(negative_df)}"
    )

    print(
        f"N正确："
        f"{negative_df['pair_correct'].sum()}"
    )

    wrong_negative = negative_df[
        ~negative_df["pair_correct"]
    ]

    print(
        f"N错误：{len(wrong_negative)}"
    )

    print("\nN主要被预测成：")

    print(
        wrong_negative[
            "pair_pred"
        ].value_counts()
    )

    print(
        "\nN正确样本平均margin："
        f"{negative_df[negative_df['pair_correct']]['n_margin'].mean():.4f}"
    )

    print(
        "N错误样本平均margin："
        f"{wrong_negative['n_margin'].mean():.4f}"
    )

    # ==========================================
    # 7. E vs F/B边界
    # ==========================================

    # E logit
    e_logit = all_logits[:, 2]

    # F / B里更高的那个
    directional_logit = (
        all_logits[:, :2]
        .max(dim=1)
        .values
    )

    # > 0 更偏向E
    # < 0 更偏向F/B
    e_margin = (
        e_logit
        - directional_logit
    )

    analysis_df["e_margin"] = (
        e_margin.numpy()
    )

    # F/B被预测成E
    fb_to_e = analysis_df[
        (
            analysis_df["gold_id"]
            .isin([0, 1])
        )
        &
        (
            analysis_df["pair_pred_id"]
            == 2
        )
    ]

    # E被预测成F/B
    e_to_fb = analysis_df[
        (
            analysis_df["gold_id"]
            == 2
        )
        &
        (
            analysis_df["pair_pred_id"]
            .isin([0, 1])
        )
    ]

    print("\n========== E <-> F/B 边界 ==========")

    print(
        f"F/B -> E：{len(fb_to_e)}"
    )

    print(
        f"E -> F/B：{len(e_to_fb)}"
    )

    if len(fb_to_e) > 0:
        print(
            "F/B -> E 平均 e_margin："
            f"{fb_to_e['e_margin'].mean():.4f}"
        )

    if len(e_to_fb) > 0:
        print(
            "E -> F/B 平均 e_margin："
            f"{e_to_fb['e_margin'].mean():.4f}"
        )

    # ==========================================
    # 8. 保存结果
    # ==========================================

    analysis_dir = (
        RUN_DIR
        / "deep_analysis"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 全部样本
    analysis_df.to_csv(
        analysis_dir
        / "all_dev_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Raw错 -> Pair对
    analysis_df[
        (~analysis_df["raw_correct"])
        & analysis_df["pair_correct"]
    ].to_csv(
        analysis_dir
        / "pair_fixed_examples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Raw对 -> Pair错
    analysis_df[
        analysis_df["raw_correct"]
        & (~analysis_df["pair_correct"])
    ].to_csv(
        analysis_dir
        / "pair_broken_examples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Pair级别结果
    pair_df.to_csv(
        analysis_dir
        / "pair_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # N错误
    wrong_negative.to_csv(
        analysis_dir
        / "negative_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # F/B -> E
    fb_to_e.to_csv(
        analysis_dir
        / "fb_to_e_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # E -> F/B
    e_to_fb.to_csv(
        analysis_dir
        / "e_to_fb_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\n分析结果已保存到："
        f"{analysis_dir.resolve()}"
    )

    # ==========================================
    # 继续分析：18个错误Pair
    # ==========================================

    wrong_pairs = pair_df[
        pair_df["correct"] == False
        ].copy()

    # 按 margin 从小到大排序
    # margin越小，说明模型越犹豫
    wrong_pairs = wrong_pairs.sort_values(
        by="margin",
        ascending=True
    )

    print("\n========== 所有错误Pair（按margin从小到大） ==========")

    print(
        wrong_pairs[
            [
                "pair_id",
                "text1",
                "text2",
                "gold_pair_name",
                "pred_pair_name",
                "score_fb",
                "score_bf",
                "score_ee",
                "margin"
            ]
        ].to_string(index=False)
    )

    print("\n========== 错误Pair margin统计 ==========")

    print(
        wrong_pairs["margin"].describe()
    )

    # 看错误主要是哪种类型
    print("\n========== 错误Pair类型 ==========")

    wrong_type = (
        wrong_pairs
        .groupby([
            "gold_pair_name",
            "pred_pair_name"
        ])
        .size()
        .reset_index(name="count")
    )

    print(
        wrong_type.to_string(index=False)
    )

    # 单独看 margin 很小的错误
    print("\n========== margin < 2 的错误Pair ==========")

    small_margin = wrong_pairs[
        wrong_pairs["margin"] < 2
        ]

    print(
        small_margin[
            [
                "pair_id",
                "text1",
                "text2",
                "gold_pair_name",
                "pred_pair_name",
                "score_fb",
                "score_bf",
                "score_ee",
                "margin"
            ]
        ].to_string(index=False)
    )

    # 单独看模型非常自信但依然错的Pair
    print("\n========== margin > 5 的高置信错误Pair ==========")

    high_margin = wrong_pairs[
        wrong_pairs["margin"] > 5
        ]

    print(
        high_margin[
            [
                "pair_id",
                "text1",
                "text2",
                "gold_pair_name",
                "pred_pair_name",
                "score_fb",
                "score_bf",
                "score_ee",
                "margin"
            ]
        ].to_string(index=False)
    )

    # 保存
    wrong_pairs.to_csv(
        analysis_dir / "wrong_pairs_sorted_by_margin.csv",
        index=False,
        encoding="utf-8-sig"
    )

if __name__ == "__main__":
    checkAndprocess_data()
    # train_model_real_pair_ce()
    best_epoch, best_score, best_preds = evaluate_pair_aware_model()
    submission_output(best_preds)