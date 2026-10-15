
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

# 诊断实验：只跑 1 轮，用来观察 classification loss / consistency loss 大小
# RUN_DIR = Path(f"./model/deberta_diagnostic_seed_{SEED}")

# 一致性训练：consistency_weight = 1.0
# RUN_DIR = Path(f"./model/deberta_consistency_seed_{SEED}")

# 一致性训练：consistency_weight = 0.1
# RUN_DIR = Path(f"./model/deberta_consistency_w0p1_seed_{SEED}")

# 普通 DeBERTa baseline：没有 consistency loss
# best_epoch: 8
RUN_DIR = Path(f"./model/deberta_seed_{SEED}")

RUN_DIR.mkdir(parents=True, exist_ok=True)

# 准备数据集
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')

model_name = "microsoft/deberta-v3-base"
cache_path = r"D:\BaiduNetdiskDownload\PretrainedModel\models"

my_tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path
)

# 模型统一使用 float32
my_model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_path,
    torch_dtype=torch.float32
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
        ['pair_id', 'text1','text2','label']
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
        'FORWARD_ENTAILMENT' : 0,
        'BACKWARD_ENTAILMENT' : 1,
        'EQUIVALENCE' : 2,
        'NEGATIVE_OTHER' : 3
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

# todo 2.1 为一致性训练提供正反两个方向输入
class ConsistencyDataset(DiCoDataset):
    # 继承后直接调用父类的DiCoDataset.__init__()
    def __getitem__(self, index):
        # 调取原来的方法, 获取正向输入和正确标签
        input_ids, attention_mask, label = super().__getitem__(index)

        row = self.df.iloc[index]
        text1 = row['text1']
        text2 = row['text2']

        reverse_encoded = self.Deberta_tokenizer(
            text2,
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        reverse_input_ids = reverse_encoded['input_ids'].squeeze(0)
        reverse_attention_mask = reverse_encoded['attention_mask'].squeeze(0)

        return input_ids, attention_mask, reverse_input_ids,reverse_attention_mask, label

# todo 3. 创建训练集和开发集的 DataLoader
def create_dataloaders(batch_size=16):
    # 1. 创建 train_dataset
    train_dataset = DiCoDataset(train_data,my_tokenizer,max_length=128)
    # 2. 创建 dev_dataset
    dev_dataset = DiCoDataset(dev_data,my_tokenizer,max_length=128)
    # 3. 创建 train_loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 4. 创建 dev_loader
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # 5. 返回两个 loader
    return train_dataloader, dev_dataloader

# todo 3.1. 创建一致性训练的 DataLoader
def create_consistency_train_loader(batch_size=16):
    dataset = ConsistencyDataset(train_data,my_tokenizer,max_length=128)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_dataloader

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
        cls_feature = last_hidden_state[:, 0,:]
        # 3. 经过 Dropout
        cls_feature  = self.dropout(cls_feature)
        # 4. 经过分类层并返回
        output = self.output(cls_feature)

        return output

# todo 5. 训练 Deberta 四分类模型
def train_model():
    # 1. 选择 CPU 或 GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 获取 train_loader
    train_dataloader, _ = create_dataloaders()

    # 3. 创建模型并移到对应设备
    deberta_model = debertaDiCoModel().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    # 5. 编写训练循环
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        deberta_model.train()

        for input_ids, attention_mask, label in tqdm(train_dataloader):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            deberta_output = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(deberta_output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

         # 当前epoch平均loss
        avg_loss = total_loss / len(train_dataloader)

        print(f'epoch:{epoch + 1}/{epochs} \n loss:{avg_loss:.4f}')
        # 每轮保存一次模型
        torch.save(
            deberta_model.state_dict(),
            RUN_DIR / f"epoch_{epoch + 1}.pth"
        )

# todo 5.1 训练 DeBERTa：分类损失 + 正反方向一致性损失
def train_model_consistency():
    train_dataloader = create_consistency_train_loader(batch_size=16)
    # 3. 创建模型并移到对应设备
    deberta_model = debertaDiCoModel().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    # 5. 编写训练循环
    epochs = 1

    # 前后不一致的罚分
    consistency_weight = 1.0

    for epoch in range(epochs):
        total_loss = 0
        total_classification = 0
        total_consistency = 0

        deberta_model.train()

        for (input_ids, attention_mask, reverse_input_ids,
             reverse_attention_mask, label) in tqdm(train_dataloader):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            reverse_input_ids = reverse_input_ids.to(device)
            reverse_attention_mask = reverse_attention_mask.to(device)
            label = label.to(device)

            deberta_output = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            deberta_reverse_output = deberta_model(
                input_ids=reverse_input_ids,
                attention_mask=reverse_attention_mask
            )

            # 只保留真实标签不是 NEGATIVE_OTHER 的样本
            # 因为 0/1/2 三种关系有明确的正反对应关系，label=3 不参与一致性约束
            reversible_mask = label != 3

            # 正向输入 (text1, text2) 的预测概率
            # deberta_output[reversible_mask]：
            #   先删掉“真实标签为3”的那些样本，只保留可逆关系样本
            # softmax(dim=1)：
            #   把每条样本的4个 logits 转成4个类别概率
            # [:, [0,1,2]]：
            #   所有样本都保留，只取前3个类别概率
            #   即 [FORWARD, BACKWARD, EQUIVALENCE]
            forward_probs = torch.softmax(
                deberta_output[reversible_mask], # 删掉真实标签为3的
                dim = 1
            )[: , [0,1,2]]

            # 反向输入 (text2, text1) 的预测概率
            # 同样先删掉真实标签为3的样本
            # 反向以后类别对应关系是：
            #   FORWARD <-> BACKWARD
            #   EQUIVALENCE <-> EQUIVALENCE
            # 所以把原来的类别顺序 [0,1,2]
            # 重排成 [1,0,2]，让它能和 forward_probs 一一对应
            reverse_probs = torch.softmax(
                deberta_reverse_output[reversible_mask],
                dim = 1
            )[:, [1,0,2]]

            # loss = criterion(deberta_output, label)

            # 计算总损失

            # 1. 分类损失
            # deberta_output: 模型正向输入的4分类原始分数，形状 [batch_size, 4]
            # label: 每条样本的真实标签，形状 [batch_size]
            # 四种类别全部参与分类训练，包括 NEGATIVE_OTHER
            # 作用：模型预测错类别时，就产生分类惩罚
            classification_loss = criterion(deberta_output, label)

            # 2. 一致性损失, 要求正反方向对应的概率接近
            # reversible_mask = label != 3
            # 例如：
            # label = [0, 3, 1, 2]
            # reversible_mask = [True, False, True, True]
            # .any()：只要这一批里至少有一个 True，就返回 True
            # 也就是：这一批里至少有一条可逆关系样本（0/1/2）
            if reversible_mask.any(): # 这一批数据里，有没有至少一个 True

                # forward_probs 和 reverse_probs 已经提前去掉了真实标签为 NEGATIVE_OTHER 的样本
                # 所以这里可以直接比较正向和反向对应的概率
                consistency_loss = (
                    (forward_probs - reverse_probs)  # 正向概率 - 对应的反向概率
                    .pow(2)  # 概率差平方，差得越大惩罚越大
                    .sum(dim=1)  # 每条样本的3个类别差异加起来
                    .mean()  # 对这一批样本取平均
                )
            else:
                # 如果这一整个 batch 全都是 NEGATIVE_OTHER
                # 就没有任何可逆关系可以计算一致性损失
                # 因此把一致性损失设置为 0
                # new_zeros(())：创建一个和 deberta_output 同设备、同类型的标量 0
                consistency_loss = deberta_output.new_zeros(())

            # 3. 总损失
            # classification_loss：
            #     要求模型把类别预测正确
            # consistency_loss：
            #     要求 (text1,text2) 和 (text2,text1)
            #     的对应预测尽量保持一致
            # consistency_weight：
            #     控制一致性损失对总损失的影响程度
            #     如果 consistency_weight = 1.0，就是正常加入一份一致性损失
            loss = classification_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_classification += classification_loss.item()
            total_consistency += consistency_loss.item()

        # 当前epoch平均loss
        avg_loss = total_loss / len(train_dataloader)

        num_batches = len(train_dataloader)

        avg_classification = total_classification / num_batches
        avg_consistency = total_consistency / num_batches
        weighted_consistency = consistency_weight * avg_consistency

        print(
            f"第 {epoch + 1} 轮\n"
            f"分类损失：{avg_classification:.6f}\n"
            f"原始一致性损失：{avg_consistency:.6f}\n"
            f"乘权重后的一致性损失：{weighted_consistency:.6f}"
        )

        # 每轮保存一次模型
        torch.save(
            deberta_model.state_dict(),
            RUN_DIR / f"epoch_{epoch + 1}.pth"
        )

# todo 6. 评估保存的模型
def evaluate_model(negative_penalty=0.0):
    # 1. 选择 device，获取 dev_dataloader
    _, dev_dataloader = create_dataloaders()
    best_score = -1.0
    best_epoch = None

    for epoch in range(1,11):
        # 2. 创建 DebertaDiCoModel，并加载保存的参数
        deberta_model = debertaDiCoModel().to(device)
        deberta_model.load_state_dict(
            torch.load(
                RUN_DIR / f"epoch_{epoch}.pth",
                map_location=device,
                weights_only=True
            )
        )
        # 3. 切换到评估模式
        deberta_model.eval()
        # 4. 遍历开发集，收集预测类别和正确标签
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, label in tqdm(dev_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)

                deberta_output = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                deberta_output[:, 3] -= negative_penalty
                pred = torch.argmax(deberta_output, dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 5. 计算、打印并返回 Weighted F1
            score = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

            print(f'第 {epoch} 轮：Weighted F1 = {score:.4f}')

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_preds = all_preds.clone()
                best_labels = all_labels.clone()

    print(classification_report(
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
    ))

    names = ["FORWARD", "BACKWARD", "EQUIVALENCE", "NEGATIVE"]

    # 通过混淆矩阵，查看预测和选择
    # 发现大量其他类别 被错误预测为 NEGATIVE
    matrix = confusion_matrix(best_labels, best_preds, labels=[0, 1, 2, 3])

    print(pd.DataFrame(
        matrix,
        index=[f"真实_{name}" for name in names],
        columns=[f"预测_{name}" for name in names]
    ).to_string())

    return best_epoch, best_score, best_preds

# todo 6.1 评估模型 + pair-aware 后处理
def evaluate_pair_aware_model():

    # 1. 获取开发集
    _, dev_dataloader = create_dataloaders()

    # 创建模型
    deberta_model = debertaDiCoModel().to(device)

    best_score = -1.0
    best_epoch = None
    best_preds = None
    best_labels = None

    # 2. 依次评估 10 个 epoch
    for epoch in range(8,9):
        # 加载当前 epoch 的模型参数
        deberta_model.load_state_dict(
            torch.load(
                RUN_DIR / f"epoch_{epoch}.pth",
                map_location=device,
                weights_only=True
            )
        )

        deberta_model.eval()

        # 3. 收集整个 dev 的 logits 和 label
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, label in tqdm(dev_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)

                # 模型输出
                # [batch_size, 4]
                deberta_output = deberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 这里不能直接存 argmax 后的结果
                # pair-aware 需要使用每个类别的4个原始分数
                all_logits.append(deberta_output.cpu())
                all_labels.append(label.cpu())

        # 4. 拼接所有 batch
        # [660, 4]
        all_logits = torch.cat(all_logits, dim=0)

        # [660]
        all_labels = torch.cat(all_labels, dim=0)

        # 7. 计算pair-aware 后的 Weighted F1
        final_preds = pair_aware(all_logits, dev_data)

        score = f1_score(all_labels, final_preds, average="weighted", zero_division=0)

        print(f'第{epoch}轮: '
              f'Pair-aware Weighted F1 = {score:.4f}'
        )

        # 8. 保存最好的一轮
        if score > best_score:
            best_score = score
            best_epoch = epoch

            best_preds = final_preds.clone()
            best_labels = all_labels.clone()

    # 9. 打印最终结果
    print(f'\n 最佳 epoch:{best_epoch}'
          f'\n 最佳 Pair-aware Weighted F1 = {best_score:.4f}')


    assert best_preds is not None

    # 10. 查看 pair-aware 后各类别表现
    print(classification_report(
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
    ))

    # 11. 混淆矩阵
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

    print(pd.DataFrame(
        matrix,
        index=[f"真实_{name}" for name in names],
        columns=[f"预测_{name}" for name in names]
    ).to_string())

    return best_epoch, best_score, best_preds

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
# todo n. 测试
if __name__ == '__main__':
    checkAndprocess_data()

    train_model()
    # train_model_consistency()

    # for penalty in [0.0, 0.5, 1, 1.5]:
    # for penalty in [0.0]:
    #     best_epoch, best_score,best_preds = evaluate_model(negative_penalty=penalty)
    #     print(f'best_epoch:{best_epoch} best_score:{best_score}')
    #     print(f'negative_penalty={penalty}，Weighted F1={best_score:.4f}')
    #     submission_output(best_preds, penalty)
    best_epoch, best_score, best_preds = evaluate_pair_aware_model()

    analyze_pair_aware_errors(best_preds)
    submission_output(best_preds)