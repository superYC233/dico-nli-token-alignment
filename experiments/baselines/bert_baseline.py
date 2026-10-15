
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
from transformers import BertTokenizer, BertModel

# 准备数据集
train_data = pd.read_csv('./data/train/dico_nli_train_track1_participant_labeled.csv')
dev_data = pd.read_csv('./data/dev/dico_nli_dev_track1_participant_labeled.csv')
# 准备bert模型
model_path = 'D:/BaiduNetdiskDownload/PretrainedModel/models/google-bert--bert-base-uncased/snapshots/master'
my_tokenizer = BertTokenizer.from_pretrained(model_path)
my_model = BertModel.from_pretrained(model_path)

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

# todo 2. 自定义短语对数据集
class DiCoDataset(Dataset):
    def __init__(self, df, bert_tokenizer, max_length):
        self.df = df
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text1 = row['text1']
        text2 = row['text2']
        label = row['label_id']

        encoded = self.bert_tokenizer(
            text1,
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        token_type_ids = encoded['token_type_ids'].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask,token_type_ids, label

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

# todo 4. 定义 BERT 四分类模型
class BertDiCoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 保存 BERT 模型
        self.bert_model = my_model
        # 2. 定义 Dropout
        self.dropout = nn.Dropout(0.2)
        # 3. 定义四分类层
        self.output = nn.Linear(self.bert_model.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 1. 将三个输入传给 BERT
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 2. 取出 [CLS] 位置的向量
        last_hidden_state = bert_output.last_hidden_state
        cls_feature = last_hidden_state[:, 0,:]
        # 3. 经过 Dropout
        cls_feature  = self.dropout(cls_feature)
        # 4. 经过分类层并返回
        output = self.output(cls_feature)

        return output

# todo 5. 训练 BERT 四分类模型
def train_model():
    # 1. 选择 CPU 或 GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 获取 train_loader
    train_dataloader, _ = create_dataloaders()

    # 3. 创建模型并移到对应设备
    bert_model = BertDiCoModel().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        bert_model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    # 5. 编写训练循环
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        bert_model.train()

        for input_ids, attention_mask, token_type_ids, label in tqdm(train_dataloader):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            bert_output = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            loss = criterion(bert_output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

         # 当前epoch平均loss
        avg_loss = total_loss / len(train_dataloader)

        print(f'epoch:{epoch + 1}/{epochs} \n loss:{avg_loss:.4f}')
        # 每轮保存一次模型
        torch.save(bert_model.state_dict(), f'./model/bert_dico_epoch_{epoch + 1}.pth')

# todo 6. 评估保存的模型
def evaluate_model(negative_penalty=0.0):
    # 1. 选择 device，获取 dev_dataloader
    _, dev_dataloader = create_dataloaders()
    best_score = -1.0
    best_epoch = None

    for epoch in range(8, 9):
        # 2. 创建 BertDiCoModel，并加载保存的参数
        bert_model = BertDiCoModel().to(device)
        bert_model.load_state_dict(torch.load(f'./model/bert_dico_epoch_{epoch}.pth'))
        # 3. 切换到评估模式
        bert_model.eval()
        # 4. 遍历开发集，收集预测类别和正确标签
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, label in tqdm(dev_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                label = label.to(device)

                bert_output = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                bert_output[:, 3] -= negative_penalty
                pred = torch.argmax(bert_output, dim=1)
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

# todo 7. 导出开发集预测结果
def submission_output(best_preds, negative_penalty):
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
        f"./dev_predictions_penalty_{negative_penalty}.csv",
        index=False
    )

# todo n. 测试
if __name__ == '__main__':
    checkAndprocess_data()

    for penalty in [0.0, 0.5, 1, 1.5]:
        best_epoch, best_score,best_preds = evaluate_model(negative_penalty=penalty)
        print(f'negative_penalty={penalty}，Weighted F1={best_score:.4f}')
        submission_output(best_preds, penalty)