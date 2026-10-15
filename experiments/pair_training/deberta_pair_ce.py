
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



# pair_weight 0.1
# RUN_DIR = Path(f"./model/deberta_pair_ce_w0p1_seed_{SEED}")

#  pair_weight 0.5
# RUN_DIR = Path(f"./model/deberta_pair_ce_w0p5_seed_{SEED}")

# pair_weight 0.7
RUN_DIR = Path(f"./model/deberta_pair_ce_w0p7_seed_{SEED}")

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


# todo 5. 四分类CE + pair-level CE
def train_model_pair_ce():
    # 1. 获取正向、反向的dataloader
    train_dataloader = create_consistency_train_loader(batch_size=16)

    # 2. 创建模型
    deberta_model = debertaDiCoModel().to(device)

    # 3. 创建损失函数和优化器
    # CrossEntropyLoss 同时用于：
    # 1. 普通四分类 CE
    # 2. Pair 三分类 CE
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        deberta_model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )

    epochs = 10
    # loss = classification_loss + pair_weight * pair_loss
    pair_weight = 0.7

    # 4. 定义训练
    for epoch in range(epochs):
        deberta_model.train()

        # 用于统计每轮平均损失
        total_classification = 0.0
        total_pair = 0.0

        # row_count:  一共参与普通四分类 CE 的样本数量
        # pair_count: 一共参与 Pair CE 的 F/B/E 样本数量
        row_count = 0
        pair_count = 0

        for (input_ids, attention_mask, reverse_input_ids,
             reverse_attention_mask, label)in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            reverse_input_ids = reverse_input_ids.to(device)
            reverse_attention_mask = reverse_attention_mask.to(device)
            label = label.to(device)

            # 5. 正向输入、反向输入分别经过同一个 DeBERTa 得到原始分数logits
            forward_logits = deberta_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            reverse_logits = deberta_model(
                input_ids=reverse_input_ids,
                attention_mask=reverse_attention_mask
            )

            # 6. 普通四分类 CE
            classification_loss = criterion(forward_logits, label)

            # 7. 找出能进行Pair CE的样本
            reversible_mask = label != 3

            # 当前batch有多少条不含 NEGATIVE_OTHER的
            valid_count = int(reversible_mask.sum().item())

            # 8. 计算Pair CE
            if valid_count > 0:
                # 9. 删除真实值为的 NEGATIVE样本(不可能是pair)
                # 假设：
                # forward_logits [16, 4]
                # 若当前有 13 条可逆样本
                # =>  forward_valid: [13, 4]
                # => reverse_valid: [13, 4]
                #
                forward_valid = forward_logits[reversible_mask]
                reverse_valid = reverse_logits[reversible_mask]

                # 10. 构造三种合法 Pair 的联合分数
                score_fb = forward_valid[:, 0] + reverse_valid[:, 1]
                score_bf = forward_valid[:, 1] + reverse_valid[:, 0]
                score_ee = forward_valid[:, 2] + reverse_valid[:, 2]

                # 11. 把三个分数组成 Pair 三分类 logits
                '''
                score_fb = [8.0, 1.0, 0.5]
                score_bf = [2.0, 7.0, 1.0]
                score_ee = [0.5, 1.0, 6.0]
                =>
                pair_logits =
                [
                    [8.0, 2.0, 0.5],   # 第1条：FB / BF / EE
                    [1.0, 7.0, 1.0],   # 第3条：FB / BF / EE
                    [0.5, 1.0, 6.0]    # 第4条：FB / BF / EE
                ]
                '''
                pair_logits = torch.stack([score_fb, score_bf, score_ee],dim=1)

                # 12. 构建 Pair 正确答案
                '''
                例如:
                label = [0, 3, 1, 2]
                reversible_mask = [True, False, True, True]
                =>
                pair_targets = [0, 1, 2]

                在 Pair 三分类里:
                0 -> FB
                1 -> BF
                2 -> EE
                所以对应上面的 pair_logits:
                pair_logits =
                [
                    [8.0, 2.0, 0.5],   # 正确答案是 FB -> target=0
                    [1.0, 7.0, 1.0],   # 正确答案是 BF -> target=1
                    [0.5, 1.0, 6.0]    # 正确答案是 EE -> target=2
                ]

                因此：
                pair_targets = [0, 1, 2]
                '''
                pair_targets = label[reversible_mask]

                pair_loss = criterion(pair_logits, pair_targets)

            else:
                # 如果这个 batch 恰好全是 NEGATIVE_OTHER，
                # 创建一个:标量 0 并且设备、dtype 与 forward_logits 一样。
                '''
                pair_loss = torch.tensor(
                    0.0,
                    device=forward_logits.device,
                    dtype=forward_logits.dtype
                )
                '''
                pair_loss = forward_logits.new_zeros(())

            loss = classification_loss + pair_weight * pair_loss

            # 15. 梯度清理, 反向传播, 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 16. 统计loss
            batch_rows = label.size(0)

            total_classification += (classification_loss.item() * batch_rows)
            total_pair += (pair_loss.item() * valid_count)

            row_count += batch_rows
            pair_count += valid_count

        # 17. 计算整个epoch 的平均loss
        avg_classification_loss = total_classification / row_count
        avg_pair_loss = total_pair / pair_count

        # 18. 打印当前训练情况
        print(
            f"\n第 {epoch + 1} 轮\n"
            f"四分类 CE: {avg_classification_loss:.6f}\n"
            f"Pair CE: {avg_pair_loss:.6f}\n"
            f"乘权重后的 Pair CE: "
            f"{pair_weight * avg_pair_loss:.6f}"
        )

        # 19. 保存当前 epoch

        torch.save(
            deberta_model.state_dict(),
            RUN_DIR / f"epoch_{epoch + 1}.pth"
        )


# todo 6. 评估模型 + pair-aware 后处理
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
    for epoch in range(1,11):
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

# todo 10. 对比新旧模型的预测结果
def compare_predictions():
    project_dir = (PROJECT_ROOT)

    # 1. 读取标准答案
    gold = pd.read_csv(
        project_dir / "data/dev/dico_nli_dev_track1_reference.csv"
    )[["instance_id", "label"]]

    gold = gold.rename(columns={"label": "gold_label"})

    # 2. 读取旧模型预测：普通 CE + pair-aware
    old_preds = pd.read_csv(
        project_dir
        / "model/deberta_seed_42/dev_predictions_pair_aware.csv"
    ).rename(columns={"label": "old_pred"})

    # 3. 读取新模型预测：四分类 CE + Pair CE + pair-aware
    new_preds = pd.read_csv(
        project_dir
        / "model/deberta_pair_ce_w0p5_seed_42/dev_predictions_pair_aware.csv"
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
        / "results/deberta_pair_ce_w0p5_seed_42"
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

# todo n. 测试
if __name__ == '__main__':
    checkAndprocess_data()

    train_model_pair_ce()

    # 找出 pair-aware F1 最高的一轮
    best_epoch, best_score, best_preds = evaluate_pair_aware_model()

    # 保存这一轮的开发集预测
    submission_output(best_preds)

    print(f"最佳轮次：{best_epoch}")
    print(f"最佳 Pair-aware F1：{best_score:.4f}")

    # compare_predictions()
