
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
import torch.nn.functional as F

# todo 0. 固定随机种子，设置本次实验目录
SEED = 42
# SEED = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RUN_DIR = (PROJECT_ROOT / 'model') / f"deberta_nli_base_token_alignment_clean_seed_{SEED}"

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


# todo 4. DeBERTa 四分类 + Token级对齐比较分支
class debertaDiCoModel(nn.Module):
    # 1. 初始化模型
    def __init__(self):
        super().__init__()

        # 1. DeBERTa 主干模型
        # 输入: input_ids, attention_mask => 输出: last_hidden_state
        self.deberta_model = my_model

        # 768
        hidden_size = self.deberta_model.config.hidden_size

        # 四分类分支
        # 之前: DeBERTa -> CLS向量 -> Dropout -> Linear -> [batch_size,4]
        self.dropout = nn.Dropout(p=0.2)

        self.output = nn.Linear(hidden_size, 4)

        # 3. 保存cls_id和sep_id
        # 后面通过 CLS/SEP 的位置 来判断 哪些token属于A, 哪些属于 B
        self.cls_id = my_tokenizer.cls_token_id
        self.sep_id = my_tokenizer.sep_token_id

        # 4. 新增 Token 对齐比较分支
        # 目的：
        # 原来的 CLS 更偏向：
        #   "A 和 B 整体是什么关系？"
        #
        # 新分支专门观察：
        #   "A 和 B 到底哪些 token 对应？"
        #   "匹配以后还有哪些差异？"
        # 例如：
        # A = "train station" B = "station"
        # station <-> station   很匹配
        # train   <-> ???       没有完全对应
        #
        # 这可能帮助模型判断：
        # 两句话虽然很相似，但不一定是 EQUIVALENCE。
        #
        # torch.random.fork_rng(devices=[])
        # 原来代码：
        # 分类头用了 1,2,3
        # 后面的 Dropout 用 4,5,6
        #
        # 现在你中间突然加一个新层：
        # 分类头用了 1,2,3
        # 新层用了 4,5,6
        # 后面的 Dropout 就只能用 7,8,9
        # fork_rng ≈ “新层自己随机，别影响外面。”
        # 这样做可以判断是否 compare_layer真的有效, 而非 只是这次随机初始化/Dropout碰巧更好
        with torch.random.fork_rng(devices=[]):
            # 4.1 Token 投影层
            # 将[batch, 128, 768] => [batch, 128, 128]
            # 降维, 减少后面 token 对齐的计算量, 同时保留主要信息
            '''
            768 保留更多信息，但计算更慢、显存更大，也可能保留很多对“对齐”没用的噪声
            128 会丢一点信息，但强迫模型提取更有用的匹配特征，还更省计算
            没降维：
            128 × 128 个 token 对
            每对比较 768 个特征
            降维后：
            128 × 128 个 token 对
            每对只比较 128 个特征
            '''
            self.align_projection = nn.Linear(hidden_size, 128)

            # 4.2 Token 比较层
            '''
              把当前 token 和它在另一个句子中匹配到的 token 做进一步比较
            假设：
              tokens  = 当前句子的 token 特征
              aligned = 另一个句子中与它匹配的 token 特征
            每个都是 128 维
            构造 4 组特征：
              1. tokens            当前 token 自己
              2. aligned           对方匹配到的 token
              3. tokens - aligned  两者的差异
              4. tokens * aligned  两者共同/相似的特征
            拼接后：
              128 + 128 + 128 + 128 = 512
            再通过 Linear：
              [batch, seq_len, 512]
                       ↓
              [batch, seq_len, 128]
            GELU：
              增加非线性，让模型学习更复杂的比较关系
            最终得到：
              每个 token “和对方比较之后”的 128 维特征
            '''
            self.compare_layer = nn.Sequential(
                nn.Linear(4 * 128, 128),
                nn.ReLU()
            )

            # 4.3 Token对齐分支, 最终分类器
            '''
            A 的比较信息：
               mean pooling [128]
               max pooling  [128]
               => [256]
            B 同样：
               => [256]
            最后 A + B：
            [256] + [256]
                  ↓
                [512]
            再输出四分类分数：
            [512] → [4]
            '''
            self.match_output = nn.Sequential(
                nn.LayerNorm(128 * 4),  # 对每个样本的这 512 个特征做归一化，让它们的数值分布更稳定
                nn.Dropout(p=0.2),
                nn.Linear(128 * 4, 4)
            )

            # 4.4 让 Token 对齐分支刚开始输出 0
            '''
             最终结果:
             final_logits = cls_logits + match_logits

             match_output 最后一层是 Linear(512, 4)
             默认情况下它的 weight 和 bias 是随机初始化的，
             那么刚开始训练时 match_logits 也是随机的，
             可能会干扰原来的 CLS 分类结果。

             所以把最后一层参数先全部设成 0:

             weight = 0
             bias   = 0
             match_logits = x × weight + bias
             上一层的输出，会变成下一层的输入 x
             只要最后一层的weight和bias是0, 那么无论前面的 x 是什么, 最后match_logits都是0 

             这样训练刚开始时：
             match_logits = [0, 0, 0, 0]
             所以：
             final_logits = cls_logits + 0
                          = cls_logits
             注意：
             这里只是让“新分支的最终输出”一开始为 0，
             前面的 Token 对齐、比较、pooling 仍然正常计算。
             后面训练时，这个 Linear 的参数会逐渐被更新，
             match_logits 也会慢慢变成有用的修正分数。

             第1个 batch：
             match 最后一层从 0 开始学习
             第2、3、4...个 batch：
             match_output 已经不再是 0
             开始正常参与预测
            '''

            nn.init.zeros_(self.match_output[-1].weight)
            nn.init.zeros_(self.match_output[-1].bias)

    # 2. Token对齐函数
    def _align(self, tokens, scores, target_mask):
        """
        函数作用：
            让每个 token 在另一个文本中寻找匹配 token。
        例如：
            A = "train station" B = "station"
        对 A 中的 train：
            train -> B中的station
        对 A 中的 station：
            station -> B中的station
        但不是直接选一个 token，
        而是根据相似度做加权求和。
        参数形状：
            tokens: [batch, seq_len, 128]
            scores: [batch, seq_len, seq_len]
            例如:   [16,A的token数,B的token数] = [16, 128, 128]
                含义: 每一个 token 与另外 128 个位置的相似度。
            target_mask: [batch, seq_len]
                指定允许匹配哪些 token。
                例如 A 匹配 B 时：
                target_mask = mask_b
        返回: aligned: [batch, seq_len, 128]
             每个 token 都得到一个
             "从另一段文本中找来的匹配向量"。
        """

        # [16,128] => [16,1,128]
        # 16: batch 中有 16 个样本，每个样本都有自己的一份 mask
        # 1： 对于每个样本，只需要准备 1 行 B 的 mask，
        #    后面这 1 行会给 A 的 128 个 token 重复使用
        # 128:B 的 128 个 token
        mask = target_mask.unsqueeze(1)

        # 把 "不允许参与匹配" 的位置屏蔽掉
        # mask 中: True  = 这个位置可以参与匹配 False = 这个位置不能参与匹配
        # 例如：
        # A自己的位置  CLS  SEP  PAD 都不能参与匹配
        # mask = [True, True, False, False]
        # ~mask 表示取反：
        # [False, False, True, True]
        #
        # masked_fill 的意思：
        #   在 ~mask == True 的位置，把 scores 替换掉
        #
        # 所以原来: scores = [0.3, 0.8, 0.5, 0.9]
        # 屏蔽后:  scores = [0.3, 0.8, 极小值, 极小值]
        #
        # torch.finfo(scores.dtype).min
        # 表示当前浮点数类型能表示的一个非常非常小的数
        #
        # 后面经过 softmax：
        # 极小值对应的概率 ≈ 0
        #
        # 最终效果：
        # 模型只会在允许的 token 中寻找匹配，
        # CLS / SEP / PAD / 非目标文本位置不会被匹配到
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min
        )

        # 对 scores 最后一维做 softmax，
        # 即：对于 A 中的每一个 token，
        # 计算它与 B 中所有 token 的匹配权重
        # scores.shape = [16, 128, 128]
        #                 ↑    ↑    ↑
        #               batch  A    B
        # 例如某个 A token：
        # scores = [2.0, 1.0, -∞, -∞]
        #            B1   B2   B3   B4
        # softmax 后：
        # [0.73, 0.27, 0, 0]
        # 表示：
        # 当前 A token 主要匹配 B1，其次匹配 B2
        # 再乘 mask：
        # mask = [1, 1, 0, 0]
        # [0.73, 0.27, 0, 0]
        #        *
        # [1,    1,    0, 0]
        # =
        # [0.73, 0.27, 0, 0]
        # 作用：
        # 再次确保 CLS / SEP / PAD 等无效位置的匹配权重为 0
        weights = torch.softmax(scores, dim=-1) * mask

        # 对匹配权重再次归一化
        #
        # weights.shape = [batch, A_len, B_len]
        # 对于 A 中的每一个 token，
        # 都有一行它对 B 所有 token 的匹配权重。
        # 例如：
        # weights =
        # [0.6, 0.2, 0, 0]
        # 当前这一行权重总和：
        # 0.6 + 0.2 + 0 + 0 = 0.8
        #
        # weights.sum(dim=-1, keepdim=True)
        # 得到：
        # [0.8]
        # 然后每个权重都除以 0.8：
        # [0.6, 0.2, 0, 0] / 0.8
        # =
        # [0.75, 0.25, 0, 0]
        # 此时：
        # 0.75 + 0.25 = 1
        #
        # 作用：
        # 保证每个 A token 对 B 的有效匹配权重加起来约等于 1。
        # clamp_min(1e-6)：
        # 防止分母等于 0。
        # 如果某一行全是 0：
        # weights.sum(...) = 0
        # 直接除以 0 会出错，
        # 所以至少把分母限制为 1e-6
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 根据匹配权重，对目标文本的 token 特征做加权求和
        # weights: [batch, A_len, B_len]
        # 表示: A 中每个 token 对 B 中所有 token 的匹配权重
        # tokens: [batch, B_len, 128]
        # 表示: B 中每个 token 的 128 维特征
        # torch.bmm：
        # [batch, A_len, B_len]
        #          ×
        # [batch, B_len, 128]
        #          ↓
        # [batch, A_len, 128]
        #
        # 例如：
        # A中的 train 对 B 的匹配权重：
        # [0.7, 0.2, 0.1]
        #
        # B中的token：
        # railway = [1,2]
        # station = [3,4]
        # car     = [5,6]
        #
        # 最终：
        # 0.7*railway + 0.2*station + 0.1*car
        #
        # 得到一个新的向量，
        # 作为 train 在 B 中找到的“匹配表示 aligned”。
        return torch.bmm(
            weights,
            tokens
        )

    # 3. Pooling函数
    def _pool(self, features, mask):
        '''
        函数作用：
           将一整段文本的多个 token 特征，
           汇总成固定长度向量。
        输入：
           features:
               [batch, seq_len, 128]
           mask:
               [batch, seq_len]
        输出：
           [batch, 256]

        因为：
           mean pooling = 128维
           max pooling  = 128维
           拼接以后：
           128 + 128 = 256
        '''

        # mask.shape = [batch, seq_len] = [16, 128]
        # features.shape = [batch, seq_len, hidden] = [16, 128, 128]
        # unsqueeze(-1):
        # [16, 128] => [16, 128, 1]
        # 这样后面 features * float_mask 时，
        # 每个 token 的一个 mask 值可以作用到这个 token 的全部 128 个特征：
        # token有效：mask=1
        # [0.2, 0.5, -0.3] * 1 = [0.2, 0.5, -0.3]
        # token无效：mask=0
        # [0.2, 0.5, -0.3] * 0 = [0, 0, 0]
        # .to(features.dtype)：
        # 把 mask 转成和 features 相同的浮点类型，
        # 方便后面直接相乘
        float_mask = mask.unsqueeze(-1).to(features.dtype)

        # Mean Pooling:对每个样本的有效 token 特征取平均
        #
        # 假设：
        # features.shape = [2, 4, 3]
        #
        # 表示：
        # 2 = batch 中有 2 个样本
        # 4 = 每个样本有 4 个 token
        # 3 = 每个 token 有 3 个特征
        #
        # features =
        # [
        #     [   # 第1个样本
        #         [1, 2, 3],      # token1
        #         [4, 5, 6],      # token2
        #         [7, 8, 9],      # token3
        #         [10,11,12]      # PAD
        #     ],
        #
        #     [   # 第2个样本
        #         [2, 4, 6],      # token1
        #         [8,10,12],      # token2
        #         [14,16,18],     # PAD
        #         [20,22,24]      # PAD
        #     ]
        # ]
        #
        # mask = [batch,seq_len] = [2, 4]
        # [
        #     [1,1,1,0],          # 第1个样本有3个有效token
        #     [1,1,0,0]           # 第2个样本有2个有效token
        # ]
        #
        # float_mask = mask.unsqueeze(-1) 后： [2, 4, 1]
        #
        # [
        #     [[1],[1],[1],[0]],
        #     [[1],[1],[0],[0]]
        # ]
        #
        # 先:
        # features * float_mask
        # PAD位置会被乘0:
        # 然后：
        # (features * float_mask).sum(dim=1)
        # dim=1 表示把每个样本的 token 维加起来
        #
        # 第1个样本：
        # [1,2,3] + [4,5,6] + [7,8,9]
        # = [12,15,18]
        #
        # 第2个样本：
        # [2,4,6] + [8,10,12]
        # = [10,14,18]
        #
        # 再：
        # float_mask.sum(dim=1)
        # 得到有效 token 数量：
        # 第1个样本：[3]
        # 第2个样本：[2]
        #
        # 最后做除法：
        #
        # 第1个样本：
        # [12,15,18] / 3
        # = [4,5,6]
        #
        # 第2个样本：
        # [10,14,18] / 2
        # = [5,7,9]
        #
        # 最终：
        # mean_feature =
        # [
        #     [4,5,6],
        #     [5,7,9]
        # ]
        # shape变化：
        # [2,4,3] -> [2,3]
        # 对应真实代码：
        # [16,128,128] -> [16,128]
        # clamp_min(1)：
        # 防止某个样本有效token数量为0，导致除以0
        mean_feature = (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)

        # Max Pooling
        # mask.shape = [2, 4]
        # mask.unsqueeze(-1)
        # [2, 4] -> [2, 4, 1]
        # 具体：
        # [
        #     [[1], [1], [1], [0]],   # 第1个样本 -> [4,1]
        #     [[1], [1], [0], [0]]    # 第2个样本 -> [4,1]
        # ]
        # features.shape          = [2, 4, 3]
        # mask.unsqueeze(-1).shape = [2, 4, 1]
        # [2,4,1] 会广播成 [2,4,3]，
        # 让每个 token 的 mask 值作用到它的3个特征。
        # masked_fill 后：
        # shape仍然是 [2,4,3]
        # [
        #     [   # 第1个样本 -> [4,3]
        #         [1, 2, 3],
        #         [4, 5, 1],
        #         [2, 8, 6],
        #         [-∞,-∞,-∞]     # PAD被屏蔽
        #     ],
        #
        #     [   # 第2个样本 -> [4,3]
        #         [3, 1, 5],
        #         [7, 4, 2],
        #         [-∞,-∞,-∞],    # PAD
        #         [-∞,-∞,-∞]     # PAD
        #     ]
        # ]
        # .max(dim=1)
        # dim=1 是 token 维：
        # [2,4,3]
        #    ↑
        #  对4个token取最大值
        #    ↓
        #  [2,3]
        # 第1个样本：
        #
        # 特征1：max(1,4,2,-∞) = 4
        # 特征2：max(2,5,8,-∞) = 8
        # 特征3：max(3,1,6,-∞) = 6
        #
        # 得到：
        # [4,8,6] -> shape=[3]
        #
        #
        # 第2个样本：
        #
        # 特征1：max(3,7,-∞,-∞) = 7
        # 特征2：max(1,4,-∞,-∞) = 4
        # 特征3：max(5,2,-∞,-∞) = 5
        #
        # 得到：
        # [7,4,5] -> shape=[3]
        #
        #
        # 最终：
        #
        # max_feature =
        # [
        #     [4,8,6],
        #     [7,4,5]
        # ]
        # 多个 token 的同一个特征，谁最大就留下谁。
        # max_feature.shape = [2,3]
        max_feature = features.masked_fill(
            ~mask.unsqueeze(-1),
            torch.finfo(features.dtype).min
        ).max(dim=1).values

        # 极端情况下，如果某个文本没有有效 token，
        # 就把 max_feature 设置为全0，
        # 防止留下极小值
        max_feature = torch.where(
            mask.any(
                dim=1,
                keepdim=True
            ),
            max_feature,
            torch.zeros_like(max_feature)
        )

        # mean：
        # [batch,128]
        # max：
        # [batch,128]
        # 拼接：
        # [batch,256]
        return torch.cat(
            [
                mean_feature,
                max_feature
            ],
            dim=-1
        )

    # 4. 前向传播
    def forward(self, input_ids, attention_mask):
        deberta_output = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = deberta_output.last_hidden_state
        # 原来的CLS分类
        cls_logits = self.output(
            self.dropout(
                last_hidden_state[:, 0, :]
            )
        )

        # Token对齐比较分支
        # 找出SEP的位置 (2个)
        # 输入大致:
        # [CLS] A A A [SEP] B B [SEP] PAD PAD
        # is_sep:
        # False False False False True False False True False...
        is_sep = (input_ids == self.sep_id)

        # cumsum：
        # 累计前面出现了几个 SEP
        # 例如：
        # token:
        # CLS A A SEP B B SEP PAD
        # sep_count:
        #  0  0 0  1  1 1  2   2
        #
        # sep_count == 0
        #   → 第一个 SEP 以前
        #   → A
        # sep_count == 1
        #   → 第一、第二个 SEP 之间
        #   → B
        sep_count = (
            is_sep.long().cumsum(dim=1)
        )

        # 找出真正有效的普通 token
        # 只有同时满足下面3个条件的位置才为 True：
        # 1. attention_mask.bool()
        #    不是 PAD
        # 2. ~is_sep
        #    不是 [SEP]
        # 3. input_ids != self.cls_id
        #    不是 [CLS]
        # 例如：
        # token:
        # [CLS] train station [SEP] station [SEP] [PAD] [PAD]
        # attention_mask.bool():
        # [ T,    T,     T,      T,     T,      T,     F,     F ]
        # ~is_sep:
        # [ T,    T,     T,      F,     T,      F,     T,     T ]
        # input_ids != cls_id:
        # [ F,    T,     T,      T,     T,      T,     T,     T ]
        # 三个条件做 & 后：
        # valid:
        # [ F,    T,     T,      F,     T,      F,     F,     F ]
        valid = (
                attention_mask.bool() & (~is_sep) & (input_ids != self.cls_id)
        )

        # A/B文本 mask
        # valid
        # 且
        # 还没经过第一个 SEP
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)

        # Token 特征降维
        # last_hidden_state:
        # [batch,128,768]
        # ↓ Linear
        # tokens:
        # [batch,128,128]
        tokens = self.align_projection(last_hidden_state)

        # L2归一化
        # 归一化以后:
        # token之间点积
        # 基本就等价于 cosine similarity。
        # shape 不变：
        # [batch,128,128]
        normalized = F.normalize(tokens, dim=-1)

        # 7.4 计算所有 token 两两相似度
        #
        # normalized：[batch, seq_len, 128]
        # transpose：[batch, 128, seq_len]
        #
        # bmm: [B,L,128] @ [B,128,L] = [B,L,L]
        #
        # 如果 seq_len=128：
        # scores = [batch,128,128]
        # scores[i,j,k]:
        #
        # 第 i 个样本中
        # 第 j 个 token
        # 和第 k 个 token
        # 的相似程度。

        # 0.1 相当于 temperature。
        #
        # 数值变大以后，
        # softmax 会更集中于最相似的 token。
        # [0.7, 0.5, 0.2] / 0.1 => [7,5,2]
        scores = torch.bmm(
            normalized,
            normalized.transpose(1, 2)
        ) / 0.1

        # A中的token去B中寻找匹配
        #
        # tokens  = 所有 token 的特征
        # scores  = token 和 token 之间的匹配分数
        # mask_b  = 哪些位置属于 B，只有 B 的 token 允许被匹配
        #
        # 例如输入：
        # [CLS] train station [SEP] railway station [SEP] [PAD]
        # A = train station
        # B = railway station
        # mask_b =
        # [ F, F, F, F, T, T, F, F ]
        #                  ↑  ↑
        #               railway station
        # 假设 A 中 train 对所有 token 的匹配分数：
        #              CLS train station SEP railway station SEP PAD
        # scores       0.1  0.3    0.2   0.1   0.8     0.2   0.1 0.0
        # _align() 根据 mask_b，只允许和 B 中的 token 匹配：
        #              CLS train station SEP railway station SEP PAD
        # 屏蔽后       -∞   -∞    -∞    -∞    0.8     0.2   -∞  -∞
        # softmax 后，例如：
        # railway  -> 0.65
        # station  -> 0.35
        # 然后通过 torch.bmm(weights, tokens)：
        # aligned_b(train) = 0.65 * railway的特征 + 0.35 * station的特征
        # 最终得到 train 在 B 中的一个 128 维匹配特征
        aligned_b = self._align(tokens, scores, mask_b)

        # B中的token去A中寻找匹配
        aligned_a = self._align(tokens, scores, mask_a)

        # 比较 A 中每个 token 和它在 B 中找到的匹配特征
        #
        # tokens.shape    = [batch, seq_len, 128]
        # aligned_b.shape = [batch, seq_len, 128]
        # 例如只看 A 中的 train：
        # tokens(train) = train 自己的 128 维特征
        # aligned_b(train) = train 根据匹配权重，从 B 中得到的 128 维匹配特征
        # 然后构造 4 组特征：
        # 1. tokens             train 自己是什么                        -> [128]
        # 2. aligned_b          train 在 B 中匹配到了什么                -> [128]
        # 3. tokens - aligned_b train 和 B 中匹配结果有什么差异           -> [128]
        # 4. tokens * aligned_b train 和 B 中匹配结果有哪些共同/相似特征    -> [128]
        #
        # torch.cat(..., dim=-1)
        # 表示沿最后一个“特征维度”进行拼接：
        # [128] + [128] + [128] + [128]
        #                  ↓
        #                [512]
        # 所以整体 shape：
        # [batch, seq_len, 128] × 4
        #            ↓ cat(dim=-1)
        # [batch, seq_len, 512]
        # 然后进入 compare_layer：
        # Linear(512,128) + GELU
        # [batch, seq_len, 512]
        #            ↓
        # [batch, seq_len, 128]
        # 最终 features_a：
        # A 中每个 token 与 B 匹配比较之后得到的 128 维特征
        features_a = self.compare_layer(
            torch.cat(
                [
                    tokens,
                    aligned_b,
                    tokens - aligned_b,
                    tokens * aligned_b
                ],
                dim=-1
            )
        )
        features_b = self.compare_layer(
            torch.cat(
                [
                    tokens,
                    aligned_a,
                    tokens - aligned_a,
                    tokens * aligned_a
                ],
                dim=-1
            )
        )

        # features_a.shape = [batch, seq_len, 128]
        # features_b.shape = [batch, seq_len, 128]
        #
        # self._pool(features_a, mask_a)
        # 会对 A 的有效 token 做：
        #   mean pooling -> [batch, 128]
        #   max pooling  -> [batch, 128]
        #
        # 然后拼接：
        # [128] + [128]
        #      ↓
        # [256]
        #
        # 所以：
        # self._pool(features_a, mask_a) -> [batch, 256]
        # self._pool(features_b, mask_b) -> [batch, 256]
        #
        # 最后再把 A 和 B 拼起来：
        # A的pool结果 [256] + B的pool结果 [256]
        #                  ↓ torch.cat(dim=-1)
        #                [512]
        # pair_feature.shape = [batch, 512]
        #
        # pair_feature 表示：
        # A 和 B 两句话经过 token 对齐、比较、pooling 后，
        # 得到的整对句子的最终比较特征。
        pair_feature = torch.cat(
            [
                self._pool(
                    features_a,
                    mask_a
                ),
                self._pool(
                    features_b,
                    mask_b
                )
            ],
            dim=-1
        )

        # 7.8 Token 对齐分支输出四分类分数
        # pair_feature：
        # [batch,512]
        # ↓ LayerNorm
        # ↓ Dropout
        # ↓ Linear
        # match_logits：[batch,4]
        # 例如：
        # [ F分数,B分数,E分数,N分数]
        match_logits = self.match_output(pair_feature)

        # 8. 两个分支融合
        #
        # 原始 CLS: cls_logits [batch,4]
        # Token 对齐分支: match_logits [batch,4]
        #
        # 两者直接相加：
        # final_logits = cls_logits + match_logits
        # 可以理解为：
        # CLS: "整体语义上我觉得是什么关系"
        # Token Alignment: "逐词比较以后，我再补充一些判断"
        # 最终：
        #   F / B / E / N
        return cls_logits + match_logits


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
    # 对照：相同 seed 的均匀平均版
    old_preds = pd.read_csv(
        project_dir
        / "model"
        / f"deberta_nli_base_uniform_alignment_seed_{SEED}"
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


# todo 12. 分析 F/B 与 E 的混淆
def analyze_equivalence_errors():
    project_dir = (PROJECT_ROOT)

    # 1. 读取标准答案
    gold = pd.read_csv(
        project_dir / "data/dev/dico_nli_dev_track1_reference.csv"
    )[["instance_id", "label"]]

    gold = gold.rename(columns={"label": "gold_label"})

    # 2. 读取当前模型已保存的预测
    predictions = pd.read_csv(
        RUN_DIR / "dev_predictions_pair_aware.csv"
    ).rename(columns={"label": "pred_label"})

    # 3. 按 instance_id，把文本、答案、预测对应起来
    data = dev_data[
        ["instance_id", "pair_id", "text1", "text2"]
    ].merge(
        gold,
        on="instance_id",
        validate="one_to_one"
    ).merge(
        predictions[["instance_id", "pred_label"]],
        on="instance_id",
        validate="one_to_one"
    )

    # 4. 找出 F/B 被预测成 E 的样本
    direction_labels = [
        "FORWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT"
    ]

    direction_to_e = (
            data["gold_label"].isin(direction_labels)
            & (data["pred_label"] == "EQUIVALENCE")
    )

    # 5. 找出 E 被预测成 F/B 的样本
    e_to_direction = (
            (data["gold_label"] == "EQUIVALENCE")
            & data["pred_label"].isin(direction_labels)
    )

    errors = data[direction_to_e | e_to_direction].copy()

    # 6. 保存全部错误行
    output_path = RUN_DIR / "equivalence_boundary_errors.csv"

    errors.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"F/B → E：{direction_to_e.sum()} 条")
    print(f"E → F/B：{e_to_direction.sum()} 条")

    # 同一个 pair 往往有正反两行，查看文本时只展示一行
    unique_pairs = errors.drop_duplicates(subset="pair_id")

    print(f"涉及不同 pair：{len(unique_pairs)} 组")
    print(
        unique_pairs[
            ["text1", "text2", "gold_label", "pred_label"]
        ].to_string(index=False)
    )

    print(f"\n错误文件：{output_path}")


if __name__ == "__main__":
    checkAndprocess_data()
    train_model_real_pair_ce()

    best_epoch, best_score, best_preds = (
        evaluate_pair_aware_model()
    )

    submission_output(best_preds)