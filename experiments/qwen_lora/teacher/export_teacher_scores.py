"""导出 DeBERTa 教师分数；只推理，不训练。"""

# 分类后仍从 qwen_lora 读取公共模块和原有 outputs。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

script_dir = QWEN_DIR
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from data_variants import build_train_dataset
from project_paths import get_model_cache_dir

LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {
    'FORWARD_ENTAILMENT': 'F',
    'BACKWARD_ENTAILMENT': 'B',
    'EQUIVALENCE': 'E',
    'NEGATIVE_OTHER': 'N',
}
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
CHECKPOINT = (
    project_root / 'model'
    / 'deberta_nli_base_token_alignment_official_full_control_seed_42'
    / 'epoch_7.pth'
)
OUTPUT_DIR = script_dir / 'outputs/teacher_scores/run01'
BATCH_SIZE = 16
MAX_LENGTH = 128


class TeacherModel(nn.Module):
    """与 train.py 的 DeBERTa + Token Alignment 保持相同结构和参数名。"""

    def __init__(self, backbone, tokenizer):
        super().__init__()
        self.deberta_model = backbone
        hidden_size = backbone.config.hidden_size
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_size, 4)
        self.align_projection = nn.Linear(hidden_size, 128)
        self.compare_layer = nn.Sequential(
            nn.Linear(128 * 4, 128), nn.ReLU()
        )
        self.match_output = nn.Sequential(
            nn.LayerNorm(128 * 4), nn.Dropout(0.2), nn.Linear(128 * 4, 4)
        )

    def _align(self, tokens, scores, target_mask):
        mask = target_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(weights, tokens)

    def _pool(self, features, mask):
        float_mask = mask.unsqueeze(-1).to(features.dtype)
        mean_feature = (
            (features * float_mask).sum(dim=1)
            / float_mask.sum(dim=1).clamp_min(1)
        )
        max_feature = features.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(features.dtype).min
        ).max(dim=1).values
        max_feature = torch.where(
            mask.any(dim=1, keepdim=True), max_feature,
            torch.zeros_like(max_feature)
        )
        return torch.cat([mean_feature, max_feature], dim=-1)

    def forward(self, input_ids, attention_mask):
        hidden = self.deberta_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        cls_logits = self.output(self.dropout(hidden[:, 0, :]))

        # [CLS] Text1 [SEP] Text2 [SEP]：分别找出两个短语的有效位置。
        is_sep = input_ids == self.sep_id
        sep_count = is_sep.long().cumsum(dim=1)
        valid = attention_mask.bool() & ~is_sep & (input_ids != self.cls_id)
        mask_a = valid & (sep_count == 0)
        mask_b = valid & (sep_count == 1)

        tokens = self.align_projection(hidden)
        normalized = F.normalize(tokens, dim=-1)
        scores = torch.bmm(normalized, normalized.transpose(1, 2)) / 0.1
        aligned_b = self._align(tokens, scores, mask_b)
        aligned_a = self._align(tokens, scores, mask_a)

        features_a = self.compare_layer(torch.cat(
            [tokens, aligned_b, tokens - aligned_b, tokens * aligned_b], dim=-1
        ))
        features_b = self.compare_layer(torch.cat(
            [tokens, aligned_a, tokens - aligned_a, tokens * aligned_a], dim=-1
        ))
        pair_feature = torch.cat([
            self._pool(features_a, mask_a), self._pool(features_b, mask_b)
        ], dim=-1)
        return cls_logits + self.match_output(pair_feature)


def load_teacher():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'找不到教师权重：{CHECKPOINT}')

    cache_dir = get_model_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    backbone = AutoModel.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir,
        dtype=torch.float32, use_safetensors=True
    )
    model = TeacherModel(backbone, tokenizer)

    # 严格检查全部参数；不能用 strict=False 跳过不匹配的权重。
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print('教师权重：', CHECKPOINT)
    print('推理设备：', device, '；精度：float32')
    return model, tokenizer, device


def prepare_data():
    train = pd.read_csv(
        project_root / 'data/train/dico_nli_train_track1_participant_labeled.csv',
        keep_default_na=False, dtype=str
    )
    dev = pd.read_csv(
        project_root / 'data/dev/dico_nli_dev_track1_participant_labeled.csv',
        keep_default_na=False, dtype=str
    )

    # 复用你的 C 构造函数；不导入 data_utils，避免顺带加载 Qwen tokenizer。
    original_size = len(train)
    dataset = Dataset.from_pandas(train, preserve_index=False)
    train_c = build_train_dataset(dataset, mode='C').to_pandas()
    train_c['source'] = 'train_original'
    train_c.loc[original_size:, 'source'] = 'train_reverse_n'
    dev['source'] = 'dev'

    for frame in [train_c, dev]:
        if not frame['instance_id'].is_unique:
            raise ValueError('instance_id 存在重复，请检查数据。')
        if not frame['label'].isin(LABEL_MAP).all():
            raise ValueError('发现未知标签，请检查 LABEL_MAP。')

    print('原始 Train：', original_size)
    print('C 训练集：', len(train_c))
    print('新增反向 N：', len(train_c) - original_size)
    print('Dev：', len(dev))
    return train_c, dev


def predict_scores(model, tokenizer, device, data, description):
    all_logits = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(data), BATCH_SIZE), desc=description):
            batch = data.iloc[start:start + BATCH_SIZE]
            inputs = tokenizer(
                batch['text1'].tolist(), batch['text2'].tolist(),
                truncation=True, padding='max_length',
                max_length=MAX_LENGTH, return_tensors='pt'
            )
            logits = model(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device)
            )
            all_logits.append(logits.float().cpu())

    logits = torch.cat(all_logits, dim=0)
    if logits.shape != (len(data), 4) or not torch.isfinite(logits).all():
        raise ValueError('教师输出形状不正确，或包含非有限数值。')

    result = data.copy().reset_index(drop=True)
    result['gold_short'] = result['label'].map(LABEL_MAP)
    result['teacher_prediction'] = [LABELS[i] for i in logits.argmax(dim=1)]
    for index, label in enumerate(LABELS):
        result[f'logit_{label}'] = logits[:, index].numpy()

    # 四类内归一化。T 越大，概率越平缓；原始 logits 仍完整保留。
    for temperature in [1, 2, 4]:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        for index, label in enumerate(LABELS):
            result[f'prob_{label}_T{temperature}'] = probabilities[:, index].numpy()
    return result


def check_dev_reproduction(dev_scores):
    """仅检查模型是否加载正确；导出的教师分数始终不做结构解码。"""
    predictions = dev_scores['teacher_prediction'].copy()
    for _, group in dev_scores.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        scores = [
            a['logit_F'] + b['logit_B'],
            a['logit_B'] + b['logit_F'],
            a['logit_E'] + b['logit_E']
        ]
        combinations = [('F', 'B'), ('B', 'F'), ('E', 'E')]
        label_a, label_b = combinations[scores.index(max(scores))]
        predictions.loc[group.index[0]] = label_a
        predictions.loc[group.index[1]] = label_b

    score = f1_score(dev_scores['gold_short'], predictions, average='weighted')
    print(f'教师 Dev pair-aware W-F1（仅校验）：{score:.9f}')

    reference = pd.read_csv(
        project_root / 'artifacts/final/dev_predictions_pair_aware.csv',
        keep_default_na=False
    ).set_index('instance_id')
    expected = reference['label'].map(LABEL_MAP).reindex(dev_scores['instance_id'])
    if expected.isna().any():
        raise ValueError('历史预测文件的样本或标签不完整。')
    differences = int((predictions.to_numpy() != expected.to_numpy()).sum())
    print('与历史预测不同的行数：', differences)
    if differences:
        raise ValueError('未完全复现历史教师预测，请先核对环境和权重，暂不进行蒸馏。')


def summarize_scores(train_scores, dev_scores):
    rows = []
    groups = [
        ('train_original', train_scores[train_scores['source'] == 'train_original']),
        ('train_reverse_n', train_scores[train_scores['source'] == 'train_reverse_n']),
        ('dev', dev_scores),
        ('dev_N', dev_scores[dev_scores['gold_short'] == 'N'])
    ]
    for name, group in groups:
        if len(group) == 0:
            continue
        accuracy = (group['gold_short'] == group['teacher_prediction']).mean()
        for temperature in [1, 2, 4]:
            columns = [f'prob_{label}_T{temperature}' for label in LABELS]
            probabilities = torch.tensor(group[columns].to_numpy())
            max_probability = probabilities.max(dim=1).values
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=1)
            rows.append({
                'split': name, 'rows': len(group), 'accuracy': accuracy,
                'temperature': temperature,
                'mean_max_probability': max_probability.mean().item(),
                'mean_entropy': entropy.mean().item()
            })
    return pd.DataFrame(rows)


def main():
    train_c, dev = prepare_data()
    model, tokenizer, device = load_teacher()

    # 先用 Dev 检查权重和模型结构；Dev 预测只做诊断，不作训练目标。
    dev_scores = predict_scores(model, tokenizer, device, dev, '教师预测 Dev')
    check_dev_reproduction(dev_scores)
    train_scores = predict_scores(model, tokenizer, device, train_c, '教师预测 Train C')
    summary = summarize_scores(train_scores, dev_scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_scores.to_csv(OUTPUT_DIR / 'train_C_teacher_scores.csv', index=False, encoding='utf-8-sig')
    dev_scores.to_csv(OUTPUT_DIR / 'dev_teacher_scores.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT_DIR / 'teacher_summary.csv', index=False, encoding='utf-8-sig')

    print('\n===== 教师分数统计 =====')
    print(summary.to_string(index=False))
    print('\n四类顺序：', LABELS)
    print('训练目标文件：', OUTPUT_DIR / 'train_C_teacher_scores.csv')
    print('Dev 文件仅用于诊断，不用于训练。')
    print('统计表：', OUTPUT_DIR / 'teacher_summary.csv')


if __name__ == '__main__':
    main()
