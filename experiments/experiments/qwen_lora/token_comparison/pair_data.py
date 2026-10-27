"""只给模型 Prompt；分类标签不拼进输入，两个文本段单独做掩码。"""
import torch
from torch.utils.data import Dataset

LABELS = ['F', 'B', 'E', 'N']
LABEL_TO_ID = {
    'FORWARD_ENTAILMENT': 0,
    'BACKWARD_ENTAILMENT': 1,
    'EQUIVALENCE': 2,
    'NEGATIVE_OTHER': 3,
}


def render_prompt(template, text1, text2):
    # 在填入正文之前确定位置；正文自身即使包含 Text 2: 也不会匹配错。
    if template.count('{text1}') != 1 or template.count('{text2}') != 1:
        raise ValueError('模板需要各包含一次 {text1} 和 {text2}。')
    prefix, rest = template.split('{text1}')
    middle, suffix = rest.split('{text2}')
    prompt = prefix + text1 + middle + text2 + suffix
    start1 = len(prefix)
    start2 = start1 + len(text1) + len(middle)
    return prompt, (start1, start1 + len(text1)), (start2, start2 + len(text2))


class PhraseDataset(Dataset):
    def __init__(self, frame, tokenizer, template, max_length=512):
        if not tokenizer.is_fast:
            raise ValueError('需要 fast tokenizer 的 offset_mapping 来确定文本位置。')
        self.frame = frame.reset_index(drop=True).copy()
        self.examples = []
        for row in self.frame.to_dict('records'):
            prompt, span1, span2 = render_prompt(template, row['text1'], row['text2'])
            encoded = tokenizer(
                prompt, add_special_tokens=False, return_offsets_mapping=True,
                truncation=False
            )
            if len(encoded['input_ids']) > max_length:
                raise ValueError(
                    f"{row['instance_id']} 超过 max_length={max_length}；"
                    '本实验不静默截断短语，请增大 max_length。'
                )
            offsets = encoded['offset_mapping']
            mask_a = [end > span1[0] and start < span1[1] and end > start
                      for start, end in offsets]
            mask_b = [end > span2[0] and start < span2[1] and end > start
                      for start, end in offsets]
            if not any(mask_a) or not any(mask_b):
                raise ValueError(f"{row['instance_id']} 的短语为空或无法定位。")
            if any(a and b for a, b in zip(mask_a, mask_b)):
                raise ValueError('一个 token 同时跨越两个短语，请检查模板分隔符。')
            self.examples.append({
                'input_ids': encoded['input_ids'],
                'mask_a': mask_a,
                'mask_b': mask_b,
                'labels': LABEL_TO_ID[row['label']],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PhraseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # 分类时按有效文本段汇总，所以训练和预测都可以采用右补齐。
        length = max(len(row['input_ids']) for row in examples)
        batch_size = len(examples)
        ids = torch.full((batch_size, length), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((batch_size, length), dtype=torch.long)
        mask_a = torch.zeros((batch_size, length), dtype=torch.bool)
        mask_b = torch.zeros((batch_size, length), dtype=torch.bool)
        labels = []
        for index, row in enumerate(examples):
            size = len(row['input_ids'])
            ids[index, :size] = torch.tensor(row['input_ids'])
            attention[index, :size] = 1
            mask_a[index, :size] = torch.tensor(row['mask_a'])
            mask_b[index, :size] = torch.tensor(row['mask_b'])
            labels.append(row['labels'])
        return {
            'input_ids': ids, 'attention_mask': attention,
            'mask_a': mask_a, 'mask_b': mask_b,
            'labels': torch.tensor(labels, dtype=torch.long),
        }
