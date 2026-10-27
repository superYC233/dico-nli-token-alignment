"""先构造 C，再加入与 DeBERTa 相同的200条增强；不加载模型。"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

from experiments.qwen_lora.data_variants import build_train_dataset
from experiments.deberta_original_aug.scope_augmentation import make_scope_rows, validate_holdout


TRAIN_PATH = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
LABEL_MAP = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
             'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}

# 与原 C+r16 的旧短 Prompt 一致，使用裸文本，不套 chat template。
PROMPT_TEMPLATE = '''Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
'''


def prepare_data(output_dir, add_scope=True):
    output_dir = Path(output_dir)
    if (output_dir / 'run_config.json').exists():
        raise FileExistsError('此 RUN_NAME 已有训练记录，请换新名字。')
    original = pd.read_csv(TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    original_dataset = Dataset.from_pandas(original, preserve_index=False)
    c_dataset = build_train_dataset(original_dataset, mode='C')
    reverse_n = c_dataset.select(range(len(original_dataset), len(c_dataset)))

    # 必须用原始 Train 构造这200条，然后加到 C 后面。
    # 不对增强数据再调用 C，否则40条新增N也会被反转，变成另一个实验。
    if add_scope:
        extra, audit = make_scope_rows(original)
        validate_holdout(extra, dev)
    else:
        extra = original.iloc[:0].copy()
        audit = pd.DataFrame(columns=['instance_id', 'pair_id', 'word_source_ids', 'annotation_status'])
    extra_dataset = Dataset.from_pandas(extra, preserve_index=False, features=original_dataset.features)
    train = concatenate_datasets([c_dataset, extra_dataset]) if len(extra) else c_dataset
    if len(set(train['instance_id'])) != len(train):
        raise ValueError('构造后的 instance_id 重复。')

    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame = train.to_pandas()
    train_frame.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    reverse_n.to_pandas().to_csv(output_dir / 'c_reverse_n.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(output_dir / 'extra_train_rows.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(output_dir / 'augmentation_audit.csv', index=False, encoding='utf-8-sig')
    info = {
        'original_rows': len(original), 'c_reverse_n_rows': len(reverse_n),
        'c_rows': len(c_dataset), 'scope_rows': len(extra), 'train_rows': len(train),
        'dev_rows': len(dev), 'add_scope': add_scope,
        'class_counts': train_frame.label.value_counts().to_dict(),
        'scope_class_counts': extra.label.value_counts().to_dict(),
        'effective_train_sha256': hashlib.sha256(train_frame.to_csv(index=False).encode('utf-8')).hexdigest(),
        'scope_sha256': hashlib.sha256(extra.to_csv(index=False).encode('utf-8')).hexdigest(),
        'train_source_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        'dev_source_sha256': hashlib.sha256(DEV_PATH.read_bytes()).hexdigest(),
        'annotation_note': 'Train vocabulary and synthetic templates; no independent label adjudication. Dev informed development.',
    }
    (output_dir / 'data_config.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('原 Train / C反向N / 范围增强 / 总计：',
          len(original), '/', len(reverse_n), '/', len(extra), '/', len(train))
    print('训练类别数量：', info['class_counts'])
    return train, Dataset.from_pandas(dev, preserve_index=False), dev, info


def build_sft_example(row, prompt_template):
    return {'prompt': prompt_template.format(text1=row['text1'], text2=row['text2']),
            'completion': LABEL_MAP[row['label']]}


def get_sft_datasets(train, dev, prompt_template=PROMPT_TEMPLATE):
    # 模板作为显式参数参与缓存指纹，避免换Prompt后仍使用旧缓存。
    return DatasetDict(train=train, dev=dev).map(
        build_sft_example, fn_kwargs={'prompt_template': prompt_template},
        load_from_cache_file=True,
    )


def check_supervision(processed, raw_sft, tokenizer):
    if len(processed) != len(raw_sft):
        raise ValueError('SFT预处理丢了样本，需要检查截断或completion mask。')
    for row, original in zip(processed, raw_sft):
        targets = [token for token in row['labels'] if token != -100]
        expected = tokenizer.encode(original['completion'] + tokenizer.eos_token,
                                    add_special_tokens=False)
        if targets != expected:
            raise ValueError(f'答案/EOS监督位置不正确：{original["instance_id"]}，实际{targets}，应为{expected}')
        first_answer = next(index for index, token in enumerate(row['labels']) if token != -100)
        expected_prompt = tokenizer.encode(original['prompt'], add_special_tokens=True)
        if row['input_ids'][:first_answer] != expected_prompt:
            raise ValueError('实际训练Prompt与保存模板不同：' + original['instance_id'])
    print('全部训练样本的旧短Prompt、答案和EOS监督位置检查通过。')
