from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

model_path = 'D:/LLM/model/Qwen3-0.6B-Base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
project_root = Path(__file__).resolve().parents[2]

label_map = {
    'FORWARD_ENTAILMENT' : 'F',
    'BACKWARD_ENTAILMENT' : 'B',
    'EQUIVALENCE' : 'E',
    'NEGATIVE_OTHER' : 'N'
}

def build_prompt(text1, text2):
    prompt = f"""
Classify the semantic relation between Text 1 and Text 2.

F: Text 1 entails Text 2, but not vice versa.
B: Text 2 entails Text 1, but not vice versa.
E: Text 1 and Text 2 entail each other.
N: Neither text entails the other.

Output exactly one label: F, B, E, or N.

Text 1: {text1}
Text 2: {text2}
Answer:
"""

    return prompt.strip() + '\n'

def load_data():
    data_files = {
        'train': str(project_root / 'data/train/dico_nli_train_track1_participant_labeled.csv'),
        'dev': str(project_root / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'),
    }

    dataset = load_dataset('csv', data_files=data_files)

    return dataset


def build_sft_example(example, prompt_template=None):
    if prompt_template is None:
        prompt_template = build_prompt('{text1}', '{text2}')

    return {
        'prompt': prompt_template.format(
            text1=example['text1'],
            text2=example['text2']
        ),
        'completion': label_map[example['label']]
    }

def get_datasets(dataset):
    # 将完整 Prompt 模板显式传给 map。
    # 模板改变，缓存标识也会改变。
    prompt_template = build_prompt('{text1}', '{text2}')

    return dataset.map(
        build_sft_example,
        fn_kwargs={'prompt_template': prompt_template},
        load_from_cache_file=True
    )

def encode_prompt(prompt):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True
    )

    return inputs

if __name__ == '__main__':
    dataset = load_data()
    sft_dataset = get_datasets(dataset)

