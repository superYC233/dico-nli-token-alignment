from datasets import load_dataset
from transformers import AutoTokenizer

model_path = 'D:/LLM/model/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    messages = [
        {'role': 'user', 'content': prompt.strip()}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
def load_data():
    data_files = {
        'train': '../../data/train/dico_nli_train_track1_participant_labeled.csv',
        'dev': '../../data/dev/dico_nli_dev_track1_participant_labeled.csv',
    }

    dataset = load_dataset('csv', data_files=data_files)

    return dataset


def build_sft_example(example):
    text1 = example['text1']
    text2 = example['text2']
    label = label_map[example['label']]

    prompt = build_prompt(text1, text2)

    return {
        'prompt': prompt,
        'completion': label
    }

def get_datasets(dataset):
    return dataset.map(
        build_sft_example,
        load_from_cache_file=False
    )

def encode_prompt(prompt):
    tokenizer.padding_side = 'left'

    return tokenizer(
        prompt,
        return_tensors='pt',
        padding=True
    )

    return inputs

if __name__ == '__main__':
    dataset = load_data()
    sft_dataset = get_datasets(dataset)

