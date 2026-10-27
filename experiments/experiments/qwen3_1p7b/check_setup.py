"""先检查 tokenizer、全部输入和下载完整性，不加载 1.7B 权重。"""
import config as cfg
from data_utils import (
    load_data, load_tokenizer, build_train_dataset, build_prompt_template,
    get_datasets, check_examples, check_model_files,
)


def main():
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    template = build_prompt_template(tokenizer)
    dataset = load_data()
    dataset['train'] = build_train_dataset(dataset['train'], cfg.MODE)
    dataset = get_datasets(dataset, template)
    print('训练 / Dev：', len(dataset['train']), '/', len(dataset['dev']))
    check_examples(dataset, tokenizer, cfg.MAX_LENGTH)
    print('第一条完整 Prompt：')
    print(repr(dataset['train'][0]['prompt']))
    check_model_files(cfg.MODEL_PATH)
    print('检查通过，可以运行 train_lora_abc.py。')


if __name__ == '__main__':
    main()
