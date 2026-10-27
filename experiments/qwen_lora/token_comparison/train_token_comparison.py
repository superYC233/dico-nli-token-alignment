import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import setup_experiment_paths
QWEN_DIR = setup_experiment_paths(__name__ == '__main__')

from transformers import TrainingArguments, set_seed

from data_utils import load_data, tokenizer, build_prompt
from data_variants import build_train_dataset
from pair_data import LABELS, PhraseDataset, PhraseCollator
from pair_model import build_model
from comparison_trainer import ComparisonTrainer


# 新版本：词级对齐 + 整体信息。旧版本仍可选择 'alignment' 或 'simple'。
HEAD_TYPE = 'alignment'
SEED = 42
RUN_NAME = 'run01'
MODEL_PATH = 'D:/LLM/model/Qwen3-0.6B-Base'
OUTPUT_DIR = Path(
    f'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_token_comparison/{RUN_NAME}/{HEAD_TYPE}'
)
RESULTS_DIR = QWEN_DIR / f'outputs/token_comparison/{RUN_NAME}/{HEAD_TYPE}'


def main():
    set_seed(SEED)
    # 所有版本都使用当前旧短 Prompt；保存模板，预测时不会误用后来改过的模板。
    template = build_prompt('{text1}', '{text2}')
    if 'Check both directions separately:' in template:
        raise ValueError('请保持旧短 Prompt，避免同时改变多个实验条件。')
    config = {
        'base_model_path': MODEL_PATH,
        'head_type': HEAD_TYPE,
        'dimension': 128,
        'head_dropout': 0.1,
        'max_length': 512,
        'prompt_template': template,
        'labels': LABELS,
        'bf16': True,
    }

    dataset = load_data()
    dataset['train'] = build_train_dataset(dataset['train'], mode='C')
    train_frame = dataset['train'].to_pandas()
    dev_frame = dataset['dev'].to_pandas()
    train_dataset = PhraseDataset(train_frame, tokenizer, template, config['max_length'])
    dev_dataset = PhraseDataset(dev_frame, tokenizer, template, config['max_length'])
    tokenizer.padding_side = 'right'

    # 从基础 Qwen 重新初始化 LoRA；不加载当前主线已训练的 Adapter。
    model = build_model(config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print('比较头版本：', HEAD_TYPE)
    print('训练 / Dev 样本数：', len(train_dataset), '/', len(dev_dataset))
    print('可训练参数：', trainable)
    print('模型保存目录：', OUTPUT_DIR)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=16,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=SEED,
        data_seed=SEED,
        eval_strategy='epoch',
        save_strategy='best',
        load_best_model_at_end=True,
        metric_for_best_model='eval_pair_f1',
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        remove_unused_columns=False,
        logging_steps=20,
        report_to='none',
    )
    trainer = ComparisonTrainer(
        model=model, args=args,
        train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=PhraseCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        dev_frame=dev_frame, results_dir=RESULTS_DIR,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {'model': config, 'training': args.to_dict(), 'train_rows': len(train_dataset)}
    (OUTPUT_DIR / 'run_config.json').write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / 'best_model'))
    print('最佳 checkpoint：', trainer.state.best_model_checkpoint)
    print('最佳 pair-aware F1：', trainer.state.best_metric)
    print('最佳完整模型目录：', OUTPUT_DIR / 'best_model')


if __name__ == '__main__':
    main()
