"""离线检查：真实 tokenizer + 随机微型 Qwen，不下载 MNLI、不加载大权重。"""
import copy
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Value, ClassLabel
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, Qwen3Config, set_seed
from trl import SFTConfig, SFTTrainer

from experiments.qwen_mnli_transfer import config as cfg
from experiments.qwen_mnli_transfer.common import (
    load_tokenizer, load_training_model, save_adapter, check_mnli_adapter,
    check_supervision, check_empty_output,
)
from experiments.qwen_mnli_transfer.data_utils import (
    select_mnli_rows, normalize, belongs_to_validation, prepare_dico, check_examples,
)
from experiments.qwen_mnli_transfer.best_f1_callback import BestF1Callback
from experiments.generative_nli import predict_with_scores, pair_aware_decode, score_methods


def test_data_selection(tokenizer):
    # 故意颠倒 MNLI 编号，检查程序读取 ClassLabel 而非硬编码 0=entailment。
    names = ['neutral', 'contradiction', 'entailment']
    rows = []
    for group in range(300):
        for label in range(3):
            rows.append({
                'premise': f'A person visits place {group}.',
                'hypothesis': f'Example {group} has relation {names[label]}.',
                'label': label, 'pairID': f'{group}_{label}',
            })
    rows += [copy.deepcopy(rows[0])]
    rows += [{'premise': 'BLOCKED TEXT', 'hypothesis': 'blocked', 'label': 0, 'pairID': 'blocked'}]
    rows += [{'premise': 'word ' * 1000, 'hypothesis': 'long', 'label': 1, 'pairID': 'long'}]
    raw = Dataset.from_list(rows, features=Features({
        'premise': Value('string'), 'hypothesis': Value('string'),
        'label': ClassLabel(names=names), 'pairID': Value('string'),
    }))
    first, summary = select_mnli_rows(raw, tokenizer, {'blocked text'}, 30, 9, 42, 512)
    second, _ = select_mnli_rows(raw, tokenizer, {'blocked text'}, 30, 9, 42, 512)
    assert first['train'].to_dict() == second['train'].to_dict()
    assert first['validation'].to_dict() == second['validation'].to_dict()
    assert set(first['train']['nli_label']) == {'entailment', 'neutral', 'contradiction'}
    assert summary['label_counts']['train'] == {'neutral': 10, 'contradiction': 10, 'entailment': 10}
    train_premises = {normalize(row['text1']) for row in first['train']}
    eval_premises = {normalize(row['text1']) for row in first['validation']}
    assert train_premises.isdisjoint(eval_premises)
    for split, data in first.items():
        pairs = set()
        for row in data:
            key = (normalize(row['text1']), normalize(row['text2']))
            assert key not in pairs
            pairs.add(key)
            assert key[0] != 'blocked text'
            assert row['completion'] == names[raw[row['source_row']]['label']]
            assert belongs_to_validation(row['text1'], 42, 30, 9) == (split == 'validation')
            assert len(tokenizer(row['prompt'] + row['completion'] + tokenizer.eos_token)['input_ids']) <= 512
    print('MNLI 三类抽样、可重复性、来源编号、同 premise 隔离与长度检查通过。')
    return first


def tiny_args(path, epochs=1, with_eval=False):
    return SFTConfig(
        output_dir=str(path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2, num_train_epochs=epochs,
        learning_rate=1e-3, optim='adamw_torch',
        completion_only_loss=True, packing=False, padding_free=False,
        max_length=512, loss_type='chunked_nll',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=42, data_seed=42, prediction_loss_only=True,
        eval_strategy='epoch' if with_eval else 'no',
        save_strategy='best' if with_eval else 'no',
        load_best_model_at_end=with_eval,
        metric_for_best_model='eval_pair_f1' if with_eval else None,
        greater_is_better=True if with_eval else None,
        save_total_limit=1, save_only_model=True,
        logging_steps=1, report_to='none', dataloader_pin_memory=False,
    )


def main():
    torch.set_num_threads(2)
    set_seed(42)
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    nli = test_data_selection(tokenizer)
    # 读取实际 C 和 Dev，核对 Prompt 及 token 边界，但不训练完整数据。
    dico = prepare_dico()
    assert len(dico['train']) == 3517 and len(dico['dev']) == 660
    check_examples(dico, tokenizer, cfg.MAX_LENGTH)
    work = cfg.EXPERIMENT_DIR / 'outputs' / f'_cpu_check_{time.time_ns()}'
    work.mkdir(parents=True)
    original_gold = cfg.GOLD_PATH
    original_model = cfg.MODEL_PATH
    cfg.MODEL_PATH = work / 'tiny_base'
    cfg.MNLI_ADAPTER_DIR = work / 'mnli_adapter'
    cfg.PREDICT_BATCH_SIZE = 2

    tiny = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    base = AutoModelForCausalLM.from_config(tiny)
    base.save_pretrained(cfg.MODEL_PATH)
    tokenizer.save_pretrained(cfg.MODEL_PATH)
    del base

    # 第一阶段：真实 SFTTrainer、完整词表 CE、答案+EOS监督、累计梯度。
    model = load_training_model()
    before = {key: value.clone() for key, value in get_peft_model_state_dict(model).items()}
    stage1_data = nli['train'].select(range(4))
    trainer = SFTTrainer(
        model=model, args=tiny_args(work / 'stage1'),
        train_dataset=stage1_data, processing_class=tokenizer,
    )
    check_supervision(trainer, stage1_data, tokenizer)
    trainer.train()
    after = {key: value.clone() for key, value in get_peft_model_state_dict(model).items()}
    assert any(not torch.equal(before[key], after[key]) for key in before)
    save_adapter(trainer, tokenizer, cfg.MNLI_ADAPTER_DIR, {'stage': 'mnli'})
    check_mnli_adapter()
    assert all('lora_' in key for key in after)
    assert not (cfg.MNLI_ADAPTER_DIR / 'model.safetensors').exists()

    # 第二阶段必须载入第一阶段数值，并且 LoRA 仍然可训练。
    transferred = load_training_model(cfg.MNLI_ADAPTER_DIR)
    for key, value in get_peft_model_state_dict(transferred).items():
        torch.testing.assert_close(value, after[key], rtol=0, atol=0)
    assert all(parameter.requires_grad for name, parameter in transferred.named_parameters() if 'lora_' in name)
    model.eval()
    transferred.eval()
    probe = tokenizer('Text 1: a dog\nText 2: an animal\nAnswer:\n', return_tensors='pt')
    with torch.inference_mode():
        old_scores = model(**probe, logits_to_keep=1).logits
        loaded_scores = transferred(**probe, logits_to_keep=1).logits
    torch.testing.assert_close(old_scores, loaded_scores, rtol=0, atol=0)

    # 小 Dev 包含一个 F/B 组、一个 E/E 组和一条 N；用相同官方评分器。
    frame = dico['dev'].to_pandas()
    selected = frame[frame['pair_id'] == frame.iloc[0]['pair_id']]
    e_id = frame[frame['label'] == 'EQUIVALENCE'].iloc[0]['pair_id']
    selected = pd.concat([selected, frame[frame['pair_id'] == e_id], frame[frame['label'] == 'NEGATIVE_OTHER'].head(1)])
    dev = Dataset.from_pandas(selected, preserve_index=False)
    gold = pd.read_csv(original_gold, keep_default_na=False)
    cfg.GOLD_PATH = work / 'tiny_gold.csv'
    gold[gold['instance_id'].isin(dev['instance_id'])].to_csv(cfg.GOLD_PATH, index=False)
    stage2_data = dico['train'].select(range(4))
    trainer2 = SFTTrainer(
        model=transferred, args=tiny_args(work / 'stage2', epochs=2, with_eval=True),
        train_dataset=stage2_data, eval_dataset=dev, processing_class=tokenizer,
        callbacks=[BestF1Callback(dev, tokenizer)],
    )
    check_supervision(trainer2, stage2_data, tokenizer)
    trainer2.train()
    assert trainer2.state.best_model_checkpoint is not None
    assert trainer2.state.best_metric is not None
    assert len(list((work / 'stage2').glob('checkpoint-*'))) == 1
    stage2_weights = get_peft_model_state_dict(transferred)
    assert any(not torch.equal(after[key], stage2_weights[key]) for key in after)
    best_weights = load_file(str(Path(trainer2.state.best_model_checkpoint) / 'adapter_model.safetensors'))
    for key, value in stage2_weights.items():
        torch.testing.assert_close(value, best_weights[key], rtol=0, atol=0)
    final_dir = work / 'best_adapter'
    save_adapter(trainer2, tokenizer, final_dir, {'stage': 'dico'})
    fresh = load_training_model(final_dir)
    for key, value in get_peft_model_state_dict(fresh).items():
        torch.testing.assert_close(value, stage2_weights[key], rtol=0, atol=0)

    # callback 本身必须恢复 RNG、训练模式和 padding_side。
    callback = BestF1Callback(dev, tokenizer)
    tokenizer.padding_side = 'right'
    fresh.train()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    metrics = {}
    callback.on_evaluate(trainer2.args, trainer2.state, trainer2.control, model=fresh, metrics=metrics)
    assert 'eval_pair_f1' in metrics
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert fresh.training and tokenizer.padding_side == 'right'
    result = predict_with_scores(fresh, tokenizer, dev, 2)
    result = pair_aware_decode(result)
    reports = score_methods(result, work / 'final_evaluation', cfg.GOLD_PATH)
    assert reports['pair_aware'].weighted_f1 == trainer2.state.best_metric
    try:
        check_empty_output(final_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError('已有 Adapter 未被覆盖保护拦截。')
    print('两阶段权重接续 / LoRA 更新 / 保存重载 / 最佳 F1 选模 / 官方评分均通过。')
    print('CPU 检查文件：', work)
    print('真实模型仅使用 tokenizer，没有加载大权重：', original_model)


if __name__ == '__main__':
    main()
