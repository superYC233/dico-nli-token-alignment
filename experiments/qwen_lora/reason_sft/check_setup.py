"""检查全部C数据、Prompt、损失及LoRA梯度；不创建优化器、不训练。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig

from experiments.qwen_lora.reason_sft import config as cfg
from experiments.qwen_lora.reason_sft.reason_data import (
    load_train_frame, build_dataset, ReasonCollator, reverse_reason_reference,
)
from experiments.qwen_lora.reason_sft.reason_trainer import ReasonSFTTrainer
from experiments.qwen_lora.reason_sft.train_lora_reason import load_tokenizer
from experiments.qwen_lora.reason_sft.reason_evaluation import pair_aware_decode


def expected_ce(model, batch):
    with torch.no_grad():
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           batch['labels'][:, 1:].reshape(-1), ignore_index=-100)


def check_loss():
    torch.set_num_threads(2)
    vocab = {word: index for index, word in enumerate(['<pad>', '<eos>', '<unk>'] + [f'w{i}' for i in range(61)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    config = Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, eos_token_id=1, bos_token_id=None, attention_dropout=0.0,
    )
    torch.manual_seed(42)
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0, target_modules='all-linear',
        bias='none', task_type='CAUSAL_LM',
    ))
    examples = [
        {'input_ids': [3, 4, 5, 1], 'attention_mask': [1]*4, 'labels': [-100, -100, 5, 1],
         'reason_input_ids': [3, 7, 8, 9, 10, 1], 'reason_labels': [-100, -100, -100, 9, 10, 1]},
        {'input_ids': [3, 6, 7, 5, 1], 'attention_mask': [1]*5, 'labels': [-100, -100, -100, 5, 1],
         'reason_input_ids': [], 'reason_labels': []},
    ]
    dataset = Dataset.from_list(examples)
    args = SFTConfig(
        output_dir=str(cfg.RESULTS_ROOT / 'setup_check'), use_cpu=True, bf16=False, fp16=False,
        packing=False, padding_free=False, completion_only_loss=True, loss_type='chunked_nll',
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=1,
        report_to='none', save_strategy='no', eval_strategy='no', prediction_loss_only=True,
    )
    collator = ReasonCollator(0)
    trainer = ReasonSFTTrainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
                               processing_class=tokenizer, data_collator=collator, reason_weight=0.2)
    model.train()
    batch = collator(examples)
    label_batch = {key: value for key, value in batch.items() if key != 'reason_inputs'}
    label_expected = expected_ce(model, label_batch)
    reason_expected = expected_ce(model, batch['reason_inputs'])
    actual = trainer.compute_loss(model, batch)
    assert torch.allclose(actual, label_expected + 0.2 * reason_expected, atol=1e-6)
    assert trainer.optimizer is None
    # 单独解释损失必须能更新同一套LoRA；这里只求梯度，不执行更新。
    aux = model(**batch['reason_inputs'], use_cache=False).loss
    parameters = [p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    grads = torch.autograd.grad(aux, parameters, allow_unused=True)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    trainer.reason_weight = 0.0
    zero = trainer.compute_loss(model, batch)
    assert torch.allclose(zero, label_expected, atol=1e-6)
    trainer.reason_weight = 0.2
    blank = collator([examples[1]])
    assert blank['reason_inputs'] is None
    assert torch.isfinite(trainer.compute_loss(model, blank))
    # 复制相同的解释样本，均值CE不变，不按标签token数归一化。
    repeated = {key: value.repeat(2, 1) for key, value in batch['reason_inputs'].items()}
    repeated_loss = model(**repeated, use_cache=False).loss
    assert torch.allclose(repeated_loss, reason_expected, atol=1e-6)
    model.eval()
    calls = []
    handle = model.get_base_model().base_model.register_forward_hook(lambda *args: calls.append(1))
    try:
        eval_loss = trainer.compute_loss(model, batch)
    finally:
        handle.remove()
    assert len(calls) == 1 and torch.allclose(eval_loss, label_expected, atol=1e-6)
    metrics = trainer.evaluate()
    assert 'eval_loss' in metrics and metrics['eval_loss'] >= 0
    return {'separately_normalized_losses': True, 'zero_weight_equals_label_ce': True,
            'reason_gradient_reaches_lora': True, 'no_reason_batch_finite': True,
            'eval_uses_only_label_forward': True, 'trainer_eval_path_valid': True,
            'optimizer_created': False, 'training_performed': False,
            'model_check': 'random tiny Qwen3 + LoRA on CPU; not an accuracy test'}


def main():
    tokenizer = load_tokenizer()
    frame = load_train_frame()
    dataset = build_dataset(frame, tokenizer, include_reason=True)
    assert len(frame) == 3517 and frame.use_reason.sum() == 2040
    assert frame.instance_id.str.endswith('__swap').sum() == 475
    original = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    assert frame.iloc[:3042][original.columns].reset_index(drop=True).equals(original)
    # 原短Prompt逐字符核对；导入历史data_utils仅加载本地tokenizer，不加载主干。
    from experiments.qwen_lora.data_utils import build_prompt
    assert cfg.LABEL_PROMPT == build_prompt('{text1}', '{text2}')
    assert reverse_reason_reference('The second phrase adds color; second marks event order.') == (
        'The first phrase adds color; second marks event order.')
    for row, item in zip(frame.itertuples(), dataset):
        prompt_ids = tokenizer.encode(cfg.LABEL_PROMPT.format(text1=row.text1, text2=row.text2), add_special_tokens=False)
        assert item['input_ids'][:len(prompt_ids)] == prompt_ids
        assert all(x == -100 for x in item['labels'][:len(prompt_ids)])
        supervised = [x for x in item['labels'] if x != -100]
        assert supervised == tokenizer.encode(cfg.LABEL_MAP[row.label], add_special_tokens=False) + [tokenizer.eos_token_id]
        assert bool(item['reason_input_ids']) == row.use_reason
        if row.use_reason:
            reason_prompt = cfg.REASON_PROMPT.format(text1=row.text1, text2=row.text2)
            n = len(tokenizer.encode(reason_prompt, add_special_tokens=False))
            assert item['reason_labels'][:n] == [-100]*n
            target = [x for x in item['reason_labels'] if x != -100]
            assert tokenizer.decode(target, skip_special_tokens=True) == row.reason
    # Dev分词只构造标签目标，没有理由文件列也可以运行。
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    dev_data = build_dataset(dev, tokenizer, include_reason=False)
    assert all(not row['reason_input_ids'] for row in dev_data)
    historical = pd.read_csv(cfg.QWEN_DIR / 'outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv')
    recomputed = pair_aware_decode(historical)
    assert recomputed.pair_prediction.tolist() == historical.pair_prediction.tolist()
    result = {'label_rows': len(frame), 'reason_rows': int(frame.use_reason.sum()),
              'dev_rows': len(dev), 'historical_label_prompt_unchanged': True,
              'label_and_eos_mask_checked_all_rows': True, 'reason_mask_checked_all_eligible_rows': True,
              'historical_pair_decoding_unchanged': True, **check_loss()}
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.PREPARED_DIR / 'setup_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
