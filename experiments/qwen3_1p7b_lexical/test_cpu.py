"""只运行随机微型 Qwen 和真实 tokenizer，不加载 1.7B 大权重。"""
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3Config, TrainingArguments, set_seed

from experiments.qwen3_1p7b_lexical import config as cfg
from experiments.qwen3_1p7b_lexical.data_utils import (
    load_data, load_tokenizer, build_train_dataset, build_prompt_template,
    get_datasets, check_examples, prepare_training_dataset, LexicalCollator, get_label_ids,
)
from experiments.qwen3_1p7b_lexical.model import QwenLexicalModel
from experiments.qwen3_1p7b_lexical.prediction import predict_with_scores
from experiments.qwen3_1p7b_lexical.lexical_trainer import LexicalTrainer
from experiments.qwen3_1p7b_lexical.best_f1_callback import BestF1Callback
from experiments.generative_nli import predict_with_scores as original_predict


def tiny_config(vocab_size, eos_token_id=7, pad_token_id=0):
    return Qwen3Config(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512, tie_word_embeddings=True,
        bos_token_id=None, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
    )


def wrap_lora(base, dropout=0.0):
    base.config.use_cache = False
    return get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, target_modules='all-linear', lora_dropout=dropout, task_type='CAUSAL_LM',
    ))


def test_full_vocabulary_loss():
    set_seed(42)
    language_model = wrap_lora(AutoModelForCausalLM.from_config(tiny_config(128)))
    reference = copy.deepcopy(language_model)
    rng = torch.get_rng_state().clone()
    model = QwenLexicalModel(language_model, [3, 4, 5, 6], 7, chunk_size=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(parameter.numel() for parameter in model.lexical_head.parameters()) == 804
    inputs = torch.tensor([[1, 8, 9, 10, 3, 7, 0], [1, 11, 12, 13, 14, 6, 7]])
    masks = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]])
    labels = torch.tensor([[-100, -100, -100, -100, 3, 7, -100], [-100, -100, -100, -100, -100, 6, 7]])
    features = torch.linspace(0.0, 1.0, 40).reshape(2, 20)
    model.eval()
    reference.eval()
    loss = model(inputs, masks, features, labels=labels).loss
    original = reference(input_ids=inputs, attention_mask=masks, labels=labels, use_cache=False).loss
    torch.testing.assert_close(loss, original, rtol=1e-6, atol=1e-6)
    loss.backward()
    original.backward()
    ours = dict(language_model.named_parameters())
    for name, parameter in reference.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(ours[name].grad, parameter.grad, rtol=2e-5, atol=1e-7)
    assert model.lexical_head[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in ours.items() if 'lora_' in n)

    # 手动构造完整词表 logits，只在正确的答案前一位置修正四列。
    with torch.no_grad():
        model.lexical_head[-1].weight.normal_(0, 0.2)
        model.lexical_head[-1].bias.copy_(torch.tensor([0.2, -0.3, 0.4, -0.1]))
    raw = language_model(input_ids=inputs, attention_mask=masks, use_cache=False).logits.float()
    corrected = raw.clone()
    correction = model.lexical_correction(features)
    positions = torch.tensor([3, 4])
    for row in range(2):
        corrected[row, positions[row], model.label_ids] += correction[row]
        # EOS 的预测发生在 label 所在位置，此处不能添加修正。
        torch.testing.assert_close(corrected[row, positions[row] + 1], raw[row, positions[row] + 1], rtol=0, atol=0)
    expected = F.cross_entropy(corrected[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1))
    actual = model(inputs, masks, features, labels=labels).loss
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    scaled = model(inputs, masks, features, labels=labels, num_items_in_batch=8).loss
    torch.testing.assert_close(scaled, actual / 2, rtol=1e-6, atol=1e-6)
    model.train()
    model.gradient_checkpointing_enable({'use_reentrant': False})
    model.zero_grad()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        mixed_loss = model(inputs, masks, features, labels=labels).loss
    mixed_loss.backward()
    assert torch.isfinite(mixed_loss)
    assert model.lexical_head[0].weight.grad.abs().sum() > 0
    assert model.lexical_head[-1].weight.grad.abs().sum() > 0
    print('全词表 CE、标签/EOS位置、梯度累计缩放、LoRA/词汇梯度及 bf16 检查通过。')


def main():
    PartialState(cpu=True)
    torch.set_num_threads(2)
    test_full_vocabulary_loss()
    work = Path(__file__).resolve().parent / 'outputs/_cpu_check'
    work.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.MODEL_PATH)
    template = build_prompt_template(tokenizer)
    dataset = load_data()
    dataset['train'] = build_train_dataset(dataset['train'], cfg.MODE)
    assert len(dataset['train']) == 3517 and len(dataset['dev']) == 660
    formatted = get_datasets(dataset, template)
    check_examples(formatted, tokenizer, cfg.MAX_LENGTH)
    train_dataset = prepare_training_dataset(formatted['train'], tokenizer)
    indices = []
    for label in cfg.LABELS:
        indices.extend([i for i, row in enumerate(formatted['train']) if row['completion'] == label][:2])
    train = train_dataset.select(indices)
    collator = LexicalCollator(tokenizer.pad_token_id)
    batch = collator([train[i] for i in range(4)])
    assert batch['lexical_features'].shape == (4, 20)
    assert ((batch['labels'] != -100).sum(dim=1) == 2).all()

    set_seed(42)
    base = AutoModelForCausalLM.from_config(tiny_config(
        len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id,
    ))
    base.save_pretrained(work / 'tiny_base')
    model = QwenLexicalModel(wrap_lora(base, 0.05), get_label_ids(tokenizer), tokenizer.eos_token_id)
    small_dev = formatted['dev'].select(range(4))
    before = original_predict(model.language_model, tokenizer, small_dev, 2)
    after = predict_with_scores(model, tokenizer, small_dev, 2)
    for column in ['prediction', 'generated_prediction', 'logit_F', 'logit_B', 'logit_E', 'logit_N']:
        assert before[column].equals(after[column]), column
    assert tokenizer.padding_side == 'right'
    print('零初始化词汇修正时，真实 tokenizer 的生成结果和四类 logits 与旧函数逐条完全一致。')

    args = TrainingArguments(
        output_dir=str(work / 'training'), use_cpu=True, bf16=False,
        max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=1, save_strategy='best',
        load_best_model_at_end=True, metric_for_best_model='eval_pair_f1', greater_is_better=True,
        save_total_limit=1, save_only_model=True, prediction_loss_only=True,
        label_names=['labels'], remove_unused_columns=False,
        learning_rate=1e-4, logging_steps=1, report_to='none', dataloader_pin_memory=False,
    )
    trainer = LexicalTrainer(
        model=model, args=args, train_dataset=train, eval_dataset=train,
        data_collator=collator, processing_class=tokenizer,
        lexical_learning_rate=1e-3, run_config={'cpu_test': True},
        callbacks=[BestF1Callback(formatted['dev'], tokenizer, 16)],
    )
    optimizer = trainer.create_optimizer()
    lexical_ids = {id(p) for p in model.lexical_head.parameters()}
    counted = []
    for group in optimizer.param_groups:
        for p in group['params']:
            counted.append(id(p))
            assert group['lr'] == (1e-3 if id(p) in lexical_ids else 1e-4)
    assert len(counted) == len(set(counted))
    assert set(counted) == {id(p) for p in model.parameters() if p.requires_grad}
    trainer.train()
    assert trainer.state.best_model_checkpoint is not None
    assert trainer.state.best_metric is not None
    assert tokenizer.padding_side == 'right'
    # 自动回载最佳时，head 也必须和最佳 checkpoint 完全一致。
    saved_head = torch.load(Path(trainer.state.best_model_checkpoint) / 'lexical_head.pt', weights_only=True)
    for name, value in model.lexical_head.state_dict().items():
        torch.testing.assert_close(value, saved_head[name], rtol=0, atol=0)
    assert saved_head['2.weight'].abs().sum() > 0
    trainer.save_model(str(work / 'best_adapter'))
    assert not (work / 'best_adapter/model.safetensors').exists()
    assert (work / 'best_adapter/adapter_model.safetensors').exists()
    tokenizer.save_pretrained(work / 'best_adapter')
    # 全新实例加载，不能只测同一内存对象。
    reloaded_lm = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(work / 'tiny_base', local_files_only=True),
        str(work / 'best_adapter'),
    )
    reloaded = QwenLexicalModel(reloaded_lm, get_label_ids(tokenizer), tokenizer.eos_token_id)
    reloaded.load_adapter_and_head(work / 'best_adapter')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), reloaded.state_dict()[name].cpu(), rtol=0, atol=0)
    before = predict_with_scores(model, tokenizer, small_dev, 2)
    after = predict_with_scores(reloaded, tokenizer, small_dev, 2)
    assert before.equals(after)
    checkpoints = list((work / 'training').glob('checkpoint-*'))
    assert len(checkpoints) == 1
    print('两步真实 Trainer 流程、四类分组学习率、最佳 F1 选模和 LoRA+词汇分支完整重载通过。')
    print('仅保留一份最佳 checkpoint；未保存基础大权重。')
    print('全部 CPU 检查通过；测试中的随机模型分数不代表实际模型效果。')


if __name__ == '__main__':
    main()
