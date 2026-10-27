"""原DeBERTa四分类+Token Alignment，只给编码器添加LoRA。"""
import json
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_lexical.model import DebertaDiCoModel


def add_lora(model, settings):
    # 在CPU注入。恢复随机数，避免新增A/B矩阵改变后面的shuffle/dropout序列。
    with torch.random.fork_rng(devices=[]):
        config = LoraConfig(**settings)
        model.deberta_model = inject_adapter_in_model(config, model.deberta_model)
    # 只包装编码器，因此output、align_projection、compare_layer、match_output继续训练。
    return model


def build_model(model_name, tokenizer, use_lora, lora_settings, revision=None):
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=get_model_cache_dir(), local_files_only=True,
        revision=revision, dtype=torch.float32, use_safetensors=True,
    )
    model = DebertaDiCoModel(backbone, tokenizer, feature_size=0, use_lexical_features=False)
    if use_lora:
        add_lora(model, lora_settings)
    return model


def parameter_report(model):
    records = []
    for name, parameter in model.named_parameters():
        if 'lora_A.' in name or 'lora_B.' in name:
            group = 'lora'
        elif name.startswith('deberta_model.'):
            group = 'original_encoder'
        else:
            group = 'task_heads'
        records.append({
            'name': name, 'group': group, 'parameters': parameter.numel(),
            'trainable': parameter.requires_grad,
        })
    return records


def save_trainable_weights(model, path):
    # LoRA的A/B和两个任务头一起保存；冻结的NLI编码器不用重复保存。
    path = Path(path)
    weights = {name: parameter.detach().cpu()
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    temporary = path.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(path)


def load_trainable_weights(model, path):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(weights) != expected:
        raise ValueError('保存的参数与当前LoRA/任务头不一致，请使用该模型目录自己的run_config.json。')
    # 缺失的键是冻结编码器，它们已由同一NLI初始化恢复；所有任务头必须存在。
    result = model.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys


def load_saved_model(model_dir, device):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    model = build_model(
        config['model_name'], tokenizer, config['use_lora'], config['lora'],
        revision=config['base_revision'],
    )
    load_trainable_weights(model, model_dir / 'best_model.pth')
    model.to(device).eval()
    return model, tokenizer, config
