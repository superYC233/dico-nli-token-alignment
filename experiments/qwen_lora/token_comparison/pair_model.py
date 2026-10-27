"""Qwen + LoRA + 比较头；可选择是否加入最后有效 token 的整体信息。"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import AutoModel

from pair_data import LABELS


class ComparisonHead(nn.Module):
    def __init__(self, hidden_size, head_type, dimension=128, dropout=0.1):
        super().__init__()
        if head_type not in ['simple', 'alignment', 'alignment_global']:
            raise ValueError('head_type 只能是 simple、alignment 或 alignment_global。')
        self.head_type = head_type
        self.use_global = head_type == 'alignment_global'
        self.projection = nn.Linear(hidden_size, dimension)
        self.compare = nn.Sequential(nn.Linear(4 * dimension, dimension), nn.ReLU())
        # 每段 mean+max 得到 2d；拼接 a、b、a-b、a*b，得到 8d。
        # 新版本再加入 d 维整体向量；原来的两个版本仍保持 8d。
        classifier_size = (9 if self.use_global else 8) * dimension
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_size),
            nn.Dropout(dropout),
            nn.Linear(classifier_size, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 4),
        )

    def align(self, tokens, scores, target_mask):
        mask = target_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(weights, tokens)

    def pool(self, features, mask):
        float_mask = mask.unsqueeze(-1).to(features.dtype)
        mean = (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)
        maximum = features.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(features.dtype).min
        ).max(dim=1).values
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum,
                              torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=-1)

    def forward(self, hidden, mask_a, mask_b, attention_mask=None):
        # 比较模块固定用 float32，训练和单独预测时采用同一种计算方式。
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            tokens = self.projection(hidden.float())
            if self.head_type in ['alignment', 'alignment_global']:
                normalized = F.normalize(tokens, dim=-1)
                scores = torch.bmm(normalized, normalized.transpose(1, 2)) / 0.1
                # A 中的 token 参考 B；B 中的 token 参考 A。
                reference_a = self.align(tokens, scores, mask_b)
                reference_b = self.align(tokens, scores, mask_a)
            else:
                # 对照：参考自身，不做跨文本匹配；层数、参数量保持相同。
                reference_a = tokens
                reference_b = tokens

            features_a = self.compare(torch.cat([
                tokens, reference_a, tokens - reference_a, tokens * reference_a
            ], dim=-1))
            features_b = self.compare(torch.cat([
                tokens, reference_b, tokens - reference_b, tokens * reference_b
            ], dim=-1))
            a = self.pool(features_a, mask_a)
            b = self.pool(features_b, mask_b)
            features = [a, b, a - b, a * b]
            if self.use_global:
                if attention_mask is None:
                    raise ValueError('alignment_global 需要 attention_mask 来排除补齐位置。')
                # 取每行最后一个有效位置，不能直接用 -1（右侧可能是 padding）。
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                last_positions = positions.unsqueeze(0).expand_as(attention_mask)
                last_positions = last_positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
                if (last_positions < 0).any():
                    raise ValueError('输入不能全部是 padding。')
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                # 此位置位于两段正文之后，可以看到完整输入；复用已有投影层。
                global_features = tokens[batch_indices, last_positions]
                features.append(global_features)
            return self.classifier(torch.cat(features, dim=-1))


class PhraseClassifier(nn.Module):
    def __init__(self, backbone, experiment_config):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.experiment_config = dict(experiment_config)
        self.head = ComparisonHead(
            self.config.hidden_size,
            experiment_config['head_type'],
            experiment_config['dimension'],
            experiment_config['head_dropout'],
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, **kwargs
        )
        self.backbone.enable_input_require_grads()

    def forward(self, input_ids, attention_mask, mask_a, mask_b, labels=None):
        positions = attention_mask.cumsum(dim=1) - 1
        positions = positions.masked_fill(attention_mask == 0, 0)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        logits = self.head(outputs.last_hidden_state, mask_a, mask_b, attention_mask)
        result = {'logits': logits}
        if labels is not None:
            # 所有版本都只用相同的四分类 CE，不叠加 Pair CE 或 R-Drop。
            result['loss'] = F.cross_entropy(logits, labels)
        return result

    def save_checkpoint(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory / 'adapter')
        torch.save(self.head.state_dict(), directory / 'comparison_head.pt')
        tokenizer.save_pretrained(directory)
        (directory / 'comparison_config.json').write_text(
            json.dumps(self.experiment_config, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def restore_checkpoint(self, directory):
        directory = Path(directory)
        saved = json.loads((directory / 'comparison_config.json').read_text(encoding='utf-8'))
        if saved != self.experiment_config:
            raise ValueError('checkpoint 配置与当前比较模型不一致。')
        adapter_state = load_peft_weights(str(directory / 'adapter'), device='cpu')
        set_peft_model_state_dict(self.backbone, adapter_state, adapter_name='default')
        head_state = torch.load(directory / 'comparison_head.pt', map_location='cpu', weights_only=True)
        self.head.load_state_dict(head_state, strict=True)


def build_model(config):
    backbone = AutoModel.from_pretrained(config['base_model_path'], dtype='auto')
    # FEATURE_EXTRACTION 表示我们使用 token 表示，不再通过 LM head 生成标签。
    backbone = get_peft_model(backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules='all-linear', bias='none', task_type='FEATURE_EXTRACTION'
    ))
    return PhraseClassifier(backbone, config)


def load_model(directory, device):
    directory = Path(directory)
    config = json.loads((directory / 'comparison_config.json').read_text(encoding='utf-8'))
    if config['labels'] != LABELS:
        raise ValueError('保存的类别顺序不是 F/B/E/N。')
    backbone = AutoModel.from_pretrained(config['base_model_path'], dtype='auto')
    backbone = PeftModel.from_pretrained(backbone, directory / 'adapter')
    model = PhraseClassifier(backbone, config)
    head_state = torch.load(directory / 'comparison_head.pt', map_location='cpu', weights_only=True)
    model.head.load_state_dict(head_state, strict=True)
    return model.to(device).eval(), config
