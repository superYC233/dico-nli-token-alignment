"""原 completion-only SFT + 文本对级监督对比损失，不是三组合 Pair CE。"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from trl import SFTTrainer


def supervised_contrastive_loss(features, labels, groups, temperature):
    if temperature <= 0:
        raise ValueError('temperature 必须大于 0。')

    # 每行代表整个有序文本对，不是单独一个词或一个句子。
    features = F.normalize(features.float(), dim=-1)
    count = len(features)

    if count < 2:
        return features.sum() * 0, features.new_tensor(0.0)

    same_label = labels[:, None] == labels[None, :]
    same_group = groups[:, None] == groups[None, :]
    diagonal = torch.eye(count, dtype=torch.bool, device=features.device)

    # 同源、同标签的副本既不作正例，也不作负例。
    # 同源 F/B 是方向不同的类别，仍可作负例。
    candidates = ~diagonal & ~(same_label & same_group)
    positives = candidates & same_label
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    valid_fraction = valid.float().mean()

    if not valid.any():
        return features.sum() * 0, valid_fraction

    similarities = features @ features.T / temperature
    # 用有限的极小值，防止某行没有候选时 logsumexp 产生无穷或 NaN。
    denominator_scores = similarities.masked_fill(
        ~candidates, torch.finfo(similarities.dtype).min
    )
    log_probabilities = similarities - torch.logsumexp(denominator_scores, dim=1, keepdim=True)
    positive_log_probabilities = torch.where(
        positives, log_probabilities, torch.zeros_like(log_probabilities)
    )
    mean_log_probability = positive_log_probabilities.sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_probability[valid].mean()
    return loss, valid_fraction


class ContrastiveCollator:
    def __init__(self, original_collator):
        self.original_collator = original_collator

    def __call__(self, examples):
        features = []
        groups = []

        for example in examples:
            example = dict(example)
            groups.append(example.pop('contrast_group', -1))
            features.append(example)

        batch = self.original_collator(features)
        batch['contrast_group'] = torch.tensor(groups, dtype=torch.long)
        return batch


class ContrastiveSFTTrainer(SFTTrainer):
    def __init__(self, label_ids, contrastive_weight, temperature, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.label_ids = list(label_ids)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.data_collator = ContrastiveCollator(self.data_collator)

        if contrastive_weight < 0 or temperature <= 0:
            raise ValueError('contrastive_weight 不能小于 0；temperature 必须大于 0。')
        if self.padding_free or self.args.packing:
            raise ValueError('请保持 packing=False、padding_free=False。')
        if not self.completion_only_loss:
            raise ValueError('需要 completion_only_loss=True，才能定位答案。')
        if self.args.remove_unused_columns:
            raise ValueError('需要 remove_unused_columns=False，保留训练分组编号。')
        if 'contrast_group' not in self.train_dataset.column_names:
            raise ValueError('训练数据缺少 contrast_group，请先调用 add_contrast_groups。')

        base = self.model.get_base_model()
        if base.config.model_type != 'qwen3':
            raise ValueError('当前脚本只针对 Qwen3，不直接适用于其他主干。')
        if not hasattr(self.model, 'contrastive_head'):
            raise ValueError('模型缺少 contrastive_head，请按训练入口创建投影层。')

    def forward_with_features(self, model, inputs, positions, num_items_in_batch=None):
        """一次原 SFT 前向，同时取得标签之前的隐藏向量，不额外跑一次 Qwen。"""
        unwrapped = self.accelerator.unwrap_model(model)
        base = unwrapped.get_base_model()
        captured = []

        def capture_answer_hidden(module, args, outputs):
            hidden = outputs.last_hidden_state
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            # 不 detach；对比损失需要沿这里更新 LoRA。
            captured.append(hidden[rows, positions])

        handle = base.base_model.register_forward_hook(capture_answer_hidden)
        try:
            sft_loss, outputs = super().compute_loss(
                model, dict(inputs), return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )
        finally:
            # 及时移除，避免影响梯度检查点的重算、评估和 generate。
            handle.remove()

        if len(captured) != 1:
            raise ValueError('没有正确取得答案前的隐藏向量，请检查模型结构。')

        # 小投影层使用 float32，提高相似度和对比损失计算的稳定性。
        with torch.autocast(device_type=captured[0].device.type, enabled=False):
            features = unwrapped.contrastive_head(captured[0].float())
        return sft_loss, outputs, features

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        groups = inputs.pop('contrast_group', None)

        # Dev loss 沿用标准 SFT；选模型依旧看 BestF1Callback 的 pair-aware F1。
        # 权重为 0 时，也完全走标准 SFT 路径，方便做同代码的零权重对照。
        if not model.training or self.contrastive_weight == 0:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch
            )

        if groups is None or (groups < 0).any():
            raise ValueError('训练批次缺少有效的同源分组编号。')
        labels = inputs['labels']
        supervised = labels != -100
        if not supervised.any(dim=1).all():
            raise ValueError('存在没有答案监督的训练样本。')

        first_answer = supervised.long().argmax(dim=1)
        if (first_answer == 0).any():
            raise ValueError('答案之前没有 Prompt，无法定位关系表示。')
        rows = torch.arange(len(labels), device=labels.device)
        answer_ids = labels[rows, first_answer]
        allowed = torch.tensor(self.label_ids, device=labels.device)
        if not torch.isin(answer_ids, allowed).all():
            raise ValueError('首个监督 token 不是 F/B/E/N，请检查 completion 与缓存。')

        # 位置 t-1 预测位置 t 的标签，不能取已经读入正确标签的位置 t。
        positions = first_answer - 1
        sft_loss, outputs, features = self.forward_with_features(
            model, inputs, positions, num_items_in_batch
        )
        contrastive_loss, valid_fraction = supervised_contrastive_loss(
            features, answer_ids, groups, self.temperature
        )
        loss = sft_loss + self.contrastive_weight * contrastive_loss

        self._metrics['train']['sft_loss'].append(sft_loss.detach().item())
        self._metrics['train']['contrastive_loss'].append(contrastive_loss.detach().item())
        self._metrics['train']['weighted_contrastive_loss'].append(
            self.contrastive_weight * contrastive_loss.detach().item()
        )
        self._metrics['train']['contrastive_valid_fraction'].append(valid_fraction.detach().item())
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # 仍保存标准 LoRA Adapter，原来的预测代码可以直接加载。
        super()._save(output_dir, state_dict)
        directory = Path(output_dir or self.args.output_dir)
        model = self.accelerator.unwrap_model(self.model)
        torch.save(model.contrastive_head.state_dict(), directory / 'contrastive_head.pt')
        config = {
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature,
            'label_ids': self.label_ids,
            'projection_dim': model.contrastive_head[-1].out_features,
            'prompt_template': self.prompt_template,
        }
        (directory / 'contrastive_config.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def _load_best_model(self):
        super()._load_best_model()
        if self.state.best_model_checkpoint is not None:
            path = Path(self.state.best_model_checkpoint) / 'contrastive_head.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            model = self.accelerator.unwrap_model(self.model)
            model.contrastive_head.load_state_dict(state, strict=True)

