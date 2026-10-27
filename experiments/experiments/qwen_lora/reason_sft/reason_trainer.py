"""标签与解释分别做completion-only CE，再加权；没有Pair CE。"""
from trl import SFTTrainer


class ReasonSFTTrainer(SFTTrainer):
    def __init__(self, reason_weight, **kwargs):
        super().__init__(**kwargs)
        self.reason_weight = reason_weight
        # 两项分别按本批监督token归一化，不使用标签token数量作解释损失分母。
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        reason_inputs = inputs.pop('reason_inputs', None)
        label_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None,
        )
        reason_loss = label_loss.new_zeros(())
        reason_rows = 0
        if model.training and self.reason_weight > 0 and reason_inputs is not None:
            # SFTTrainer已经把同一模型切为chunked_nll；解释前向也只投影监督位置。
            reason_outputs = model(**reason_inputs, use_cache=False)
            reason_loss = reason_outputs.loss
            reason_rows = len(reason_inputs['input_ids'])
        loss = label_loss + self.reason_weight * reason_loss
        if model.training:
            self._metrics['train']['label_loss'].append(label_loss.detach().item())
            self._metrics['train']['reason_loss'].append(reason_loss.detach().item())
            self._metrics['train']['reason_rows'].append(reason_rows)
        return (loss, outputs) if return_outputs else loss
