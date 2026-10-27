import json
from pathlib import Path

import torch
from transformers import Trainer


class LexicalTrainer(Trainer):
    """沿用 Trainer 的训练/选模，仅适配分组学习率和 LoRA + 小分支的保存。"""
    def __init__(self, *args, lexical_learning_rate=1e-3, run_config=None, **kwargs):
        self.lexical_learning_rate = lexical_learning_rate
        self.run_config = run_config or {}
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        model = self.model if model is None else model
        if self.optimizer is None:
            # 沿用原 AdamW 的 decay / no-decay 分组，仅给词汇分支单独学习率。
            decay_names = self.get_decay_parameter_names(model)
            groups = []
            for is_lexical in [False, True]:
                for use_decay in [True, False]:
                    parameters = [
                        parameter for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                        and name.startswith('lexical_head.') == is_lexical
                        and (name in decay_names) == use_decay
                    ]
                    if parameters:
                        groups.append({
                            'params': parameters,
                            'lr': self.lexical_learning_rate if is_lexical else self.args.learning_rate,
                            'weight_decay': self.args.weight_decay if use_decay else 0.0,
                        })
            optimizer_class, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
            self.optimizer = optimizer_class(groups, **optimizer_kwargs)
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        # 不能直接保存整个 wrapper.state_dict()，其中含 1.7B 基础权重。
        model.save_adapter_and_head(directory)
        self.processing_class.save_pretrained(str(directory))
        (directory / 'run_config.json').write_text(
            json.dumps(self.run_config, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
        )
        torch.save(self.args, directory / 'training_args.bin')

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            raise ValueError('没有记录最佳 checkpoint。')
        model = self.accelerator.unwrap_model(self.model)
        model.load_adapter_and_head(self.state.best_model_checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        raise ValueError('本实验只保存模型；不支持恢复优化器继续训练，请使用新的 RUN_NAME。')
