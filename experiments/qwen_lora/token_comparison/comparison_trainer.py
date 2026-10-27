from pathlib import Path

from transformers import Trainer

from pair_metrics import make_prediction_frame, score_prediction_frame


class ComparisonTrainer(Trainer):
    def __init__(self, dev_frame, results_dir, **kwargs):
        super().__init__(**kwargs)
        self.dev_frame = dev_frame.reset_index(drop=True).copy()
        self.results_dir = Path(results_dir)
        self.compute_metrics = self.compute_dev_metrics
        self.model_accepts_loss_kwargs = False

    def compute_dev_metrics(self, evaluation):
        # 直接使用本轮评估得到的分数，不再额外调用 generate 预测一次。
        predictions = make_prediction_frame(self.dev_frame, evaluation.predictions)
        epoch = int(round(self.state.epoch or 0))
        return score_prediction_frame(predictions, self.results_dir / f'epoch_{epoch}')

    def _save(self, output_dir=None, state_dict=None):
        # Trainer 的原生 checkpoint 轮换机制保留；只改变模型文件的保存格式。
        model = self.accelerator.unwrap_model(self.model)
        model.save_checkpoint(output_dir or self.args.output_dir, self.processing_class)

    def _load_best_model(self):
        # 除了 Adapter，必须同时恢复最佳轮次的比较头。
        if self.state.best_model_checkpoint is not None:
            model = self.accelerator.unwrap_model(self.model)
            model.restore_checkpoint(self.state.best_model_checkpoint)
