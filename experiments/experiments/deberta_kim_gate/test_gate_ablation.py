"""验证消融脚本，不加载真实主干、不启动 GPU 推理。"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.deberta_kim_gate import config as cfg
from experiments.deberta_kim_gate.data import read_data
from experiments.deberta_kim_gate.model import DebertaDiCoModel
from experiments.deberta_kim_gate.evaluation import predict_logits, score_logits
from experiments.deberta_kim_gate.evaluate_gate_ablation import (
    MODEL_DIR, VARIANTS, use_variant, check_saved_config,
    check_reproduction, compare_predictions,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(32, 16)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    saved = check_saved_config(MODEL_DIR)
    assert saved['dev_batch_size'] == 16
    model = DebertaDiCoModel(TinyBackbone(), SimpleNamespace(cls_token_id=1, sep_token_id=2), 20)
    # 模拟训练后的非零投影和门控，才能检查不同开关是否真的改变输出。
    with torch.no_grad():
        nn.init.normal_(model.match_output[-1].weight, std=0.1)
        nn.init.normal_(model.relation_projection.weight, std=0.2)
        nn.init.normal_(model.knowledge_gate.source.weight, std=0.2)
        model.knowledge_gate.relation.bias.fill_(1.0)
    _, dev = read_data()
    ids = torch.tensor([[1, 3, 4, 2, 5, 6, 2, 0]]).repeat(len(dev), 1)
    masks = ids.ne(0).long()
    lexical = torch.rand(len(dev), 20)
    relations = torch.zeros(len(dev), 8, 8, 4, dtype=torch.uint8)
    relations[:, 1, 4, 1] = 1
    labels = torch.tensor(dev['label'].map(cfg.LABEL_TO_ID).tolist())
    loader = DataLoader(TensorDataset(ids, masks, lexical, relations, labels), batch_size=16)
    weights_before = {key: value.clone() for key, value in model.state_dict().items()}
    output = cfg.EXPERIMENT_DIR / 'outputs' / f'_ablation_cpu_{time.time_ns()}'
    output.mkdir(parents=True)
    predictions = {}
    learned_frame = None
    for variant, _ in VARIANTS:
        model.train()
        previous_rng = torch.get_rng_state().clone()
        with use_variant(model, variant):
            # predict_logits 会恢复调用前的 training 状态。
            logits = predict_logits(model, loader, torch.device('cpu'))
            assert model.training
        assert model.use_kim_features is True and model.knowledge_gate.mode == 'learned'
        # DataLoader 可消耗 RNG；无需声称该独立评估入口恢复训练的 RNG。
        assert torch.get_rng_state().shape == previous_rng.shape
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
        predictions[variant] = logits
        summary = score_logits(logits, dev, output / variant)
        current = pd.read_csv(output / variant / 'predictions_with_scores.csv', keep_default_na=False)
        if variant == 'learned':
            learned_frame = current
            best_info = {'weighted_f1': float(summary.loc[summary.method == 'pair_aware', 'weighted_f1'].iloc[0])}
            check_reproduction(current, current.sample(frac=1, random_state=7), summary,
                               best_info, output / 'reproduction')
            wrong_reference = current.copy()
            before = wrong_reference.loc[0, 'prediction']
            wrong_reference.loc[0, 'prediction'] = 'F' if before != 'F' else 'N'
            try:
                check_reproduction(current, wrong_reference, summary, best_info, output / 'expected_mismatch')
            except RuntimeError:
                pass
            else:
                raise AssertionError('历史预测不一致时必须停止。')
        else:
            changes = compare_predictions(learned_frame, current, output / variant / 'changes')
            assert (changes['changed'] == changes['fixed'] + changes['broken'] + changes['changed_wrong']).all()
    assert not torch.equal(predictions['learned'], predictions['fixed_half'])
    assert not torch.equal(predictions['fixed_half'], predictions['fixed_one'])
    assert not torch.equal(predictions['fixed_one'], predictions['no_knowledge'])
    model.eval()
    with torch.inference_mode():
        zero_relations = model(ids[:2], masks[:2], lexical[:2], torch.zeros_like(relations[:2]))
        with use_variant(model, 'no_knowledge'):
            disabled = model(ids[:2], masks[:2], lexical[:2], relations[:2])
    torch.testing.assert_close(zero_relations, disabled, rtol=0, atol=0)
    try:
        with use_variant(model, 'fixed_one'):
            raise RuntimeError('模拟中途报错')
    except RuntimeError:
        pass
    assert model.use_kim_features and model.knowledge_gate.mode == 'learned'
    print('检查通过：四个分支生效、参数不变、异常恢复、历史预测校验、按 ID 对齐及官方评分。')
    print('CPU 结果仅用于代码检查，不是实际模型成绩：', output)


if __name__ == '__main__':
    main()
