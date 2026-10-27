"""CPU前向/反向及保存检查；不创建优化器、不更新模型参数。"""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DebertaV2Config

from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional.data import (
    FULL_LABELS, REVERSE, BidirectionalDataset, collate_units, model_inputs, write_train_reference,
)
from experiments.deberta_bidirectional.model import (
    BidirectionalDeberta, build_model, save_weights, load_weights, load_saved_model,
)
from experiments.deberta_bidirectional.evaluation import predict_logits, score_predictions
from experiments.deberta_bidirectional import train_deberta_bidirectional as cfg


class ToyTokenizer:
    def __call__(self, first, second, truncation, padding, max_length, return_tensors):
        vocabulary = {word: index + 3 for index, word in enumerate(sorted(
            set(' '.join(first + second).split())
        ))}
        sequences = [[1] + [vocabulary[word] for word in a.split()] + [2]
                     + [vocabulary[word] for word in b.split()] + [2]
                     for a, b in zip(first, second)]
        ids = torch.tensor([row + [0] * (max_length - len(row)) for row in sequences])
        return {'input_ids': ids, 'attention_mask': ids.ne(0).long()}


def toy_model():
    torch.manual_seed(7)
    config = DebertaV2Config(
        vocab_size=128, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        relative_attention=True, pos_att_type=['p2c', 'c2p'], share_att_key=True,
        max_relative_positions=16, position_buckets=8, position_biased_input=False,
    )
    backbone = AutoModel.from_config(config)
    model = BidirectionalDeberta(backbone, dimension=12)
    with torch.random.fork_rng(devices=[]):
        model.deberta_model = inject_adapter_in_model(
            LoraConfig(**dict(cfg.lora_settings(), r=4, lora_alpha=8)), backbone,
        )
    return model


def toy_frame():
    texts = [('bright lantern', 'lantern'), ('lantern', 'bright lantern'),
             ('quickly arrives', 'arrives quickly'), ('arrives quickly', 'quickly arrives'),
             ('sings', 'waits'), ('bright lantern', 'lantern')]
    frame = pd.DataFrame(texts, columns=['text1', 'text2'])
    # 最后一行模拟真实Train中同文异标：复用计算不能丢掉它的监督。
    frame['label'] = [FULL_LABELS[i] for i in [0, 1, 2, 2, 3, 3]]
    frame['instance_id'] = [f'row_{i}' for i in range(len(frame))]
    frame['pair_id'] = ['group_f', 'group_f', 'group_e', 'group_e', 'group_n', 'conflict_n']
    frame['text1_lang'] = 'en'
    frame['text2_lang'] = 'en'
    return frame


def check_startup(folder):
    hashes = cfg.source_hashes()
    assert hashes and all((ROOT / name).is_file() for name in hashes)
    assert any(Path(name).parts[0] == 'evaluation_functions' for name in hashes)
    assert not any(name.endswith('evaluation_functions.py') for name in hashes)
    model_dir, output_dir = folder / 'startup_model', folder / 'startup_output'
    (model_dir / 'tokenizer').mkdir(parents=True)
    output_dir.mkdir()
    (output_dir / 'training_data.csv').write_text('initialization only', encoding='utf-8')
    assert not cfg.training_records_exist(model_dir, output_dir)
    (model_dir / 'best_model.pth').write_bytes(b'checkpoint marker')
    assert cfg.training_records_exist(model_dir, output_dir)
    print('启动检查通过：评分包源码哈希、未训练目录重跑、已有权重保护。')


def check_toy(folder):
    frame = toy_frame()
    tokenizer = ToyTokenizer()
    dataset = BidirectionalDataset(frame, tokenizer, max_length=16)
    assert len(dataset) == 3
    batch = collate_units([dataset[index] for index in range(len(dataset))])
    assert len(batch['input_ids']) == 6 and len(batch['labels']) == len(frame)
    assert sorted(batch['source_rows'].tolist()) == list(range(len(frame)))
    assert torch.bincount(batch['labels'], minlength=4).tolist() == [1, 1, 2, 2]

    model = toy_model().eval()
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_units)
    logits = predict_logits(model, loader, torch.device('cpu'))
    torch.testing.assert_close(logits[0], logits[1, REVERSE], rtol=0, atol=0)
    torch.testing.assert_close(logits[2], logits[3, REVERSE], rtol=0, atol=0)
    torch.testing.assert_close(logits[0], logits[5], rtol=0, atol=0)

    with torch.inference_mode():
        first, second = torch.randn(4, 12), torch.randn(4, 12)
        original = model.joint_logits(first, second)
        swapped = model.joint_logits(second, first)
        torch.testing.assert_close(original, swapped[:, REVERSE], rtol=0, atol=0)

    # 移除pair_id/标签并改行序，模型仍只根据同一文本输出相同分数。
    shuffled = frame.sample(frac=1, random_state=21).reset_index(drop=True)
    # ToyTokenizer词表依赖整批词汇，两个数据集保持同一词汇集合。
    inference = BidirectionalDataset(shuffled[['text1', 'text2']], tokenizer, 16)
    unlabelled = predict_logits(model, DataLoader(inference, batch_size=3, collate_fn=collate_units), torch.device('cpu'))
    old_indices = frame.set_index('instance_id').index.get_indexer(shuffled.instance_id)
    torch.testing.assert_close(unlabelled, logits[old_indices], rtol=1e-5, atol=1e-6)

    # 检验CE直接约束N与其他三类，不存在只在三个组合内归一化的路径。
    model.train()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    outputs = model(**model_inputs(batch, torch.device('cpu')))
    outputs.retain_grad()
    loss = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))(outputs, batch['labels'])
    loss.backward()
    is_n = batch['labels'].eq(3)
    assert (outputs.grad[is_n, 3] < 0).all()
    assert (outputs.grad[~is_n, 3] > 0).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if 'lora_B.' in name)
    for name, parameter in model.deberta_model.named_parameters():
        if 'lora_A.' not in name and 'lora_B.' not in name:
            assert not parameter.requires_grad and parameter.grad is None
    assert all(torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters())

    # 令LoRA B非零后再检验重载，避免“保存零LoRA也看起来没问题”。
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B.' in name:
                parameter.copy_(torch.linspace(-.01, .01, parameter.numel()).reshape_as(parameter))
    model.eval()
    expected = predict_logits(model, loader, torch.device('cpu'))
    save_weights(model, folder / 'toy.pth')
    restored = toy_model().eval()
    load_weights(restored, folder / 'toy.pth')
    actual = predict_logits(restored, loader, torch.device('cpu'))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    # 走一次真正的官方评分路径，验证行顺序、标签映射和混淆矩阵输出。
    write_train_reference(frame, folder / 'toy_reference.csv')
    summary = score_predictions(expected, frame, folder / 'scores', folder / 'toy_reference.csv')
    assert len(summary) == 2
    assert (folder / 'scores/independent/confusion_matrix.csv').exists()
    print('CPU检查通过：文本复用、原行监督次数、交换关系、元数据隔离、N梯度、冻结范围、保存重载、官方评分。')


def check_real(folder):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=cfg.BASE_REVISION, local_files_only=True,
    )
    frame = toy_frame().iloc[[0, 1, 4]].copy()
    dataset = BidirectionalDataset(frame, tokenizer, cfg.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_units)
    model = build_model(cfg.MODEL_NAME, cfg.BASE_REVISION, cfg.lora_settings(), cfg.DIMENSION, cfg.HEAD_DROPOUT).eval()
    expected = predict_logits(model, loader, torch.device('cpu'))
    assert expected.shape == (3, 4) and torch.isfinite(expected).all()
    torch.testing.assert_close(expected[0], expected[1, REVERSE], rtol=0, atol=0)

    # 同一单条文本也能独立调用，不需要数据集中存在其反向行。
    single = BidirectionalDataset(frame.iloc[[0]][['text1', 'text2']], tokenizer, cfg.MAX_LENGTH)
    single_logits = predict_logits(model, DataLoader(single, batch_size=1, collate_fn=collate_units), torch.device('cpu'))
    torch.testing.assert_close(single_logits[0], expected[0], rtol=1e-4, atol=1e-5)

    model_dir = folder / 'real'
    model_dir.mkdir()
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {'model_name': cfg.MODEL_NAME, 'base_revision': cfg.BASE_REVISION,
              'lora': cfg.lora_settings(), 'dimension': cfg.DIMENSION, 'head_dropout': cfg.HEAD_DROPOUT}
    (model_dir / 'run_config.json').write_text(json.dumps(config), encoding='utf-8')
    save_weights(model, model_dir / 'best_model.pth')
    del model
    restored, _, _ = load_saved_model(model_dir, torch.device('cpu'))
    actual = predict_logits(restored, loader, torch.device('cpu'))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print('真实NLI DeBERTa检查通过：本地加载、单条独立预测、双向输出和专用加载器重载；未训练。')


def main(real=False):
    torch.set_num_threads(4)
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'bidirectional_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_startup(folder)
        check_toy(folder)
        if real:
            check_real(folder)
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('bidirectional_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
