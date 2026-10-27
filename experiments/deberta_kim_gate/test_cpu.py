"""CPU 小模型验证；不加载真实主干权重，也不启动正式训练。"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2Model

from experiments.deberta_kim_gate import config as cfg
from experiments.deberta_kim_gate.check_setup import main as check_setup
from experiments.deberta_kim_gate.data import DiCoDataset, PairDataset, build_feature_tensor
from experiments.deberta_kim_gate.model import DebertaDiCoModel
from experiments.deberta_kim_gate.relation_features import (
    build_relation_edges, relation_tensor, word_positions, REVERSE_CHANNELS,
)
from experiments.deberta_kim_gate.wordnet_relations import word_relations
from experiments.deberta_kim_gate.knowledge_gate import KnowledgeGate
from experiments.deberta_kim_gate.evaluation import get_pair_logits
from experiments.deberta_kim_gate.train_deberta_kim_gate import (
    build_optimizer, train_rows, train_pairs, evaluate_without_changing_rng,
    save_best_weights, make_run_config, set_seed,
)
from experiments.deberta_lexical.model import DebertaDiCoModel as OldModel
from experiments.deberta_lexical.data import DiCoDataset as OldDataset, PairDataset as OldPairs
from experiments.deberta_lexical.train_deberta_lexical import build_optimizer as old_optimizer
from experiments.deberta_kim.model import DebertaDiCoModel as OriginalKimModel
from experiments.deberta_kim.train_deberta_kim import (
    build_optimizer as kim_optimizer, train_rows as kim_train_rows, train_pairs as kim_train_pairs,
)


def check_gate():
    gate = KnowledgeGate(4)
    context = torch.zeros(1, 3, 4)
    relations = torch.zeros(1, 3, 3, 4)
    relations[0, 0, 1, 1] = 1
    torch.testing.assert_close(gate(context, relations), relations * 0.5, rtol=0, atol=0)
    with torch.no_grad():
        gate.source.weight[1, 0] = 3
    altered = context.clone(); altered[0, 0, 0] = 1
    assert gate(altered, relations)[0, 0, 1, 1] > gate(context, relations)[0, 0, 1, 1]
    # 同一词对不同通道可学不同门值；没有的关系保持 0。
    assert (gate(altered, relations)[relations == 0] == 0).all()
    for mode, factor in [('fixed', 0.5), ('open', 1.0)]:
        gate.mode = mode
        torch.testing.assert_close(gate(altered, relations), relations * factor, rtol=0, atol=0)
    gate.mode = 'learned'; gate.collect_stats = True
    gate.reset_stats(); gate(altered, relations)
    assert gate.gate_count.tolist() == [0, 1, 0, 0]
    assert not any('gate_sum' in key or 'gate_count' in key for key in gate.state_dict())
    print('门控初始 0.5、上下文响应、无关系不注入、固定/全开对照与统计检查通过。')


def check_relations(tokenizer):
    assert word_relations('car', 'vehicle')[1]
    assert word_relations('vehicle', 'car')[2]
    assert word_relations('hot', 'cold')[3]
    pairs = [
        ('A black car', 'A black vehicle'), ('a dog', 'a canine'),
        ('a hyperactive boy', 'an active child'), ('hot tea', 'cold tea'),
        ('a dog dog', 'a canine'), ('', ''), ('same', 'same'),
    ]
    for first, second in pairs:
        encoded = tokenizer(first, second, return_offsets_mapping=True, padding='max_length', max_length=64)
        edges, evidence = build_relation_edges(first, second, encoded)
        reverse = tokenizer(second, first, return_offsets_mapping=True, padding='max_length', max_length=64)
        _, reverse_evidence = build_relation_edges(second, first, reverse)
        expected = {(e['word2'], e['word1'], tuple(e['relation'][c] for c in REVERSE_CHANNELS)) for e in evidence}
        actual = {(e['word1'], e['word2'], tuple(e['relation'])) for e in reverse_evidence}
        assert expected == actual
        assert torch.isfinite(relation_tensor(edges, 64)).all()
    encoded = tokenizer('a black car', 'a black vehicle', return_offsets_mapping=True)
    edges, evidence = build_relation_edges('a black car', 'a black vehicle', encoded)
    assert {e['word1'] for e in evidence} == {'car'}  # 共有 black 不再重复查词典。
    assert all(r[1] for _, _, r in edges)
    # 真实多子词及原文偏移，不允许只把关系贴到词的首个子词。
    first = 'pneumonoultramicroscopicsilicovolcanoconiosis'
    encoded = tokenizer(first, 'disease', return_offsets_mapping=True)
    assert len(word_positions(first, encoded, 0)[first]) > 1
    truncated = tokenizer(first, 'disease', truncation=True, max_length=8, return_offsets_mapping=True)
    assert first not in word_positions(first, truncated, 0)
    print('词对方向、重复词、多子词、截断、共有词和空文本检查通过。')


def main():
    torch.set_num_threads(2)
    tokenizer, full_datasets = check_setup()
    check_relations(tokenizer)
    check_gate()
    output = cfg.EXPERIMENT_DIR / 'outputs' / f'_cpu_check_{time.time_ns()}'
    output.mkdir(parents=True)
    records = []
    for group, first, second, gold in [
        ('f', 'A black car', 'A black vehicle', (0, 1)),
        ('e', 'a dog', 'a canine', (2, 2)),
    ]:
        for side, label in enumerate(gold):
            a, b = (first, second) if side == 0 else (second, first)
            records.append(dict(instance_id=f'{group}{side}', pair_id=group,
                                text1=a, text2=b, label=cfg.FULL_LABELS[label]))
    records.append(dict(instance_id='n0', pair_id='n', text1='hot tea', text2='cold tea', label=cfg.FULL_LABELS[3]))
    frame = pd.DataFrame(records)
    features = build_feature_tensor(frame)
    dataset = DiCoDataset(frame, tokenizer, features, max_length=32)
    rows = DataLoader(dataset, batch_size=5, shuffle=True)
    pairs = DataLoader(PairDataset(frame, tokenizer, features, max_length=32, row_dataset=dataset), batch_size=2)
    tiny = DebertaV2Config(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
                          relative_attention=True, pos_att_type=['p2c', 'c2p'])
    set_seed(42)
    backbone = DebertaV2Model(tiny)
    state = torch.get_rng_state().clone()
    old = OldModel(copy.deepcopy(backbone), tokenizer, 20)
    old_rng = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    model = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20)
    assert torch.equal(old_rng, torch.get_rng_state())
    torch.set_rng_state(state)
    control = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20, use_kim_features=False)
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, parameter in old.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(parameter, control.state_dict()[name], rtol=0, atol=0)
    assert model.relation_projection.weight.numel() == 512
    assert model.relation_projection.weight.count_nonzero() == 0
    assert sum(p.numel() for p in model.lexical_output.parameters()) == 804
    assert sum(p.numel() for p in model.knowledge_gate.parameters()) == 1044
    batch = next(iter(rows))
    ids, masks, lexical, relations, labels = batch
    old.eval(); model.eval(); control.eval()
    with torch.no_grad():
        expected = old(ids, masks, lexical)
        torch.testing.assert_close(expected, model(ids, masks, lexical, relations), rtol=0, atol=0)
        torch.testing.assert_close(expected, control(ids, masks, lexical, relations), rtol=0, atol=0)

    # 关闭知识模块的两阶段训练必须与原代码保持参数完全一致。
    old_rows = DataLoader(OldDataset(frame, tokenizer, features, 32), batch_size=5, shuffle=True)
    old_pairs = DataLoader(OldPairs(frame, tokenizer, features, 32), batch_size=2)
    old_opt, control_opt = old_optimizer(old), build_optimizer(control)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    pair_ce = nn.CrossEntropyLoss(weight=torch.tensor(cfg.PAIR_WEIGHTS), label_smoothing=cfg.PAIR_LABEL_SMOOTHING)
    for epoch in range(2):
        set_seed(100 + epoch)
        old.train()
        for x, mask, f, y in old_rows:
            old_opt.zero_grad(); ce(old(x, mask, f), y).backward(); old_opt.step()
        for x0, m0, f0, x1, m1, f1, y in old_pairs:
            old_opt.zero_grad()
            loss = cfg.PAIR_WEIGHT * pair_ce(get_pair_logits(old(x0, m0, f0), old(x1, m1, f1)), y)
            loss.backward(); old_opt.step()
        set_seed(100 + epoch)
        control.train()
        train_rows(control, rows, control_opt, ce, torch.device('cpu'))
        train_pairs(control, pairs, control_opt, pair_ce, torch.device('cpu'))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
    print('原参数/RNG/初始分数一致；禁用 KIM 的两轮 Row→Pair 训练与原 Lexical 完全一致。')

    # 全开门控必须等价于旧 KIM，而不只是初始分数相同。
    open_model = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20, gate_mode='open')
    old_kim = OriginalKimModel(copy.deepcopy(backbone), tokenizer, 20)
    common_state = {key: value for key, value in open_model.state_dict().items()
                    if not key.startswith('knowledge_gate.')}
    old_kim.load_state_dict(common_state, strict=True)
    old_kim_opt, open_opt = kim_optimizer(old_kim), build_optimizer(open_model)
    for epoch in range(2):
        set_seed(200 + epoch); old_kim.train()
        kim_train_rows(old_kim, rows, old_kim_opt, ce, torch.device('cpu'))
        kim_train_pairs(old_kim, pairs, old_kim_opt, pair_ce, torch.device('cpu'))
        set_seed(200 + epoch); open_model.train()
        train_rows(open_model, rows, open_opt, ce, torch.device('cpu'))
        train_pairs(open_model, pairs, open_opt, pair_ce, torch.device('cpu'))
        for key, value in old_kim.state_dict().items():
            torch.testing.assert_close(value, open_model.state_dict()[key], rtol=0, atol=0)
    print('全开门控两轮 Row→Pair 训练与原无门控 KIM 完全一致。')

    # 原 match_output 零初始化，首步知识分支梯度为 0 属正常；热身后必须有梯度。
    optimizer = build_optimizer(model)
    all_params = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(all_params) == len(set(all_params)) == len(list(model.parameters()))
    assert optimizer.param_groups[2]['lr'] == cfg.RELATION_LEARNING_RATE
    assert optimizer.param_groups[3]['lr'] == cfg.GATE_LEARNING_RATE
    assert {id(p) for p in optimizer.param_groups[2]['params']} == {
        id(p) for p in model.relation_projection.parameters()
    }
    assert {id(p) for p in optimizer.param_groups[3]['params']} == {
        id(p) for p in model.knowledge_gate.parameters()
    }
    model.train()
    train_rows(model, rows, optimizer, ce, torch.device('cpu'))
    optimizer.zero_grad()
    ce(model(ids, masks, lexical, relations), labels).backward()
    assert model.relation_projection.weight.grad.abs().sum() > 0
    optimizer.step()
    # 第二次更新后知识投影已不为 0，门控才开始获得有效梯度。
    optimizer.zero_grad()
    ce(model(ids, masks, lexical, relations), labels).backward()
    assert model.knowledge_gate.source.weight.grad.abs().sum() > 0
    assert model.knowledge_gate.target.weight.grad.abs().sum() > 0
    assert model.knowledge_gate.relation.weight.grad.abs().sum() > 0
    optimizer.step()
    pair = next(iter(pairs))
    optimizer.zero_grad()
    a = model(*pair[:4]); b = model(*pair[4:8])
    pair_ce(get_pair_logits(a, b), pair[-1]).backward()
    assert model.relation_projection.weight.grad.abs().sum() > 0
    assert model.knowledge_gate.source.weight.grad.abs().sum() > 0
    train_pairs(model, pairs, optimizer, pair_ce, torch.device('cpu'))
    model.eval()
    with torch.inference_mode():
        real = model(ids, masks, lexical, relations)
        no_knowledge = model(ids, masks, lexical, torch.zeros_like(relations))
        assert not torch.equal(real, no_knowledge)
        # padding/特殊 token 的伪关系不能影响分数。
        noisy = relations.clone(); noisy[:, 0, :, :] = 1; noisy[:, :, 0, :] = 1
        torch.testing.assert_close(real, model(ids, masks, lexical, noisy), rtol=0, atol=0)
    print('Row / Pair 对新增投影均有非零梯度，局部关系影响输出，特殊位置正确屏蔽。')

    model.train()
    before = torch.get_rng_state().clone()
    dev_loader = DataLoader(full_datasets[1], batch_size=16)
    summary = evaluate_without_changing_rng(model, dev_loader, full_datasets[1].frame,
                                           torch.device('cpu'), output / 'dev')
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert summary.loc[summary.method == 'pair_aware', 'soft_cons'].iloc[0] == 1
    stats = pd.read_csv(output / 'dev/gate_statistics.csv')
    assert stats['directed_subword_edges'].sum() > 0
    assert stats['mean_gate'].dropna().between(0, 1).all()
    save_best_weights(model, output / 'tiny_best.pth')
    restored = DebertaDiCoModel(copy.deepcopy(backbone), tokenizer, 20)
    restored.load_state_dict(torch.load(output / 'tiny_best.pth', weights_only=True), strict=True)
    restored.eval(); model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, masks, lexical, relations), restored(ids, masks, lexical, relations), rtol=0, atol=0)
    config = make_run_config(True, 3042, 660, 1282)
    (output / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('660 条官方评分、模式/RNG 恢复、保存/严格重载检查通过。')
    print('全部 CPU 检查通过：', output)


if __name__ == '__main__':
    main()
