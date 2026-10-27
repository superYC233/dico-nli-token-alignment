"""右键运行：原全量Alignment联合模型，加入少量PAWS二元等价监督。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional import train_deberta_alignment_joint as base_cfg
from experiments.deberta_bidirectional.data import BidirectionalDataset, write_train_reference
from experiments.deberta_bidirectional.alignment_joint_model import (
    build_model, parameter_report, save_weights, load_weights,
)
from experiments.deberta_bidirectional.evaluation import evaluate
from experiments.deberta_bidirectional.prepare_paws_data import prepare, PREPARED_DIR
from experiments.deberta_bidirectional.paws_auxiliary import PawsDataset, train_epoch


RUN_NAME = 'run03_paws_aux_seed42'
SEED = 42
MODEL_NAME = base_cfg.MODEL_NAME
BASE_REVISION = base_cfg.BASE_REVISION
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
MAX_LENGTH = 128
UNIT_BATCH_SIZE = 8
EVAL_UNIT_BATCH_SIZE = 8
DIMENSION = 128
HEAD_DROPOUT = 0.2

PAWS_WEIGHT = 0.1
PAWS_EVERY = 4             # 每4批DiCo，把一批PAWS梯度合入第4次更新。
PAWS_UNIT_BATCH_SIZE = 8
HERE = Path(__file__).resolve().parent
MODEL_ROOT = base_cfg.MODEL_ROOT


def source_hashes():
    hashes = base_cfg.source_hashes()
    for name in ['prepare_paws_data.py', 'paws_auxiliary.py', 'train_deberta_paws.py']:
        path = HERE / name
        hashes[str(path.relative_to(ROOT))] = base_cfg.file_hash(path)
    return hashes


def main(run_name=RUN_NAME, prepare_only=False, prepared_dir=PREPARED_DIR):
    hashes = source_hashes()
    train, dev, train_hash = base_cfg.read_data()
    prepared_dir = Path(prepared_dir)
    paws, manifest = prepare(prepared_dir)
    assert manifest['request']['model_name'] == MODEL_NAME
    assert manifest['request']['base_revision'] == BASE_REVISION
    assert manifest['request']['max_length'] == MAX_LENGTH
    assert PAWS_EVERY > 0 and PAWS_WEIGHT >= 0
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    train_dataset = BidirectionalDataset(train, tokenizer, MAX_LENGTH)
    dev_dataset = BidirectionalDataset(dev, tokenizer, MAX_LENGTH)
    paws_dataset = PawsDataset(paws, tokenizer, MAX_LENGTH)
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    print('\n模型：全量DeBERTa + 原Token Alignment特征 + 双向联合四分类。')
    print('DiCo Train / Dev：', len(train), '/', len(dev))
    print('PAWS训练池：', len(paws), '；二元标签数量：', paws.paws_label.value_counts().to_dict())
    print('PAWS只监督等价/非等价；非等价不映射成N。')
    print('PAWS权重 / 间隔：', PAWS_WEIGHT, '/', PAWS_EVERY)
    steps = (len(train_dataset) + UNIT_BATCH_SIZE - 1) // UNIT_BATCH_SIZE
    auxiliary_batches = steps // PAWS_EVERY if PAWS_WEIGHT > 0 else 0
    print('每轮DiCo更新：', steps, '；其中加入PAWS的更新：', auxiliary_batches)
    print('每轮约使用PAWS样本：', auxiliary_batches * PAWS_UNIT_BATCH_SIZE,
          f'；每轮重新打乱{len(paws)}条样本池，不要求每轮遍历全池。')
    print('seed / 学习率 / 轮次：', SEED, '/', LEARNING_RATE, '/', EPOCHS)
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('准备和分词完成，没有加载模型或开始训练。')
        return
    if base_cfg.training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    # 与run02相同初始化，重新训练；不加载Dev选出的run02权重续训。
    model = build_model(MODEL_NAME, BASE_REVISION, tokenizer, DIMENSION, HEAD_DROPOUT).to(device)
    report = pd.DataFrame(parameter_report(model))
    print('\n参数统计：')
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())
    loader = DataLoader(
        train_dataset, batch_size=UNIT_BATCH_SIZE, shuffle=True,
        collate_fn=base_cfg.collate_original_units, generator=torch.Generator().manual_seed(SEED + 30),
    )
    paws_loader = DataLoader(
        paws_dataset, batch_size=PAWS_UNIT_BATCH_SIZE, shuffle=True,
        collate_fn=base_cfg.collate_original_units, generator=torch.Generator().manual_seed(SEED + 100),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False,
        collate_fn=base_cfg.collate_original_units, generator=torch.Generator().manual_seed(SEED + 1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False,
        collate_fn=base_cfg.collate_original_units, generator=torch.Generator().manual_seed(SEED + 2),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    write_train_reference(train, output_dir / 'train_reference.csv')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((base_cfg.REFERENCE_DIR / name).read_bytes())
    for source, destination in [('paws_train.csv', 'paws_train.csv'),
                                ('preparation.json', 'paws_preparation.json'),
                                ('overlap_report.json', 'paws_overlap_report.json')]:
        (output_dir / destination).write_bytes((prepared_dir / source).read_bytes())
    report.to_csv(output_dir / 'parameters.csv', index=False, encoding='utf-8-sig')
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'seed': SEED, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION,
        'use_lora': False, 'use_alignment': True, 'full_finetuning': True,
        'dimension': DIMENSION, 'head_dropout': HEAD_DROPOUT,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'row_weights': ROW_WEIGHTS, 'pair_weight': 0.0, 'max_length': MAX_LENGTH,
        'unit_batch_size': UNIT_BATCH_SIZE, 'eval_unit_batch_size': EVAL_UNIT_BATCH_SIZE,
        'dynamic_padding': False, 'dtype': 'float32',
        'train_rows': len(train), 'train_sha256': train_hash, 'dev_sha256': base_cfg.file_hash(base_cfg.DEV_PATH),
        'trainable_parameters': int(report.loc[report.trainable, 'parameters'].sum()),
        'steps_per_epoch': len(loader), 'paws_weight': PAWS_WEIGHT, 'paws_every': PAWS_EVERY,
        'paws_unit_batch_size': PAWS_UNIT_BATCH_SIZE, 'paws_pool_rows': len(paws),
        'paws_selected_sha256': manifest['selected_sha256'],
        'paws_manifest_sha256': base_cfg.file_hash(prepared_dir / 'preparation.json'),
        'paws_expected_batches_per_epoch': auxiliary_batches,
        'paws_supervision': 'binary E versus logsumexp(F,B,N); non-paraphrases do not receive N labels',
        'training_mode': f'DiCo CE every update; accumulate {PAWS_WEIGHT} PAWS binary CE every {PAWS_EVERY} updates, then one optimizer step',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties, same as run02',
        'checkpoint_format': 'full_encoder_alignment_joint_v1',
        'baseline_run': 'run02_original_alignment_joint_seed42', 'source_sha256': hashes,
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    history, selections, exposure_history = [], [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        losses, exposure = train_epoch(
            model, loader, paws_loader, optimizer, criterion, device, PAWS_WEIGHT, PAWS_EVERY,
        )
        assert losses['row_ce_rows'] == len(train)
        assert losses['paws_batches'] == auxiliary_batches
        print('本轮损失及曝光：', losses)
        for row in exposure:
            row['epoch'] = epoch
        exposure_history.extend(exposure)
        pd.DataFrame(exposure_history).to_csv(output_dir / 'paws_exposure.csv', index=False, encoding='utf-8-sig')
        summary = evaluate(model, dev_loader, dev, device, output_dir / f'epoch_{epoch}', base_cfg.GOLD_PATH)
        for row in summary.to_dict('records'):
            row.update(epoch=epoch, **losses)
            history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        selected = base_cfg.select_epoch(summary, epoch)
        selections.append(selected)
        pd.DataFrame(selections).to_csv(output_dir / 'selection_history.csv', index=False, encoding='utf-8-sig')
        if selected['selection_score'] > best_score:
            best_score, best_epoch = selected['selection_score'], epoch
            save_weights(model, model_dir / 'best_model.pth')
            for folder in [model_dir, output_dir]:
                (folder / 'best_metrics.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
            summary.to_csv(output_dir / 'best_summary.csv', index=False, encoding='utf-8-sig')
            print('刷新同轮最佳：', selected)

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    load_weights(model, model_dir / 'best_model.pth')
    evaluate(model, dev_loader, dev, device, output_dir / 'best/dev', base_cfg.GOLD_PATH)
    evaluate(model, train_loader, train, device, output_dir / 'best/train', output_dir / 'train_reference.csv')
    saved = pd.read_csv(output_dir / f'epoch_{best_epoch}/predictions_with_scores.csv')
    restored = pd.read_csv(output_dir / 'best/dev/predictions_with_scores.csv')
    assert saved.instance_id.tolist() == restored.instance_id.tolist()
    differences = {name: int(saved[name].ne(restored[name]).sum())
                   for name in ['prediction', 'pair_prediction']}
    assert all(value == 0 for value in differences.values())
    (output_dir / 'reload_check.json').write_text(json.dumps(differences, indent=2), encoding='utf-8')
    print('\n最佳轮次：', best_epoch)
    print('最佳完整模型：', model_dir / 'best_model.pth')
    print(pd.read_csv(output_dir / 'best_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED_DIR)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only, args.prepared_dir)
