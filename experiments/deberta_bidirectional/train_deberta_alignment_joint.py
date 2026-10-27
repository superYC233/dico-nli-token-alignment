"""右键运行：原0.938训练配方，保留Alignment特征，换双向联合头。"""
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

import evaluation_functions
from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional.data import (
    BidirectionalDataset, collate_units, write_train_reference,
)
from experiments.deberta_bidirectional.alignment_joint_model import (
    build_model, parameter_report, save_weights, load_weights,
)
from experiments.deberta_bidirectional.evaluation import evaluate
from experiments.deberta_bidirectional.train_deberta_bidirectional import (
    read_data, file_hash, training_records_exist, train_epoch,
)


RUN_NAME = 'run02_original_alignment_joint_seed42'
SEED = 42
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
MAX_LENGTH = 128
UNIT_BATCH_SIZE = 8       # 每个文本对AB/BA两个顺序，每批16条编码器输入。
EVAL_UNIT_BATCH_SIZE = 8
DIMENSION = 128
HEAD_DROPOUT = 0.2

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_bidirectional')


def collate_original_units(units):
    batch = collate_units(units)
    # 复用原行索引处理，但恢复0.938版固定128长度，不叠加动态PAD优化。
    batch['input_ids'] = torch.cat([unit['input_ids'] for unit in units])
    batch['attention_mask'] = torch.cat([unit['attention_mask'] for unit in units])
    return batch


def source_hashes():
    names = ['train_deberta_alignment_joint.py', 'alignment_joint_model.py',
             'train_deberta_bidirectional.py', 'model.py', 'data.py', 'evaluation.py']
    paths = [HERE / name for name in names]
    paths.extend([ROOT / 'experiments/deberta_lexical/model.py',
                  ROOT / 'experiments/deberta_lexical/evaluation.py'])
    paths.extend(sorted(Path(evaluation_functions.__file__).resolve().parent.glob('*.py')))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def select_epoch(summary, epoch):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    # 按用户“其他沿用0.938”的要求，恢复原版的结构F1选模。
    # 两项成绩仍记录在同一条记录里，不拼接不同轮次的峰值。
    return {'epoch': epoch, 'independent_f1': independent,
            'pair_aware_f1': paired, 'selection_score': paired}


def main(run_name=RUN_NAME, prepare_only=False):
    hashes = source_hashes()
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    train_dataset = BidirectionalDataset(train, tokenizer, MAX_LENGTH)
    dev_dataset = BidirectionalDataset(dev, tokenizer, MAX_LENGTH)
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    print('结构：原CLS + Token Alignment特征 → 双向联合四分类。')
    print('训练：全量微调，无LoRA，无Pair CE，不分Row/Pair两个阶段。')
    print('seed / 学习率 / 轮次：', SEED, '/', LEARNING_RATE, '/', EPOCHS)
    print('Train / Dev：', len(train), '/', len(dev))
    print('Train标签数量：', train.label.value_counts().to_dict())
    print('Train文本对 / 编码视图：', len(train_dataset), '/', 2 * len(train_dataset))
    print('每批文本对 / 编码输入：', UNIT_BATCH_SIZE, '/', 2 * UNIT_BATCH_SIZE)
    print('每条原始行一次四类CE；N不额外增加监督次数。')
    print('选模：与0.938相同，按Dev pair-aware F1最高轮保存，同时记录独立F1。')
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据、源码路径和分词检查完成；没有加载模型或训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    model = build_model(MODEL_NAME, BASE_REVISION, tokenizer, DIMENSION, HEAD_DROPOUT).to(device)
    report = pd.DataFrame(parameter_report(model))
    print('\n参数统计：')
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())
    loader = DataLoader(
        train_dataset, batch_size=UNIT_BATCH_SIZE, shuffle=True, collate_fn=collate_original_units,
        generator=torch.Generator().manual_seed(SEED + 30),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False, collate_fn=collate_original_units,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False, collate_fn=collate_original_units,
        generator=torch.Generator().manual_seed(SEED + 2),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    for name in ['data_config.json', 'overlap_report.json']:
        (output_dir / name).write_bytes((REFERENCE_DIR / name).read_bytes())
    write_train_reference(train, output_dir / 'train_reference.csv')
    report.to_csv(output_dir / 'parameters.csv', index=False, encoding='utf-8-sig')
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    config = {
        'run_name': run_name, 'seed': SEED, 'model_name': MODEL_NAME, 'base_revision': BASE_REVISION,
        'use_lora': False, 'use_alignment': True, 'full_finetuning': True,
        'dimension': DIMENSION, 'head_dropout': HEAD_DROPOUT,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'unit_batch_size': UNIT_BATCH_SIZE, 'eval_unit_batch_size': EVAL_UNIT_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'dynamic_padding': False, 'dtype': 'float32',
        'row_weights': ROW_WEIGHTS, 'pair_weight': 0.0,
        'train_rows': len(train), 'train_text_units': len(train_dataset),
        'train_encoded_views_per_epoch': 2 * len(train_dataset), 'steps_per_epoch': len(loader),
        'trainable_parameters': int(report.loc[report.trainable, 'parameters'].sum()),
        'total_parameters': int(report.parameters.sum()),
        'train_sha256': train_hash, 'dev_sha256': file_hash(DEV_PATH),
        'train_source': str(REFERENCE_DIR / 'training_data.csv'),
        'training_mode': 'four-class CE on original rows, no separate Pair CE stage',
        'architecture': 'original CLS and Token Alignment features in both orders, then joint four-class head',
        'selection': 'best Dev pair-aware weighted F1; earliest epoch on ties, as original 0.938',
        'checkpoint_format': 'full_encoder_alignment_joint_v1',
        'reference_run': str(REFERENCE_DIR),
        'comparison_note': 'Restores original data, full FT, seed42, lr2e-5, 10 epochs, alignment features, row weights and PA selection. Joint features/head and unified CE necessarily change batching, forward exposure and optimization schedule; not a one-line Pair CE replacement.',
        'source_sha256': hashes,
    }
    for folder in [model_dir, output_dir]:
        (folder / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    history, selections = [], []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== 第 {epoch}/{EPOCHS} 轮 =====')
        losses = train_epoch(model, loader, optimizer, criterion, device)
        assert losses['row_ce_rows'] == len(train)
        print('本轮损失及曝光：', losses)
        summary = evaluate(model, dev_loader, dev, device, output_dir / f'epoch_{epoch}', GOLD_PATH)
        for row in summary.to_dict('records'):
            row.update(epoch=epoch, **losses)
            history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        selected = select_epoch(summary, epoch)
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
    evaluate(model, dev_loader, dev, device, output_dir / 'best/dev', GOLD_PATH)
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
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
