"""右键运行：同一DeBERTa读取两个顺序，联合四分类，只使用四类CE。"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

import evaluation_functions
from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional.data import (
    LABELS, BidirectionalDataset, collate_units, model_inputs, write_train_reference,
)
from experiments.deberta_bidirectional.model import build_model, parameter_report, save_weights, load_weights
from experiments.deberta_bidirectional.evaluation import evaluate


RUN_NAME = 'run01_bidirectional_seed7'
SEED = 7
MODEL_NAME = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
BASE_REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
DIMENSION = 128
HEAD_DROPOUT = 0.2

EPOCHS = 20
UNIT_BATCH_SIZE = 8       # 每个文本对两个顺序，因此每批最多16条编码器输入。
EVAL_UNIT_BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.5]
TARGET_INDEPENDENT = 0.87
TARGET_PAIR = 0.94

HERE = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'experiments/deberta_manual_aug/outputs/run06_relation48_staged'
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
MODEL_ROOT = Path('D:/model/SEAMEVAL/deberta_bidirectional')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [HERE / name for name in [
        'train_deberta_bidirectional.py', 'data.py', 'model.py', 'evaluation.py',
    ]]
    paths.append(ROOT / 'experiments/deberta_lexical/evaluation.py')
    # 评分器是一个包，记录实际导入包中的源码，不假设它是单个.py文件。
    package_dir = Path(evaluation_functions.__file__).resolve().parent
    paths.extend(sorted(package_dir.glob('*.py')))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def training_records_exist(model_dir, output_dir):
    # 启动失败仅留下tokenizer/数据快照时可重跑；已有配置或权重则保留原记录。
    paths = [model_dir / 'run_config.json', model_dir / 'best_model.pth',
             output_dir / 'run_config.json', output_dir / 'epoch_summary.csv']
    return any(path.exists() for path in paths)


def read_data():
    # 沿用0.938版本的训练快照：官方3042 + scope200 + relation48。
    # 不再构造新增强，不加载0.938权重续训。
    reference = json.loads((REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    train = pd.read_csv(REFERENCE_DIR / 'training_data.csv', keep_default_na=False)
    digest = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert digest == reference['augmentation']['effective_train_sha256']
    assert len(train) == 3290
    assert file_hash(DEV_PATH) == reference['dev_sha256']
    dev = pd.read_csv(DEV_PATH, keep_default_na=False)
    return train, dev, digest


def lora_settings():
    return {'r': LORA_R, 'lora_alpha': LORA_ALPHA, 'lora_dropout': LORA_DROPOUT,
            'target_modules': ['query_proj', 'value_proj'], 'bias': 'none',
            'init_lora_weights': True, 'use_rslora': False}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    started = time.perf_counter()
    total_loss, total_weight, encoded_views = 0.0, 0.0, 0
    counts = torch.zeros(4, dtype=torch.long)
    for batch in tqdm(loader, desc='双向联合四类CE'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs(batch, device))
        labels = batch['labels'].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        weight = criterion.weight[labels].sum().item()
        total_loss += loss.item() * weight
        total_weight += weight
        encoded_views += len(batch['input_ids'])
        counts += torch.bincount(batch['labels'], minlength=4)

    record = {'row_loss': total_loss / total_weight, 'row_ce_rows': int(counts.sum()),
              'encoded_views': encoded_views, 'optimizer_steps': len(loader),
              'train_seconds': time.perf_counter() - started,
              'learning_rate': optimizer.param_groups[0]['lr']}
    record.update({f'seen_{label}': count for label, count in zip(LABELS, counts.tolist())})
    return record


def select_epoch(summary, epoch):
    independent = float(summary.loc[summary.method.eq('independent'), 'weighted_f1'].iloc[0])
    paired = float(summary.loc[summary.method.eq('pair_aware'), 'weighted_f1'].iloc[0])
    return {'epoch': epoch, 'independent_f1': independent, 'pair_aware_f1': paired,
            'selection_score': min(independent / TARGET_INDEPENDENT, paired / TARGET_PAIR)}


def main(run_name=RUN_NAME, prepare_only=False):
    # 创建正式目录、加载GPU模型前就检查全部源码路径。
    hashes = source_hashes()
    train, dev, train_hash = read_data()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=get_model_cache_dir(), revision=BASE_REVISION, local_files_only=True,
    )
    train_dataset = BidirectionalDataset(train, tokenizer, MAX_LENGTH)
    dev_dataset = BidirectionalDataset(dev, tokenizer, MAX_LENGTH)
    model_dir = MODEL_ROOT / run_name
    output_dir = HERE / 'outputs' / run_name
    print('结构：共享DeBERTa双向编码 + 联合四分类；不使用Pair CE。')
    print('Train / Dev：', len(train), '/', len(dev))
    print('Train标签数量：', train.label.value_counts().to_dict())
    print('Train文本对 / 编码视图：', len(train_dataset), '/', 2 * len(train_dataset))
    print('Dev文本对 / 编码视图：', len(dev_dataset), '/', 2 * len(dev_dataset))
    print('同一文本对只计算一次，但每条原行都计算CE，N不额外加倍。')
    print('r / alpha / dropout：', LORA_R, '/', LORA_ALPHA, '/', LORA_DROPOUT)
    print('模型目录：', model_dir)
    print('结果目录：', output_dir)
    if prepare_only:
        print('数据和分词检查完成；没有加载模型或开始训练。')
        return
    if training_records_exist(model_dir, output_dir):
        raise FileExistsError('此RUN_NAME已有模型或记录，请修改RUN_NAME，保留旧结果。')

    set_seed(SEED)
    device = torch.device('cuda')
    settings = lora_settings()
    model = build_model(MODEL_NAME, BASE_REVISION, settings, DIMENSION, HEAD_DROPOUT).to(device)
    report = pd.DataFrame(parameter_report(model))
    print('\n参数统计：')
    print(report.groupby(['group', 'trainable'], sort=False).parameters.sum().to_string())
    loader = DataLoader(
        train_dataset, batch_size=UNIT_BATCH_SIZE, shuffle=True, collate_fn=collate_units,
        generator=torch.Generator().manual_seed(SEED + 30),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False, collate_fn=collate_units,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=EVAL_UNIT_BATCH_SIZE, shuffle=False, collate_fn=collate_units,
        generator=torch.Generator().manual_seed(SEED + 2),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ROW_WEIGHTS, device=device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

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
        'lora': settings, 'dimension': DIMENSION, 'head_dropout': HEAD_DROPOUT,
        'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'unit_batch_size': UNIT_BATCH_SIZE, 'eval_unit_batch_size': EVAL_UNIT_BATCH_SIZE,
        'max_length': MAX_LENGTH, 'dynamic_padding': True, 'dtype': 'float32',
        'row_weights': ROW_WEIGHTS, 'pair_weight': 0.0,
        'train_rows': len(train), 'train_text_units': len(train_dataset),
        'train_encoded_views_per_epoch': 2 * len(train_dataset),
        'steps_per_epoch': len(loader), 'train_sha256': train_hash, 'dev_sha256': file_hash(DEV_PATH),
        'train_source': str(REFERENCE_DIR / 'training_data.csv'),
        'training_mode': 'joint four-class CE on every original row; no Pair CE; no synthetic N target duplication',
        'architecture': 'shared encoder for both input orders; symmetric E/N and antisymmetric F/B joint head',
        'batching': 'reuse exact unordered text pairs, ignoring pair_id and labels for grouping',
        'selection': 'max min(independent_f1/target_independent, pair_aware_f1/target_pair); earliest on ties',
        'target_independent': TARGET_INDEPENDENT, 'target_pair': TARGET_PAIR,
        'checkpoint_format': 'bidirectional_lora_joint_head_v1',
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
    print('最佳LoRA及联合头：', model_dir / 'best_model.pth')
    print(pd.read_csv(output_dir / 'best_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    main(args.run_name, args.prepare_only)
