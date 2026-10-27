"""右键运行：S16；成员重新训练→各存最佳权重→重新预测→固定融合。"""
import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from project_paths import get_model_cache_dir
from evaluation_functions import score_files
from experiments.deberta_lexical.model import DebertaDiCoModel
from experiments.deberta_lexical.data import PairDataset, build_feature_tensor, validate_groups
from experiments.deberta_lexical.evaluation import get_pair_logits


# todo 0. RUN_NAME区分本次结果；先复用配方匹配的已有模型，缺失的成员才训练。
SCHEME = 'S16'
RUN_NAME = 'run02_pair_select'
MODEL_ROOT = Path('D:/model/SEAMEVAL/ensemble_selected')
FUSION = 'hierarchical'
DIRECTION_MEMBERS = ['M017', 'M030', 'M046', 'M082', 'M087']
N_GATE = 'M027'
DEV_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'
GOLD_PATH = PROJECT_ROOT / 'data/dev/dico_nli_dev_track1_reference.csv'
LABELS = ['F', 'B', 'E', 'N']
FULL_LABELS = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']

# 各成员的训练数据、参数、选模规则写在本文件；不是旧模型路径。
MEMBERS = [{'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.5],
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.5,
  'pair_label_smoothing': 0.05,
  'row_batch_size': 16,
  'pair_batch_size': 16,
  'dev_batch_size': 16,
  'lexical_learning_rate': 0.001,
  'lexical_hidden_size': 32,
  'id': 'M017',
  'name': 'CLS、对齐、词汇特征拼接版',
  'recipe': 'fusion_head',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'base_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
  'use_lexical_features': False,
  'select_by': 'pair_aware'},
 {'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.5],
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.5,
  'pair_label_smoothing': 0.05,
  'row_batch_size': 16,
  'pair_batch_size': 16,
  'dev_batch_size': 16,
  'lexical_learning_rate': 0.001,
  'lexical_hidden_size': 32,
  'use_lexical_features': True,
  'id': 'M027',
  'name': '原0.9306词汇版，seed42',
  'recipe': 'staged',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'base_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
  'select_by': 'pair_aware'},
 {'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.8],
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.5,
  'pair_label_smoothing': 0.05,
  'row_batch_size': 16,
  'pair_batch_size': 16,
  'dev_batch_size': 16,
  'lexical_learning_rate': 0.001,
  'lexical_hidden_size': 32,
  'use_lexical_features': True,
  'id': 'M030',
  'name': '词汇版，N行权重1.8',
  'recipe': 'staged',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'base_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
  'select_by': 'pair_aware'},
 {'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
  'base_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.5],
  'dev_batch_size': 16,
  'group_batch_size': 8,
  'lexical_learning_rate': 0.001,
  'lexical_hidden_size': 32,
  'use_lexical_features': True,
  'consistency_weight': 0.5,
  'id': 'M046',
  'name': '词汇版，Row CE＋反向KL',
  'recipe': 'reversal',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.5,
  'pair_label_smoothing': 0.05,
  'select_by': 'pair_aware'},
 {'model_name': 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.5],
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.5,
  'pair_label_smoothing': 0.05,
  'dev_batch_size': 16,
  'group_batch_size': 8,
  'lexical_learning_rate': 0.001,
  'lexical_hidden_size': 32,
  'use_lexical_features': True,
  'gradient_checkpointing': True,
  'bf16': True,
  'optimizer_foreach': False,
  'id': 'M082',
  'name': 'DeBERTa-large＋对齐＋词汇混合训练',
  'recipe': 'mixed',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'select_by': 'pair_aware',
  'pretrained_path': 'D:\\BaiduNetdiskDownload\\PretrainedModel\\models\\models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli\\snapshots\\b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'},
 {'model_name': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
  'seed': 42,
  'epochs': 10,
  'max_length': 128,
  'learning_rate': 2e-05,
  'weight_decay': 0.01,
  'row_weights': [1.0, 1.0, 1.0, 1.5],
  'pair_weights': [1.0, 1.0, 1.25],
  'pair_weight': 0.0,
  'pair_label_smoothing': 0.05,
  'dev_batch_size': 16,
  'group_batch_size': 8,
  'use_lexical_features': False,
  'id': 'M087',
  'name': '冻结底6层，只训练Row CE',
  'recipe': 'frozen',
  'train_path': 'data/train/dico_nli_train_track1_participant_labeled.csv',
  'expected_train_rows': 3042,
  'base_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
  'select_by': 'pair_aware'}]

# 本方案六个成员统一按Dev pair-aware F1保存最佳；independent仍正常报告。
TRAIN_MEMBERS = MEMBERS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def member_dir(member, run_name):
    # 找到可复用模型时直接指向原目录，不复制大权重。
    return Path(member['_existing_model_dir']) if member.get('_existing_model_dir') else MODEL_ROOT / run_name / member['id']


def recipe_hash(member):
    value = {key: item for key, item in member.items() if not key.startswith('_')}
    value.update(train_sha256=hashlib.sha256((PROJECT_ROOT / member['train_path']).read_bytes()).hexdigest(),
                 dev_sha256=hashlib.sha256(DEV_PATH.read_bytes()).hexdigest())
    if member.get('annotation_path'):
        value['annotation_sha256'] = hashlib.sha256((PROJECT_ROOT / member['annotation_path']).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()



def saved_weight_path(directory, member):
    if member['recipe'] == 'qwen':
        return directory / 'best_adapter/adapter_model.safetensors'
    return directory / 'best_model.pth'


def find_saved_member(member, run_name):
    # outputs保存运行配置，权重可能在该目录，也可能在配置记录的D盘目录。
    # 先查当前RUN_NAME，再查已有RUN_NAME；只复用相同成员和相同训练配方。
    plain_member = {key: value for key, value in member.items() if not key.startswith('_')}
    outputs = Path(__file__).resolve().parent / 'outputs'
    default_dir = MODEL_ROOT / run_name / member['id']
    candidates = [(default_dir, outputs / 'retrained_members' / run_name / member['id'])]
    records = sorted(outputs.rglob('run_config.json')) if outputs.exists() else []
    records += sorted(MODEL_ROOT.glob('*/' + member['id'] + '/run_config.json'))
    for path in records:
        settings = json.loads(path.read_text(encoding='utf-8'))
        if settings.get('id') != member['id']:
            continue
        candidates.append((path.parent, path.parent))
        if settings.get('model_dir'):
            candidates.append((Path(settings['model_dir']), path.parent))
    seen = set()
    for directory, record_dir in candidates:
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        weight = saved_weight_path(directory, member)
        config_path, best_path = directory / 'run_config.json', directory / 'best_metrics.json'
        if not weight.is_file() or weight.stat().st_size == 0 or not config_path.is_file() or not best_path.is_file():
            continue
        settings = json.loads(config_path.read_text(encoding='utf-8'))
        if not all(k in settings for k in plain_member):
            continue
        saved_recipe = {k: settings[k] for k in plain_member}
        # 只允许选模规则不同；数据、seed、结构、损失、轮数和超参数必须一致。
        if any(settings[k] != value for k, value in plain_member.items() if k != 'select_by'):
            continue
        if settings.get('recipe_sha256') != recipe_hash(saved_recipe):
            continue
        best = json.loads(best_path.read_text(encoding='utf-8'))
        reason = '配方和选模规则一致'
        if best.get('selection', settings['select_by']) != member['select_by']:
            # 原来按independent/joint保存，也可能正好保存了同一个pair最佳轮次。
            histories = [record_dir / 'epoch_summary.csv', directory / 'epoch_summary.csv',
                         outputs / 'retrained_members' / directory.parent.name / member['id'] / 'epoch_summary.csv']
            history_path = next((path for path in histories if path.is_file()), None)
            if history_path is None:
                continue
            history = pd.read_csv(history_path)
            if member['select_by'] == 'joint':
                values = history.pivot(index='epoch', columns='method', values='weighted_f1')
                score = np.minimum(values['independent'] / member.get('target_independent', .87),
                                   values['pair_aware'] / member.get('target_pair', .94))
                target_epoch = int(score.sort_index().idxmax())
            else:
                values = history.loc[history.method.eq(member['select_by'])]
                for field in ['stage', 'phase']:
                    if field in values and values[field].eq('after_pair').any():
                        values = values.loc[values[field].eq('after_pair')]
                target_epoch = int(values.sort_values(['weighted_f1', 'epoch'], ascending=[False, True]).iloc[0].epoch)
            if int(best['best_epoch']) != target_epoch:
                continue
            reason = '原保存轮次也正好是当前选模规则的最佳轮次'
        return {'directory': str(directory), 'weight': str(weight), 'best_epoch': best['best_epoch'],
                'saved_selection': best.get('selection', settings['select_by']),
                'requested_selection': member['select_by'], 'reason': reason}
    return None


# todo 1. 官方训练行或原实验的训练数据快照；不读取历史预测CSV。
def read_train(member):
    if member['recipe'] == 'evidence':
        from experiments.deberta_lexical_mixed_pair.reason_data import load_reason_train
        frame = load_reason_train(PROJECT_ROOT / member['train_path'], PROJECT_ROOT / member['annotation_path'])
    else:
        frame = pd.read_csv(PROJECT_ROOT / member['train_path'], keep_default_na=False).reset_index(drop=True)
    if len(frame) != member['expected_train_rows']:
        raise ValueError(member['id'] + '的训练数据行数与配方不一致。')
    if member['recipe'] != 'qwen':
        validate_groups(frame)
    return frame


class DiCoDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.encoded = tokenizer(frame.text1.tolist(), frame.text2.tolist(),
                                 truncation=True, padding='max_length', max_length=max_length,
                                 return_tensors='pt')
        # 特征只读取文本，标签仅用于CE，不作为模型输入。
        self.features = build_feature_tensor(frame)
        self.labels = torch.tensor(frame.label.map(dict(zip(FULL_LABELS, range(4)))).tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.encoded['input_ids'][index], self.encoded['attention_mask'][index],
                self.features[index], self.labels[index])


def tokenizer_source(member):
    return member.get('pretrained_path', member['model_name'])


def load_tokenizer(member, saved_dir=None):
    kwargs = {'local_files_only': True, 'cache_dir': get_model_cache_dir()}
    if saved_dir is None and not member.get('pretrained_path') and member.get('base_revision'):
        kwargs['revision'] = member['base_revision']
    tokenizer = AutoTokenizer.from_pretrained(saved_dir or tokenizer_source(member), **kwargs)
    if member['recipe'] == 'qwen':
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    return tokenizer


# todo 2. 从预训练初始化开始，各成员沿用自己的网络和训练方式。
def build_model(member, tokenizer):
    kwargs = {'cache_dir': get_model_cache_dir(), 'local_files_only': True,
              'dtype': torch.float32, 'use_safetensors': member.get('use_safetensors', True)}
    if not member.get('pretrained_path') and member.get('base_revision'):
        kwargs['revision'] = member['base_revision']
    backbone = AutoModel.from_pretrained(tokenizer_source(member), **kwargs)
    mode = member['recipe']
    if mode == 'fusion_head':
        from experiments.deberta_feature_fusion.model import DebertaDiCoModel as FusionModel
        return FusionModel(backbone, tokenizer, feature_size=20,
                           lexical_hidden=member.get('lexical_hidden_size', 32))
    if mode == 'joint':
        from experiments.deberta_bidirectional.alignment_joint_model import AlignmentJointDeberta
        return AlignmentJointDeberta(backbone, tokenizer, member.get('dimension', 128), member.get('head_dropout', .2))
    if mode == 'evidence':
        from experiments.deberta_lexical_mixed_pair.reason_model import ReasonDebertaModel
        model_class = ReasonDebertaModel
    else:
        model_class = DebertaDiCoModel
    model = model_class(backbone, tokenizer, feature_size=20,
                        lexical_hidden=member.get('lexical_hidden_size', 32),
                        use_lexical_features=member['use_lexical_features'])
    if mode == 'lora':
        from experiments.deberta_manual_aug.relation_lora_model import add_lora
        model = add_lora(model, member['lora'])
    if mode == 'frozen':
        encoder = backbone.encoder
        backbone.embeddings.requires_grad_(False)
        if encoder.relative_attention:
            encoder.rel_embeddings.requires_grad_(False)
        if 'layer_norm' in encoder.norm_rel_ebd:
            encoder.LayerNorm.requires_grad_(False)
        for layer in encoder.layer[:6]:
            layer.requires_grad_(False)
        if encoder.conv is not None:
            encoder.conv.requires_grad_(False)
    if member.get('gradient_checkpointing', False):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def build_optimizer(model, member):
    original, lexical, evidence = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(('lexical_output.', 'lexical_encoder.')):
            lexical.append(parameter)
        elif name.startswith('evidence_head.'):
            evidence.append(parameter)
        else:
            original.append(parameter)
    groups = [{'params': original, 'lr': member['learning_rate']}]
    if lexical:
        groups.append({'params': lexical, 'lr': member.get('lexical_learning_rate', .001)})
    if evidence:
        groups.append({'params': evidence, 'lr': member.get('evidence_learning_rate', .001)})
    return torch.optim.AdamW(groups, weight_decay=member['weight_decay'],
                             foreach=member.get('optimizer_foreach'))


def save_weights(model, path, member):
    if member['recipe'] == 'lora':
        from experiments.deberta_manual_aug.relation_lora_model import save_trainable_weights
        save_trainable_weights(model, path)
    else:
        temporary = path.with_suffix('.tmp')
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
        temporary.replace(path)


def load_weights(model, path, member):
    if member['recipe'] == 'lora':
        from experiments.deberta_manual_aug.relation_lora_model import load_trainable_weights
        load_trainable_weights(model, path)
    else:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)


def forward_rows(model, ids, mask, features, member, device):
    if member.get('dynamic_padding', False):
        length = int(mask.sum(dim=1).max())
        ids, mask = ids[:, :length].contiguous(), mask[:, :length].contiguous()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=member.get('bf16', False) and device.type == 'cuda'):
        logits = model(ids.to(device), mask.to(device), features.to(device))
    return logits.float()


def collate_joint(units):
    from experiments.deberta_bidirectional.data import collate_units
    batch = collate_units(units)
    # M009保留固定128，不额外改变原训练配方。
    batch['input_ids'] = torch.cat([unit['input_ids'] for unit in units])
    batch['attention_mask'] = torch.cat([unit['attention_mask'] for unit in units])
    return batch


@torch.no_grad()
def predict_logits(model, tokenizer, member, dev, device):
    model.eval()
    if member['recipe'] == 'joint':
        from experiments.deberta_bidirectional.data import BidirectionalDataset
        from experiments.deberta_bidirectional.evaluation import predict_logits as joint_predict
        loader = DataLoader(BidirectionalDataset(dev, tokenizer, member['max_length']),
                            batch_size=member.get('eval_unit_batch_size', 8), collate_fn=collate_joint,
                            generator=torch.Generator().manual_seed(member['seed'] + 1))
        return joint_predict(model, loader, device).numpy().astype(np.float64)
    if member['recipe'] == 'qwen':
        from experiments.qwen_lora.reason_sft.reason_evaluation import predict_dev
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            frame = predict_dev(model, tokenizer, dev, member['label_prompt'], batch_size=8)
        return frame[[f'logit_{label}' for label in LABELS]].to_numpy(dtype=np.float64)
    loader = DataLoader(DiCoDataset(dev, tokenizer, member['max_length']),
                        batch_size=member.get('dev_batch_size', 16), shuffle=False,
                        generator=torch.Generator().manual_seed(member['seed'] + 1))
    values = []
    for ids, mask, features, _ in tqdm(loader, desc=member['id'] + '逐行预测'):
        values.append(forward_rows(model, ids, mask, features, member, device).cpu())
    return torch.cat(values).numpy().astype(np.float64)


def evaluate(model, tokenizer, member, dev, device, output_dir):
    # 评估不消耗下一轮shuffle/dropout的随机序列。
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        scores = predict_logits(model, tokenizer, member, dev, device)
        summary = score_predictions(scores, dev, output_dir)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    return summary


def select_score(summary, member):
    scores = summary.set_index('method').weighted_f1
    if member['select_by'] == 'joint':
        return min(scores['independent'] / member.get('target_independent', .87),
                   scores['pair_aware'] / member.get('target_pair', .94))
    return float(scores[member['select_by']])


# todo 3. Row阶段、Pair阶段分别训练；SAM/锚定只对相应成员启用。
def train_rows(model, loader, optimizer, criterion, member, device):
    from experiments.deberta_manual_aug.sam_perturbation import SAMParameters, capture_rng, replay_rng
    model.train()
    total = 0.0
    sam = SAMParameters([p for p in model.parameters() if p.requires_grad], member.get('sam_rho', .05))
    for ids, mask, features, labels in tqdm(loader, desc='Row CE' + (' + SAM' if member['recipe'] == 'sam' else '')):
        optimizer.zero_grad(set_to_none=True)
        states = capture_rng(device) if member['recipe'] == 'sam' else None
        loss = criterion(forward_rows(model, ids, mask, features, member, device), labels.to(device))
        total += loss.item()
        loss.backward()
        if member['recipe'] == 'sam':
            sam.perturb()
            optimizer.zero_grad(set_to_none=True)
            with replay_rng(states, device):
                perturbed = criterion(forward_rows(model, ids, mask, features, member, device), labels.to(device))
                perturbed.backward()
            sam.restore()
        optimizer.step()
    return total / len(loader)


def train_pairs(model, loader, optimizer, criterion, member, device, anchor=None):
    model.train()
    total = 0.0
    for ids0, mask0, features0, ids1, mask1, features1, targets in tqdm(loader, desc='Real Pair CE训练'):
        optimizer.zero_grad(set_to_none=True)
        first = forward_rows(model, ids0, mask0, features0, member, device)
        second = forward_rows(model, ids1, mask1, features1, member, device)
        loss = criterion(get_pair_logits(first, second), targets.to(device))
        (member['pair_weight'] * loss).backward()
        optimizer.step()
        if anchor is not None:
            anchor.apply(optimizer, member.get('anchor_strength', 1000.))
        total += loss.item()
    return total / len(loader)


def train_mixed(model, loader, optimizer, row_criterion, pair_criterion, member, device):
    from experiments.deberta_lexical_mixed_pair.train_deberta_rdrop import compute_joint_loss, compute_rdrop_loss
    from experiments.deberta_reversal_kl.train_deberta_reversal import compute_loss as reversal_loss
    from experiments.deberta_lexical_mixed_pair.reason_model import evidence_loss
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=member['recipe'] + '混合训练'):
        optimizer.zero_grad(set_to_none=True)
        labels = batch['labels'].to(device)
        pair_indices, targets = batch['pair_indices'].to(device), batch['pair_targets'].to(device)
        if member['recipe'] == 'evidence':
            logits, token_logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                                         batch['features'].to(device), return_evidence=True)
        else:
            logits = forward_rows(model, batch['input_ids'], batch['attention_mask'], batch['features'], member, device)
        if member['recipe'] == 'rdrop':
            second = forward_rows(model, batch['input_ids'], batch['attention_mask'], batch['features'], member, device)
            loss, *_ = compute_rdrop_loss(logits, second, labels, pair_indices, targets,
                                          row_criterion, pair_criterion, member['pair_weight'], member.get('rdrop_weight', 1.))
        elif member['recipe'] == 'reversal':
            loss, *_ = reversal_loss(logits, labels, pair_indices, row_criterion, member.get('consistency_weight', .5))
        elif member['recipe'] == 'frozen':
            loss = row_criterion(logits, labels)
        else:
            loss, *_ = compute_joint_loss(logits, labels, pair_indices, targets,
                                          row_criterion, pair_criterion, member['pair_weight'])
            if member['recipe'] == 'evidence':
                loss = loss + member.get('evidence_weight', .1) * evidence_loss(token_logits, batch['evidence_targets'].to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def train_joint(model, loader, optimizer, criterion, device):
    from experiments.deberta_bidirectional.data import model_inputs
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc='双向联合四分类'):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(**model_inputs(batch, device)), batch['labels'].to(device))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def make_train_loader(member, frame, tokenizer):
    mode = member['recipe']
    if mode == 'joint':
        from experiments.deberta_bidirectional.data import BidirectionalDataset
        return DataLoader(BidirectionalDataset(frame, tokenizer, member['max_length']),
                          batch_size=member.get('unit_batch_size', 8), shuffle=True, collate_fn=collate_joint,
                          generator=torch.Generator().manual_seed(member['seed'] + 30))
    if mode == 'evidence':
        from experiments.deberta_lexical_mixed_pair.reason_data import ReasonGroupDataset, collate_reason_groups
        dataset = ReasonGroupDataset(frame, tokenizer, build_feature_tensor(frame), member['max_length'])
        return DataLoader(dataset, batch_size=member.get('group_batch_size', 8), shuffle=True,
                          collate_fn=collate_reason_groups, generator=torch.Generator().manual_seed(member['seed']))
    from experiments.deberta_lexical_mixed_pair.data import GroupDataset, collate_groups
    dataset = GroupDataset(frame, tokenizer, build_feature_tensor(frame), member['max_length'])
    shuffle_seed = member['seed'] + 30 if mode == 'lora' else member['seed']
    return DataLoader(dataset, batch_size=member.get('group_batch_size', 8), shuffle=True,
                      collate_fn=collate_groups, generator=torch.Generator().manual_seed(shuffle_seed))


def train_deberta(member, train, dev, model_dir, results_dir, device):
    set_seed(member['seed'])
    tokenizer = load_tokenizer(member)
    model = build_model(member, tokenizer).to(device)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    model.deberta_model.config.save_pretrained(model_dir / 'backbone_config')
    optimizer = build_optimizer(model, member)
    row_criterion = nn.CrossEntropyLoss(weight=torch.tensor(member['row_weights'], device=device))
    pair_criterion = nn.CrossEntropyLoss(weight=torch.tensor(member['pair_weights'], device=device),
                                          label_smoothing=member['pair_label_smoothing'])
    staged = member['recipe'] in ['staged', 'fusion_head', 'anchor', 'sam']
    if staged:
        row_loader = DataLoader(DiCoDataset(train, tokenizer, member['max_length']),
                                batch_size=member.get('row_batch_size', 16), shuffle=True)
        pair_loader = DataLoader(PairDataset(train, tokenizer, build_feature_tensor(train), member['max_length']),
                                 batch_size=member.get('pair_batch_size', 16), shuffle=True)
    else:
        loader = make_train_loader(member, train, tokenizer)
    best, history = -1., []
    for epoch in range(1, member['epochs'] + 1):
        print(f"\n===== {member['id']} 第 {epoch}/{member['epochs']} 轮 =====")
        if staged:
            row_loss = train_rows(model, row_loader, optimizer, row_criterion, member, device)
            anchor = None
            if member['recipe'] == 'anchor':
                from experiments.deberta_lexical.parameter_anchor import RowParameterAnchor
                anchor = RowParameterAnchor(model)
            pair_loss = train_pairs(model, pair_loader, optimizer, pair_criterion, member, device, anchor)
            del anchor
            print('Row CE / Pair CE：', row_loss, '/', pair_loss)
        elif member['recipe'] == 'joint':
            print('Row CE：', train_joint(model, loader, optimizer, row_criterion, device))
        else:
            print('训练损失：', train_mixed(model, loader, optimizer, row_criterion, pair_criterion, member, device))
        summary = evaluate(model, tokenizer, member, dev, device, results_dir / f'epoch_{epoch}')
        history.extend([dict(epoch=epoch, **row) for row in summary.to_dict('records')])
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        selected = select_score(summary, member)
        if selected > best:
            best = selected
            save_weights(model, model_dir / 'best_model.pth', member)
            write_json(model_dir / 'best_metrics.json', {'best_epoch': epoch, 'selection_score': best,
                       'selection': member['select_by'], 'same_checkpoint_metrics': summary.to_dict('records')})
            print('刷新最佳，只保留一个best_model.pth：', model_dir)
    del model, optimizer, tokenizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def train_n_head(member, train, dev, model_dir, results_dir, device, run_name):
    from experiments.deberta_frozen_n_head.model import FrozenNHeadModel
    from experiments.deberta_frozen_n_head.train_deberta_n_head import train_head
    from torch.utils.data import TensorDataset
    dependency = next(item for item in TRAIN_MEMBERS if item['id'] == member['dependency'])
    base_dir = member_dir(dependency, run_name)
    set_seed(member['seed'])
    tokenizer = load_tokenizer(dependency, base_dir / 'tokenizer')
    relation = build_model(dependency, tokenizer)
    load_weights(relation, base_dir / 'best_model.pth', dependency)
    n_targets = torch.tensor(train.label.eq('NEGATIVE_OTHER').to_numpy(), dtype=torch.float32)
    model = FrozenNHeadModel(relation, member.get('head_hidden_size', 64), n_targets.mean().item()).to(device)
    tokenizer.save_pretrained(model_dir / 'tokenizer')
    relation.deberta_model.config.save_pretrained(model_dir / 'backbone_config')
    def extract(frame):
        loader = DataLoader(DiCoDataset(frame, tokenizer, member['max_length']), batch_size=16, shuffle=False,
                            generator=torch.Generator().manual_seed(member['seed'] + 1))
        features = []
        with torch.no_grad():
            for ids, mask, _, _ in tqdm(loader, desc='冻结主干，提取Train特征'):
                _, values = model.extract_features(ids.to(device), mask.to(device))
                features.append(values.cpu())
        return torch.cat(features)
    loader = DataLoader(TensorDataset(extract(train), n_targets), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(member['seed']))
    optimizer = torch.optim.AdamW(model.n_head.parameters(), lr=member['learning_rate'], weight_decay=member['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(member.get('n_pos_weight', 1.), device=device))
    best, history = -1., []
    for epoch in range(1, member['epochs'] + 1):
        print('N头训练损失：', train_head(model, loader, optimizer, criterion, device))
        summary = evaluate(model, tokenizer, member, dev, device, results_dir / f'epoch_{epoch}')
        history.extend([dict(epoch=epoch, **row) for row in summary.to_dict('records')])
        pd.DataFrame(history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
        selected = select_score(summary, member)
        if selected > best:
            best = selected
            save_weights(model, model_dir / 'best_model.pth', member)
            write_json(model_dir / 'best_metrics.json', {'best_epoch': epoch, 'selection_score': best,
                       'selection': member['select_by'], 'same_checkpoint_metrics': summary.to_dict('records')})
    del model, relation, optimizer, loader, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def train_qwen(member, train, dev, model_dir, results_dir, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import SFTConfig
    from experiments.qwen25_3b_reason.config import build_prompt_templates
    from experiments.qwen_lora.reason_sft.reason_data import build_dataset, ReasonCollator
    from experiments.qwen_lora.reason_sft.reason_trainer import ReasonSFTTrainer
    set_seed(member['seed'])
    tokenizer = load_tokenizer(member)
    member = dict(member)
    member['label_prompt'], reason_prompt = build_prompt_templates(tokenizer)
    write_json(model_dir / 'prompts.json', {'label_prompt': member['label_prompt'], 'reason_prompt': reason_prompt})
    for column in ['use_reason']:
        train[column] = train[column].astype(str).str.casefold().eq('true')
    train_dataset = build_dataset(train, tokenizer, include_reason=True, label_prompt=member['label_prompt'],
                                  reason_prompt=reason_prompt, max_length=member['max_length'])
    dev_dataset = build_dataset(dev, tokenizer, include_reason=False, label_prompt=member['label_prompt'],
                                max_length=member['max_length'])
    model = AutoModelForCausalLM.from_pretrained(member['pretrained_path'], dtype='auto', local_files_only=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**member['lora'], target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    best_dir = model_dir / 'best_adapter'
    class SaveBestAdapter(TrainerCallback):
        def __init__(self):
            self.best, self.history = -1., []
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            epoch = int(round(state.epoch or 0))
            summary = evaluate(model, tokenizer, member, dev, device, results_dir / f'epoch_{epoch}')
            score = select_score(summary, member)
            self.history.extend([dict(epoch=epoch, **row) for row in summary.to_dict('records')])
            pd.DataFrame(self.history).to_csv(results_dir / 'epoch_summary.csv', index=False, encoding='utf-8-sig')
            if score > self.best:
                self.best = score
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                write_json(model_dir / 'best_metrics.json', {'best_epoch': epoch, 'selection_score': score,
                           'selection': 'pair_aware', 'same_checkpoint_metrics': summary.to_dict('records')})
            return control
    args = SFTConfig(output_dir=str(model_dir), learning_rate=member['learning_rate'],
                     num_train_epochs=member['epochs'], per_device_train_batch_size=4,
                     gradient_accumulation_steps=4, per_device_eval_batch_size=4,
                     bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                     completion_only_loss=True, packing=False, padding_free=False, loss_type='chunked_nll',
                     max_length=member['max_length'], dataset_kwargs={'skip_prepare_dataset': True},
                     remove_unused_columns=False, prediction_loss_only=True, eval_strategy='epoch',
                     save_strategy='no', seed=member['seed'], data_seed=member['seed'], logging_steps=20, report_to='none')
    trainer = ReasonSFTTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=dev_dataset,
                               processing_class=tokenizer, data_collator=ReasonCollator(tokenizer.pad_token_id),
                               callbacks=[SaveBestAdapter()], reason_weight=member['reason_weight'])
    # 回调仅在刷新纪录时覆盖best_adapter，不保存每轮checkpoint。
    trainer.train()
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# todo 4. 每个成员独立重训、独立选最佳；先保存，再重新加载做本次真实预测。
def train_member(member, dev, run_name, device):
    existing = find_saved_member(member, run_name)
    if existing is not None:
        member['_existing_model_dir'] = existing['directory']
        print('发现已有模型，跳过训练：', member['id'], existing['directory'])
        print('采用第', existing['best_epoch'], '轮；', existing['reason'])
        return
    member.pop('_existing_model_dir', None)
    model_dir = member_dir(member, run_name)
    results_dir = Path(__file__).resolve().parent / 'outputs/retrained_members' / run_name / member['id']
    completed = model_dir / 'training_complete.json'
    signature = recipe_hash(member)
    if saved_weight_path(model_dir, member).is_file():
        raise ValueError('当前目录有模型，但配方或最佳轮次不匹配；请换RUN_NAME保留它：' + str(model_dir))
    # 只有CSV、配置或旧完成标记而没有实际权重，不跳过训练。
    train = read_train(member)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(member, model_dir=str(model_dir), train_sha256=hashlib.sha256((PROJECT_ROOT / member['train_path']).read_bytes()).hexdigest(),
                    dev_sha256=hashlib.sha256(DEV_PATH.read_bytes()).hexdigest(),
                    initialization='fresh pretrained backbone; no old DiCo checkpoint',
                    training_performed=True, recipe_sha256=signature)
    if member['recipe'] == 'n_head':
        dependency = next(item for item in TRAIN_MEMBERS if item['id'] == member['dependency'])
        settings['initialization'] = 'freeze saved dependency weights, reused or newly trained'
        settings['dependency_model_dir'] = str(member_dir(dependency, run_name))
    write_json(model_dir / 'run_config.json', settings)
    write_json(results_dir / 'run_config.json', settings)
    train.to_csv(results_dir / 'training_data.csv', index=False, encoding='utf-8-sig')
    print('\n===== 重新训练成员 =====', member['id'], member['name'])
    print('数据 / 轮次 / 选模：', len(train), '/', member['epochs'], '/', member['select_by'])
    print('最佳模型保存到：', model_dir)
    if member['recipe'] == 'qwen':
        train_qwen(member, train, dev, model_dir, results_dir, device)
    elif member['recipe'] == 'n_head':
        train_n_head(member, train, dev, model_dir, results_dir, device, run_name)
    else:
        train_deberta(member, train, dev, model_dir, results_dir, device)
    write_json(completed, {'retrained_in_this_pipeline': True, 'recipe_sha256': signature,
                           'training_complete': True, 'old_prediction_csv_used': False})


def predict_saved_member(member, dev, run_name, device):
    existing = find_saved_member(member, run_name)
    if existing is None:
        raise FileNotFoundError('未找到配方匹配的实际模型权重：' + member['id'])
    member['_existing_model_dir'] = existing['directory']
    model_dir = member_dir(member, run_name)
    settings = json.loads((model_dir / 'run_config.json').read_text(encoding='utf-8'))
    member = dict(settings)
    if member['recipe'] == 'qwen':
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        tokenizer = load_tokenizer(member, model_dir / 'best_adapter')
        base = AutoModelForCausalLM.from_pretrained(member['pretrained_path'], dtype='auto', local_files_only=True)
        model = PeftModel.from_pretrained(base, model_dir / 'best_adapter').to(device)
        member['label_prompt'] = json.loads((model_dir / 'prompts.json').read_text(encoding='utf-8'))['label_prompt']
    elif member['recipe'] == 'n_head':
        from experiments.deberta_frozen_n_head.model import build_model as build_n_head
        tokenizer = load_tokenizer(member, model_dir / 'tokenizer')
        config = AutoConfig.from_pretrained(model_dir / 'backbone_config', local_files_only=True)
        model = build_n_head(config, tokenizer, member.get('head_hidden_size', 64))
        load_weights(model, model_dir / 'best_model.pth', member)
        model.to(device)
    else:
        tokenizer = load_tokenizer(member, model_dir / 'tokenizer')
        model = build_model(member, tokenizer)
        load_weights(model, model_dir / 'best_model.pth', member)
        model.to(device)
    scores = predict_logits(model, tokenizer, member, dev, device)
    del model, tokenizer
    if member['recipe'] == 'qwen':
        del base
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return scores



# todo 5. 固定融合和原版三组合解码；单例保留模型判断。

def log_softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def fuse_predictions(scores):
    # N门给N概率；五个方向模型各自只在F/B/E内做softmax。
    n_probability = np.exp(log_softmax(scores[N_GATE]))[:, 3]
    directions = [np.exp(log_softmax(scores[name][:, :3])) for name in DIRECTION_MEMBERS]
    q = np.mean(directions, axis=0)
    probabilities = np.column_stack([(1 - n_probability[:, None]) * q, n_probability])
    return np.log(np.maximum(probabilities, np.finfo(float).tiny))


def pair_aware_decode(logits, frame):
    predictions = logits.argmax(axis=1).copy()
    combinations = [(0, 1), (1, 0), (2, 2)]
    for _, group in frame.groupby('pair_id', sort=False):
        if len(group) != 2:
            continue
        first, second = group.index.tolist()
        scores = [
            logits[first, 0] + logits[second, 1],
            logits[first, 1] + logits[second, 0],
            logits[first, 2] + logits[second, 2],
        ]
        predictions[first], predictions[second] = combinations[int(np.argmax(scores))]
    return predictions


def score_predictions(logits, dev, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    table = dev[['instance_id', 'pair_id', 'text1', 'text2', 'label']].rename(columns={'label': 'gold_label'}).copy()
    for index, label in enumerate(LABELS):
        table[f'logit_{label}'] = logits[:, index]
    summary = []
    gold = dev['label'].map(dict(zip(FULL_LABELS, LABELS)))
    for method, predicted, column in [
        ('independent', logits.argmax(axis=1), 'prediction'),
        ('pair_aware', pair_aware_decode(logits, dev), 'pair_prediction'),
    ]:
        table[column] = np.array(LABELS)[predicted]
        score_dir = output_dir / method
        score_dir.mkdir(parents=True, exist_ok=True)
        path = score_dir / 'official_predictions.csv'
        pd.DataFrame({'instance_id': dev['instance_id'], 'label': np.array(FULL_LABELS)[predicted]}).to_csv(path, index=False)
        report = score_files(GOLD_PATH, path, output_dir=score_dir)
        correct = table[column].eq(gold)
        record = {
            'method': method, 'weighted_f1': report.weighted_f1,
            'accuracy': report.classification.accuracy,
            'soft_cons': report.soft_cons, 'hard_cons': report.hard_cons,
            'N_correct': int((correct & gold.eq('N')).sum()),
            'FBE_correct': int((correct & gold.ne('N')).sum()),
        }
        summary.append(record)
    table.to_csv(output_dir / 'predictions_with_scores.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    return summary


# todo 6. 默认右键运行完整流程；训练交给你运行，模型依次上GPU。
def main(run_name=RUN_NAME, evaluate_only=False, prepare_only=False):
    dev = pd.read_csv(DEV_PATH, keep_default_na=False).reset_index(drop=True)
    if prepare_only:
        for member in TRAIN_MEMBERS:
            train = read_train(member)
            print(member['id'], member['name'], '行数=', len(train), '训练=', member['recipe'],
                  '轮次=', member['epochs'], '选模=', member['select_by'], '保存=', member_dir(member, run_name))
            existing = find_saved_member(member, run_name)
            print('  已有模型，直接复用：' + existing['directory'] if existing else '  未找到匹配权重，需要训练。')
        print('仅检查数据和固定配方；没有训练、加载旧模型或读取旧预测。')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练/预测需要CUDA。')
    device = torch.device('cuda')
    output_dir = Path(__file__).resolve().parent / 'outputs' / SCHEME / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not evaluate_only:
        for member in TRAIN_MEMBERS:
            consumers = [item for item in MEMBERS if item.get('dependency') == member['id']]
            if consumers and all(find_saved_member(item, run_name) is not None for item in consumers):
                print('已有完整N头模型包含冻结底座，跳过前置底座训练：', member['id'])
                continue
            train_member(member, dev, run_name, device)
    scores = {}
    for member in MEMBERS:
        print('\n加载已选定的模型重新预测：', member['id'], member_dir(member, run_name))
        scores[member['id']] = predict_saved_member(member, dev, run_name, device)
        score_predictions(scores[member['id']], dev, output_dir / 'members' / member['id'])
    print('\n===== ' + SCHEME + ' 本次模型融合结果 =====')
    summary = score_predictions(fuse_predictions(scores), dev, output_dir)
    write_json(output_dir / 'run_config.json', {'scheme': SCHEME, 'run_name': run_name,
               'fusion': FUSION, 'members': MEMBERS, 'direction_members': DIRECTION_MEMBERS,
               'n_gate': N_GATE, 'member_model_paths': {m['id']: str(member_dir(m, run_name)) for m in MEMBERS},
               'prediction_source': 'fresh forward from saved model weights, reused or newly trained',
               'historical_prediction_csv_used': False, 'evaluate_only': evaluate_only,
               'forced_singleton_N': False, 'same_checkpoint_metrics': summary.to_dict('records')})
    print('各成员实际模型目录：', {m['id']: str(member_dir(m, run_name)) for m in MEMBERS})
    print('本组预测和融合成绩：', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--evaluate-only', action='store_true', help='只重新加载本次新训练的最佳模型预测')
    parser.add_argument('--prepare-only', action='store_true', help='只检查本组训练数据和配方，不训练')
    args = parser.parse_args()
    main(args.run_name, args.evaluate_only, args.prepare_only)
