"""只读取已有Dev分数，比较多模型融合；不加载真实模型、不训练。"""
import argparse
import csv
import hashlib
import heapq
import io
import json
import os
import sys
from itertools import combinations, islice
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.deberta_lexical_mixed_pair.analyze_fusion import (
    decode, log_softmax, metrics, safe_log,
)


RUN_NAME = 'run01_all_existing'
OUTPUT = Path(__file__).resolve().parent / 'outputs' / RUN_NAME
LABELS = ['F', 'B', 'E', 'N']
FULL = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
DEV_PATH = ROOT / 'data/dev/dico_nli_dev_track1_participant_labeled.csv'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def load_prediction(path, dev):
    payload = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(payload), keep_default_na=False)
    columns = [f'logit_{label}' for label in LABELS]
    if not set(columns + ['instance_id']).issubset(frame.columns):
        raise ValueError('no four-label logits')
    if len(frame) != len(dev) or not frame.instance_id.is_unique or set(frame.instance_id) != set(dev.instance_id):
        raise ValueError('not the exact official 660 Dev IDs')
    frame = frame.set_index('instance_id').loc[dev.instance_id]
    if 'pair_id' in frame and frame.pair_id.tolist() != dev.pair_id.tolist():
        raise ValueError('pair_id differs')
    for key in ['gold_label', 'label']:
        if key in frame and frame[key].isin(FULL).all() and frame[key].tolist() != dev.label.tolist():
            raise ValueError('gold differs')
    for key in ['text1', 'text2']:
        if key in frame and frame[key].tolist() != dev[key].tolist():
            raise ValueError('input text differs')
    raw = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('non-finite logits')
    lp = log_softmax(raw)
    mismatch = None
    if 'prediction' in frame:
        saved = frame.prediction.replace(dict(zip(FULL, LABELS))).to_numpy()
        if np.isin(saved, LABELS).all():
            mismatch = int((saved != np.array(LABELS)[raw.argmax(1)]).sum())
    return lp, hashlib.sha256(payload).hexdigest(), mismatch


def checkpoint_paths(folder, settings, metadata):
    candidates = []
    for source in [metadata, settings]:
        for key in ['checkpoint', 'best_checkpoint', 'best_model_checkpoint', 'best_adapter', 'model_dir', 'output_dir']:
            value = source.get(key)
            if isinstance(value, str):
                candidates.append(Path(value))
    parts = folder.relative_to(ROOT / 'experiments').parts
    experiment = parts[0]
    relative = Path(*parts[2:])
    for prefix in [Path('D:/model/SEAMEVAL'), ROOT / 'model']:
        candidates.append(prefix / experiment / relative)
    if experiment == 'qwen25_3b_reason':
        candidates.append(Path('D:/LLM/train_model/DiCo-NLI/qwen2p5_3b_reason') / relative)
    if experiment == 'qwen_lora' and relative.parts[0] == 'qwen3_0p6b_reason':
        candidates.append(Path('D:/LLM/train_model/DiCo-NLI') / relative)
    found = []
    for path in candidates:
        if path.is_file() and path.suffix in ['.pth', '.pt', '.safetensors']:
            found.append(str(path))
        elif path.is_dir():
            for name in ['best_model.pth', 'best_model.pt', 'best.pt', 'best_adapter/adapter_model.safetensors', 'adapter_model.safetensors']:
                if (path / name).is_file():
                    found.append(str(path / name))
    return sorted(set(found))


def inventory(dev, gold, pairs):
    folders = []
    for base, directories, files in os.walk(ROOT / 'experiments'):
        directories[:] = [d for d in directories if d not in ['__pycache__', '.git', 'model_ensemble'] and not d.startswith('dico_manual_check')]
        folder = Path(base)
        if 'outputs' in folder.parts and 'best_summary.csv' in files:
            folders.append(folder)
    members, skipped = [], []
    for folder in sorted(folders):
        summary = pd.read_csv(folder / 'best_summary.csv')
        if 'split' in summary:
            summary = summary[summary.split.eq('dev')]
        if not {'method', 'weighted_f1'}.issubset(summary.columns):
            continue
        targets = {method: summary.loc[summary.method.eq(method), 'weighted_f1'].iloc[0]
                   for method in ['independent', 'pair_aware'] if summary.method.eq(method).any()}
        if 'pair_aware' not in targets:
            continue
        candidates = []
        for path in folder.rglob('*.csv'):
            if not ('prediction' in path.name or 'logits' in path.name):
                continue
            if any(x in ['train', 'train_diagnostic', 'heldout_families', 'comparison_with_original', 'comparison_with_nli_0p9306'] for x in path.relative_to(folder).parts):
                continue
            try:
                lp, sha, mismatch = load_prediction(path, dev)
                row = metrics(gold, decode(lp, pairs, 'independent'), pairs)
                pair = metrics(gold, decode(lp, pairs, 'pair_aware'), pairs)
                errors = [abs(pair['weighted_f1'] - targets['pair_aware'])]
                if 'independent' in targets:
                    errors.append(abs(row['weighted_f1'] - targets['independent']))
                if max(errors) < 1e-9:
                    priority = (0 if 'best' in path.relative_to(folder).parts else 1, len(path.parts), str(path))
                    candidates.append((priority, path, lp, sha, mismatch, row, pair))
            except (ValueError, KeyError, pd.errors.ParserError, OSError):
                continue
        if not candidates:
            skipped.append({'run': str(folder.relative_to(ROOT)), 'reason': 'No four-score CSV matches saved best_summary on exact Dev.'})
            continue
        _, path, lp, sha, mismatch, row, pair = min(candidates, key=lambda x: x[0])
        settings = read_json(folder / 'run_config.json')
        metadata = read_json(folder / 'best_metrics.json')
        checkpoints = checkpoint_paths(folder, settings, metadata)
        count = settings.get('train_rows')
        if count is None and (folder / 'training_data.csv').is_file():
            count = len(pd.read_csv(folder / 'training_data.csv', usecols=['instance_id']))
        members.append({
            'name': str(folder.relative_to(ROOT / 'experiments')).replace('\\', '/').replace('/outputs/', '/'),
            'prediction_file': str(path), 'prediction_sha256': sha,
            'model_name': settings.get('model_name', settings.get('model_path', 'unknown')),
            'train_rows': count, 'use_lexical_features': settings.get('use_lexical_features'),
            'external_auxiliary': any('paws' in str(v).lower() for k, v in settings.items() if 'paws' in k.lower()),
            'checkpoint_candidates_existing': checkpoints, 'checkpoint_found': bool(checkpoints),
            'selection': settings.get('selection'), 'saved_prediction_argmax_differences': mismatch,
            'independent_f1': row['weighted_f1'], 'pair_f1': pair['weighted_f1'],
            'N_correct': row['N_correct'], 'pair_FBE_correct': pair['FBE_correct'],
            'log_probability': lp,
        })
    # 旧Qwen结果没有best_summary，以下为历史明确指定版本，不重新挑epoch。
    explicit = [
        ('qwen_C_r16', 'experiments/qwen_lora/outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv',
         'Qwen3-0.6B-Base', 3517, 'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_abc/run04_r16/C/best_adapter/adapter_model.safetensors'),
        ('qwen_sixclass', 'experiments/qwen_lora/outputs/qwen3_0p6b_sixclass/run01/C/predictions_with_scores.csv',
         'Qwen3-0.6B-Base six-class readout', 3517, 'D:/LLM/train_model/DiCo-NLI/qwen3_0p6b_sixclass/run01/C/best_adapter/adapter_model.safetensors'),
        ('deberta_SWA_6_10', 'experiments/deberta_stage_average/outputs/run01_late_average/averaged_final/dev/predictions_with_scores.csv',
         'DeBERTa NLI + Alignment; fixed CPU average epochs 6-10', 3290, 'D:/model/SEAMEVAL/deberta_stage_average/run01_late_average/averaged_model.pth'),
    ]
    for name, filename, model_name, count, checkpoint in explicit:
        path = ROOT / filename
        lp, sha, mismatch = load_prediction(path, dev)
        row = metrics(gold, decode(lp, pairs, 'independent'), pairs)
        pair = metrics(gold, decode(lp, pairs, 'pair_aware'), pairs)
        members.append({'name': name, 'prediction_file': str(path), 'prediction_sha256': sha,
                        'model_name': model_name, 'train_rows': count, 'use_lexical_features': False,
                        'external_auxiliary': False, 'checkpoint_found': Path(checkpoint).is_file(),
                        'checkpoint_candidates_existing': [checkpoint] if Path(checkpoint).is_file() else [],
                        'selection': 'Historical fixed C+r16 version', 'saved_prediction_argmax_differences': mismatch,
                        'independent_f1': row['weighted_f1'], 'pair_f1': pair['weighted_f1'],
                        'N_correct': row['N_correct'], 'pair_FBE_correct': pair['FBE_correct'], 'log_probability': lp})
    # 相同预测分数只计一个模型，避免重复缓存给某个成员隐式加权。
    unique = {}
    for member in sorted(members, key=lambda x: (not x['checkpoint_found'], x['name'])):
        digest = hashlib.sha256(np.round(member['log_probability'], 7).tobytes()).hexdigest()
        if digest in unique:
            unique[digest]['aliases'].append(member['name'])
        else:
            member['aliases'] = []
            unique[digest] = member
    members = sorted(unique.values(), key=lambda x: x['name'])
    for index, member in enumerate(members, 1):
        member['id'] = f'M{index:03d}'
    return members, skipped


def save_inventory(members, skipped):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [{k: v for k, v in m.items() if k != 'log_probability'} for m in members]
    (OUTPUT / 'members.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / 'members.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(skipped).to_csv(OUTPUT / 'excluded_runs.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(OUTPUT / 'aligned_scores.npz', log_probability=np.stack([m['log_probability'] for m in members]))
    print('Unique members:', len(members), 'checkpoint paths found:', sum(m['checkpoint_found'] for m in members), 'excluded:', len(skipped), flush=True)
    print(table.sort_values('pair_f1', ascending=False)[['id', 'name', 'independent_f1', 'pair_f1', 'checkpoint_found']].head(12).to_string(index=False), flush=True)


def batch_metrics(gold, predicted):
    size = len(predicted)
    index = np.arange(size)[:, None] * 16 + 4 * gold[None] + predicted
    matrix = np.bincount(index.ravel(), minlength=size * 16).reshape(size, 4, 4)
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    support, counts = matrix.sum(2), matrix.sum(1)
    denominator = support + counts
    f1 = np.divide(2.0 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0)
    return ((f1 * support).sum(1) / len(gold), diagonal[:, 3], diagonal[:, :3].sum(1), matrix[:, :3, 3].sum(1))


class EnsembleSearch:
    def __init__(self, members, gold, pairs):
        self.members, self.gold, self.pairs = members, gold, pairs
        self.lp = np.stack([m['log_probability'] for m in members])
        self.prob = np.exp(self.lp)
        self.direction = np.exp(log_softmax(self.lp.reshape(-1, 4)[:, :3])).reshape(len(members), len(gold), 3)
        self.available = np.array([m['checkpoint_found'] for m in members])
        self.official = np.array([m['train_rows'] == 3042 and not m['external_auxiliary'] for m in members])
        self.gates = {
            'hier_0938': next(i for i, m in enumerate(members) if m['name'] == 'deberta_manual_aug/run06_relation48_staged'),
            'hier_0930': next(i for i, m in enumerate(members) if m['name'] == 'deberta_lexical/run02_branch_lr/lexical_seed_42'),
        }
        self.heaps = {}
        self.serial = 0
        self.fields = ['mode', 'direction_count', 'model_count', 'members', 'gate', 'independent_f1', 'pair_f1',
                       'ind_N', 'pair_N', 'ind_FBE', 'pair_FBE', 'ind_non_N_to_N', 'all_checkpoints_found', 'official3042_only']

    def combine(self, indices, mode):
        if mode == 'probability':
            return safe_log(self.prob[indices].mean(1))
        if mode == 'log_probability':
            return self.lp[indices].mean(1)
        q = self.direction[indices].mean(1)
        g = self.prob[self.gates[mode], :, 3]
        return safe_log(np.concatenate([(1 - g[None, :, None]) * q,
                                       np.broadcast_to(g[None, :, None], (*q.shape[:2], 1))], axis=2))

    def score(self, logits):
        single = logits.argmax(2)
        paired = single.copy()
        first, second = self.pairs[:, 0], self.pairs[:, 1]
        states = np.array([[0, 1], [1, 0], [2, 2]])
        group_scores = logits[:, first[:, None], states[None, :, 0]] + logits[:, second[:, None], states[None, :, 1]]
        chosen = states[group_scores.argmax(2)]
        paired[:, first], paired[:, second] = chosen[:, :, 0], chosen[:, :, 1]
        return batch_metrics(self.gold, single), batch_metrics(self.gold, paired)

    def retain(self, row, combo):
        constraints = ['all']
        if row['all_checkpoints_found']:
            constraints.append('weights_found')
        if row['official3042_only']:
            constraints.append('official3042')
            if row['all_checkpoints_found']:
                constraints.append('official3042_weights_found')
        for constraint in constraints:
            for objective in ['independent', 'pair', 'balanced']:
                value = row['independent_f1'] if objective == 'independent' else row['pair_f1']
                if objective == 'balanced':
                    value = min(row['independent_f1'] / 0.87, row['pair_f1'] / 0.94)
                key = (row['mode'], len(combo), constraint, objective)
                queue = self.heaps.setdefault(key, [])
                item = (value, self.serial, combo, row)
                self.serial += 1
                if len(queue) < 8:
                    heapq.heappush(queue, item)
                elif value > queue[0][0]:
                    heapq.heapreplace(queue, item)

    def evaluate(self, combinations_iterator, mode, writer):
        evaluated = 0
        while True:
            chunk = list(islice(combinations_iterator, 64))
            if not chunk:
                break
            indices = np.array(chunk)
            logits = self.combine(indices, mode)
            ind, pair = self.score(logits)
            gate = self.gates.get(mode)
            for j, combo in enumerate(chunk):
                total = sorted(set(combo) | ({gate} if gate is not None else set()))
                row = {
                    'mode': mode, 'direction_count': len(combo), 'model_count': len(total),
                    'members': '+'.join(self.members[i]['id'] for i in combo),
                    'gate': '' if gate is None else self.members[gate]['id'],
                    'independent_f1': float(ind[0][j]), 'pair_f1': float(pair[0][j]),
                    'ind_N': int(ind[1][j]), 'pair_N': int(pair[1][j]),
                    'ind_FBE': int(ind[2][j]), 'pair_FBE': int(pair[2][j]),
                    'ind_non_N_to_N': int(ind[3][j]),
                    'all_checkpoints_found': bool(self.available[total].all()),
                    'official3042_only': bool(self.official[total].all()),
                }
                writer.writerow(row)
                self.retain(row, combo)
            evaluated += len(chunk)
        return evaluated

    def run(self):
        counts = {}
        with (OUTPUT / 'all_combinations.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            for mode in ['probability', 'log_probability', 'hier_0938', 'hier_0930']:
                for size in [1, 2, 3]:
                    count = self.evaluate(iter(combinations(range(len(self.members)), size)), mode, writer)
                    counts[f'{mode}_{size}'] = count
                    print('Completed:', mode, 'size', size, 'recipes', count, flush=True)
                for size in [4, 5, 6]:
                    seeds = set()
                    for key, queue in self.heaps.items():
                        if key[0] == mode and key[1] == size - 1:
                            seeds.update(item[2] for item in queue)
                    extensions = {tuple(sorted((*combo, extra))) for combo in seeds
                                  for extra in range(len(self.members)) if extra not in combo}
                    count = self.evaluate(iter(sorted(extensions)), mode, writer)
                    counts[f'{mode}_{size}'] = count
                    print('Completed:', mode, 'size', size, 'beam extensions', count, flush=True)
                stream.flush()
        selected = []
        for (mode, size, constraint, objective), queue in self.heaps.items():
            best = max(queue, key=lambda x: (x[0], -x[1]))
            selected.append({'constraint': constraint, 'objective': objective, **best[3]})
        pd.DataFrame(selected).to_csv(OUTPUT / 'best_by_size.csv', index=False, encoding='utf-8-sig')
        (OUTPUT / 'search_scope.json').write_text(json.dumps({
            'evaluated': counts, 'member_count': len(self.members), 'exhaustive_sizes': [1, 2, 3],
            'sizes_4_to_6': 'Beam extension of top8 per objective and availability/data constraint; not exhaustive.',
            'probability': 'Arithmetic mean of per-model four-label softmax probabilities.',
            'log_probability': 'Mean log-probabilities (geometric pooling).',
            'hierarchical': 'Fixed N gate plus equal mean of per-model F/B/E-conditional probabilities.',
            'label_order': LABELS, 'forced_singleton_N': False, 'trained_or_loaded_models': False,
            'selection_warning': 'Large search on repeatedly reused Dev; exploratory optimistic scores, not held-out validation.',
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Search done. Results:', OUTPUT, flush=True)


def main(inventory_only=False):
    if (OUTPUT / 'search_scope.json').exists():
        raise FileExistsError('本RUN_NAME已有完整融合结果，请改RUN_NAME后再运行，保留旧结果。')
    dev = pd.read_csv(DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev.label.map(dict(zip(FULL, range(4)))).to_numpy()
    pairs = np.array([g.index.to_numpy() for _, g in dev.groupby('pair_id', sort=False) if len(g) == 2])
    members, skipped = inventory(dev, gold, pairs)
    save_inventory(members, skipped)
    if inventory_only:
        return
    EnsembleSearch(members, gold, pairs).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    main(args.inventory_only)
