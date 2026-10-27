"""数据、泛化、近邻和源语境审计。所有输出只供评估，不写回训练数据。"""

import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

from experiments.deberta_lexical.lexical_features import tokenize_words, number_set, has_negation

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs/row_ce_frozen_epoch8'
LABELS = ['F', 'B', 'E', 'N']
FULL = ['FORWARD_ENTAILMENT', 'BACKWARD_ENTAILMENT', 'EQUIVALENCE', 'NEGATIVE_OTHER']
SHORT = dict(zip(FULL, LABELS))
INVERSE = {'F': 'B', 'B': 'F', 'E': 'E', 'N': 'N'}  # N 反向仅用于审计假设。
PREPS = {'of', 'in', 'on', 'at', 'to', 'from', 'for', 'with', 'by', 'over', 'under',
         'after', 'before', 'into', 'through', 'along', 'across', 'beside', 'near', 'as'}
ARTICLES = {'a', 'an', 'the'}
PRONOUNS = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their', 'this', 'that', 'these', 'those', 'who', 'which'}


def norm(text):
    text = unicodedata.normalize('NFKC', str(text)).replace('’', "'").replace('‘', "'")
    return ' '.join(text.casefold().split())


def key(row):
    return norm(row['text1']), norm(row['text2'])


def canonical(row):
    return tuple(sorted(key(row)))


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def metrics(frame, column='prediction'):
    gold, pred = frame['gold'], frame[column]
    matrix = confusion_matrix(gold, pred, labels=LABELS)
    result = {'rows': len(frame), 'groups': frame['pair_id'].nunique(),
              'correct': int((gold == pred).sum()), 'accuracy': accuracy_score(gold, pred),
              'weighted_f1': f1_score(gold, pred, labels=LABELS, average='weighted', zero_division=0)}
    for i, label in enumerate(LABELS):
        result[f'{label}_correct'] = int(matrix[i, i])
        result[f'{label}_support'] = int(matrix[i].sum())
    return result


def features(frame, train):
    vocab = set(w for text in list(train['text1']) + list(train['text2']) for w in tokenize_words(text))
    phrases = {norm(t) for t in list(train['text1']) + list(train['text2'])}
    train_keys = {key(r) for _, r in train.iterrows()}
    records = []
    for _, row in frame.iterrows():
        a, b = tokenize_words(row['text1']), tokenize_words(row['text2'])
        sa, sb = set(a), set(b)
        ca, cb = sa - ARTICLES, sb - ARTICLES
        jac = len(sa & sb) / max(1, len(sa | sb))
        pp = bool(a and b and a[0] in PREPS and b[0] in PREPS and a[0] != b[0])
        record = {
            'word_jaccard': jac,
            'overlap_bin': '<0.25' if jac < .25 else '0.25-0.5' if jac < .5 else '0.5-0.75' if jac < .75 else '>=0.75',
            'ordered_exact_in_train': key(row) in train_keys,
            'both_phrases_seen_in_train': norm(row['text1']) in phrases and norm(row['text2']) in phrases,
            'either_phrase_seen_in_train': norm(row['text1']) in phrases or norm(row['text2']) in phrases,
            'unseen_word_count': len((sa | sb) - vocab),
            'unseen_words': ' | '.join(sorted((sa | sb) - vocab)),
            'word_set_equal': bool(sa) and sa == sb,
            'article_free_set_equal': bool(ca) and ca == cb,
            'strict_word_subset': bool(sa and sb) and (sa < sb or sb < sa),
            'different_numbers': bool(number_set(a) or number_set(b)) and number_set(a) != number_set(b),
            'number_present': bool(number_set(a) or number_set(b)),
            'negation_differs': has_negation(a) != has_negation(b),
            'preposition_start_differs': pp,
            'pronoun_or_deictic': bool((sa | sb) & PRONOUNS),
            'possessive_fragment': str(row['text1']).lstrip().startswith(("'s", '’s')) or str(row['text2']).lstrip().startswith(("'s", '’s')),
            'short_fragment': min(len(a), len(b)) <= 2,
            'word_count1': len(a), 'word_count2': len(b),
        }
        records.append(record)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def audit_duplicates(train, dev):
    all_rows = pd.concat([train.assign(split='train'), dev.assign(split='dev')], ignore_index=True)
    records = []
    summary = {}
    for name, field in [('exact_ordered', None), ('normalized_ordered', 'normalized'),
                        ('normalized_unordered_inverse', 'canonical'), ('encoded_input', 'input_hash')]:
        groups = defaultdict(list)
        for _, row in all_rows.iterrows():
            if field is None:
                group_key = (row['text1'], row['text2'])
            elif field == 'normalized':
                group_key = key(row)
            elif field == 'canonical':
                group_key = canonical(row)
            else:
                group_key = row[field]
            gold = row['gold']
            if field == 'canonical' and key(row) != canonical(row):
                gold = INVERSE[gold]
            groups[group_key].append((row, gold))
        counts = Counter()
        for group_key, rows in groups.items():
            if len(rows) < 2:
                continue
            conflict = len({gold for row, gold in rows}) > 1
            splits = {row['split'] for row, gold in rows}
            counts['duplicate_keys'] += 1
            counts['conflicting_keys'] += conflict
            counts['cross_split_keys'] += len(splits) > 1
            counts['cross_split_conflicting_keys'] += conflict and len(splits) > 1
            for row, aligned_gold in rows:
                records.append({'kind': name, 'key': json.dumps(group_key, ensure_ascii=False),
                                'conflict': conflict, 'split': row['split'],
                                'instance_id': row['instance_id'], 'pair_id': row['pair_id'],
                                'text1': row['text1'], 'text2': row['text2'], 'gold': row['gold'],
                                'aligned_gold': aligned_gold, 'prediction': row['prediction']})
        # 不同模型可见输入却要求不同答案时，确定性分类器至少错这么多行。
        irreducible = 0
        for rows in groups.values():
            labels = Counter(gold for row, gold in rows)
            irreducible += sum(labels.values()) - max(labels.values())
        summary[name] = dict(counts, minimum_combined_row_errors=irreducible)
    duplicates = pd.DataFrame(records)
    save(duplicates, 'duplicate_groups.csv')
    save(duplicates[duplicates['conflict']], 'label_conflicts.csv')
    # 单独列 cross-split 有序文本重叠，不把所有反向组当成重复污染。
    overlap = train.merge(dev, on=['text1', 'text2'], suffixes=('_train', '_dev'))
    save(overlap, 'train_dev_exact_overlap.csv')
    for split, frame in [('train', train), ('dev', dev)]:
        grouped = frame.groupby('input_hash')['gold'].agg(list)
        summary[split + '_minimum_input_collision_errors'] = int(sum(
            len(golds) - max(Counter(golds).values()) for golds in grouped))
    return summary


def source_audit(train, dev):
    files = sorted((HERE / 'sources').glob('PhrasIS.*.*.all.txt'))
    if len(files) != 4:
        raise FileNotFoundError('请先运行 prepare_sources.py，源语境必须分 headlines / images。')
    rows, slots, malformed = [], defaultdict(lambda: defaultdict(set)), []
    for path in files:
        for line_no, values in enumerate(csv.reader(path.open(encoding='utf-8-sig'), delimiter='\t'), 1):
            if not values:
                continue
            if len(values) != 7:
                raise ValueError(f'{path}:{line_no} 不是七列。')
            score, relation, text1, text2, pos1, pos2, parent = [v.strip() for v in values]
            context_id = path.name + ':' + parent
            item = {'source_file': path.name, 'source_line': line_no, 'source_score': score,
                    'source_relation': relation, 'text1': text1, 'text2': text2,
                    'pos1': pos1, 'pos2': pos2, 'context_id': context_id}
            rows.append(item)
            for side, text, positions in [(0, text1, pos1), (1, text2, pos2)]:
                words, indices = text.split(), positions.split()
                if len(words) != len(indices):
                    malformed.append(dict(item, side=side, problem='token_position_length_mismatch'))
                    continue
                for word, index in zip(words, indices):
                    if int(index) > 0:
                        slots[(context_id, side)][int(index)].add(word)
    contexts = {}
    context_records = []
    for (context_id, side), positions in slots.items():
        maximum = max(positions)
        # [GAP] 明示没有恢复的位置；不臆造缺词，也不保证最后一词之后没有遗漏。
        text = ' '.join('/'.join(sorted(positions[i])) if i in positions else '[GAP]'
                        for i in range(1, maximum + 1))
        record = {'context_id': context_id, 'side': side, 'reconstructed_context': text,
                  'holes': sum(i not in positions for i in range(1, maximum + 1)),
                  'conflicting_positions': sum(len(v) > 1 for v in positions.values())}
        contexts[(context_id, side)] = record
        context_records.append(record)
    save(pd.DataFrame(context_records), 'source_reconstructed_contexts.csv')
    save(pd.DataFrame(malformed), 'source_span_anomalies.csv')
    # 核查分领域文件与现有 all 文件的多重集合相同，避免 source 快照不同。
    source_checks = {}
    for split in ['train', 'test']:
        old_path = ROOT / f'data/phrasis_sources/PhrasIS.{split}.all.txt'
        old = Counter(tuple(v.strip() for v in r) for r in csv.reader(old_path.open(encoding='utf-8-sig'), delimiter='\t') if r)
        new = Counter((r['source_score'], r['source_relation'], r['text1'], r['text2'],
                       r['pos1'], r['pos2'], r['context_id'].rsplit(':', 1)[1])
                      for r in rows if r['source_file'].startswith(f'PhrasIS.{split}.'))
        source_checks[split] = {'old_rows': sum(old.values()), 'domain_rows': sum(new.values()),
                                'only_in_all': sum((old - new).values()),
                                'only_in_domains': sum((new - old).values())}
    index = defaultdict(list)
    inverse = {'FORW': 'BACK', 'BACK': 'FORW'}
    for item in rows:
        for swap in [False, True]:
            a, b = (item['text2'], item['text1']) if swap else (item['text1'], item['text2'])
            relation = inverse.get(item['source_relation'], item['source_relation']) if swap else item['source_relation']
            index[(norm(a), norm(b))].append((item, swap, relation))
    mappings, matches = [], []
    expected = {'FORW': 'F', 'BACK': 'B', 'EQUI': 'E', 'SIMI': 'N', 'REL': 'N', 'OPPO': 'N', 'UNR': 'N'}
    for split, frame in [('train', train), ('dev', dev)]:
        for _, row in frame.iterrows():
            hits = index[key(row)]
            relations = sorted({relation for item, swap, relation in hits})
            expected_labels = sorted({expected[r] for r in relations})
            context_ids = sorted({item['context_id'] for item, swap, relation in hits})
            mappings.append({'split': split, 'instance_id': row['instance_id'],
                             'source_relations': '|'.join(relations), 'source_hit_count': len(hits),
                             'source_relation_ambiguous': len(relations) > 1,
                             'source_expected_labels': '|'.join(expected_labels),
                             'gold_in_source_expected': row['gold'] in expected_labels,
                             'source_context_ids': '|'.join(context_ids)})
            for item, swap, relation in hits:
                c1 = contexts.get((item['context_id'], int(swap)), {})
                c2 = contexts.get((item['context_id'], int(not swap)), {})
                matches.append({'split': split, 'instance_id': row['instance_id'], 'gold': row['gold'],
                                'text1': row['text1'], 'text2': row['text2'],
                                'prediction': row['prediction'], 'correct': row['correct'],
                                **{k: item[k] for k in ['source_file', 'source_line', 'context_id', 'source_score']},
                                'matched_swapped': swap, 'oriented_source_relation': relation,
                                'context1': c1.get('reconstructed_context', ''),
                                'context2': c2.get('reconstructed_context', ''),
                                'context_has_conflicting_slots': bool(c1.get('conflicting_positions', 0) or c2.get('conflicting_positions', 0))})
    mapping = pd.DataFrame(mappings)
    matched = pd.DataFrame(matches)
    save(mapping, 'source_mapping.csv')
    save(matched, 'source_matches_with_context.csv')
    cross = matched.groupby(['split', 'gold', 'oriented_source_relation'])['instance_id'].nunique().reset_index(name='rows')
    save(cross, 'source_relation_task_label_crosstab.csv')
    # 多个匹配均保留；any-overlap 是候选共享语境，unique-overlap 证据更强。
    def all_contexts(subset):
        return {v for text in subset['source_context_ids'] for v in text.split('|') if v}
    train_contexts = all_contexts(mapping[mapping['split'] == 'train'])
    shared = []
    for _, row in mapping[mapping['split'] == 'dev'].iterrows():
        ids = {v for v in row['source_context_ids'].split('|') if v}
        shared.append({'instance_id': row['instance_id'], 'matched_context_count': len(ids),
                       'any_context_shared_with_train': bool(ids & train_contexts),
                       'unique_context_shared_with_train': len(ids) == 1 and bool(ids & train_contexts)})
    save(pd.DataFrame(shared), 'train_dev_source_context_overlap.csv')
    result = {'source_snapshot_check': source_checks,
              'source_rows': len(rows), 'reconstructed_context_sides': len(contexts),
              'contexts_with_conflicting_token_positions': sum(r['conflicting_positions'] > 0 for r in context_records),
              'span_anomaly_sides': len(malformed),
              'unmatched_task_rows': int((mapping['source_hit_count'] == 0).sum()),
              'ambiguous_source_relation_rows': int(mapping['source_relation_ambiguous'].sum()),
              'matched_but_gold_not_in_source_expected_rows': int(((mapping['source_hit_count'] > 0) & ~mapping['gold_in_source_expected']).sum()),
              'dev_any_source_context_shared': sum(r['any_context_shared_with_train'] for r in shared),
              'dev_unique_source_context_shared': sum(r['unique_context_shared_with_train'] for r in shared)}
    return mapping, matched, result


def semantic_graph_audit(train, dev):
    """跨样本传递性候选；必须假设同一短语在不同语境中含义相同，不能自动改金标。"""
    records = []
    summary = {}
    for scope, frame in [('train_only', train), ('train_plus_dev', pd.concat([train, dev], ignore_index=True))]:
        graph = defaultdict(list)
        for _, row in frame.iterrows():
            a, b = key(row)
            if row['gold'] in ['F', 'E']:
                graph[a].append((b, row['instance_id']))
            if row['gold'] in ['B', 'E']:
                graph[b].append((a, row['instance_id']))

        def find_path(start, end):
            pending = deque([start])
            back = {start: None}
            while pending:
                node = pending.popleft()
                if node == end:
                    path, ids = [node], []
                    while back[node] is not None:
                        previous, edge_id = back[node]
                        ids.append(edge_id)
                        path.append(previous)
                        node = previous
                    return list(reversed(path)), list(reversed(ids))
                for target, edge_id in graph[node]:
                    if target not in back:
                        back[target] = (node, edge_id)
                        pending.append(target)
            return None

        for _, row in frame.iterrows():
            a, b = key(row)
            checks = [(a, b), (b, a)] if row['gold'] == 'N' else [(b, a)] if row['gold'] == 'F' else [(a, b)] if row['gold'] == 'B' else []
            for start, end in checks:
                path = find_path(start, end)
                if path is not None:
                    records.append({'scope': scope, 'instance_id': row['instance_id'],
                                    'text1': row['text1'], 'text2': row['text2'], 'gold': row['gold'],
                                    'prediction': row['prediction'],
                                    'conflicting_path': ' -> '.join(path[0]),
                                    'path_instance_ids': '|'.join(path[1]), 'path_edges': len(path[1]),
                                    'status': '语义传递性审核候选，未消除多义词和语境差异'})
        selected = [r for r in records if r['scope'] == scope]
        summary[scope] = {'candidate_rows': len({r['instance_id'] for r in selected}),
                          'candidate_paths': len(selected)}
    save(pd.DataFrame(records), 'semantic_transitivity_review.csv')
    return summary


def historical_comparison(dev):
    original = ROOT / 'experiments/qwen_lora/outputs/teacher_scores/run01/dev_teacher_scores.csv'
    if not original.exists():
        return {'historical_file_available': False}
    frame = pd.read_csv(original, keep_default_na=False)
    frame = frame.set_index('instance_id').loc[dev['instance_id']].reset_index()
    if 'teacher_prediction' in frame:
        frame['prediction'] = frame['teacher_prediction']
    z = frame[[f'logit_{c}' for c in LABELS]].to_numpy(dtype=float)
    exp = np.exp(z - z.max(1, keepdims=True))
    p = exp / exp.sum(1, keepdims=True)
    for i, c in enumerate(LABELS):
        frame[f'prob_{c}'] = p[:, i]
    frame['gold'] = dev['gold'].to_numpy()
    frame['pair_id'] = dev['pair_id'].to_numpy()
    result = {}
    for name, current in [('row_ce_frozen_epoch8', dev), ('original_epoch7', frame)]:
        logits = current[[f'logit_{c}' for c in LABELS]].to_numpy(dtype=float)
        n_probability = current['prob_N'].to_numpy(dtype=float)
        paired = current['gold'].to_numpy() != 'N'
        choice_fbe = np.array(LABELS[:3])[logits[:, :3].argmax(1)]
        result[name] = {**metrics(current),
                        'FBE_correct_conditional_top1_without_N': int((choice_fbe[paired] == current['gold'].to_numpy()[paired]).sum()),
                        'N_roc_auc': roc_auc_score(current['gold'] == 'N', n_probability),
                        'N_average_precision': average_precision_score(current['gold'] == 'N', n_probability),
                        'predicted_N': int((current['prediction'] == 'N').sum()),
                        'nonN_predicted_N': int(((current['gold'] != 'N') & (current['prediction'] == 'N')).sum())}
    result['note'] = '不同训练及选模配方，描述性比较，不是冻结或某个损失的单变量因果实验。'
    return result


def nearest_audit(train, dev):
    # 检索器只在 Train 上拟合。标签只用于事后分类近邻，不参与相似度计算。
    def word_text(frame):
        return [' '.join(['a_' + w for w in tokenize_words(a)] + ['b_' + w for w in tokenize_words(b)])
                for a, b in zip(frame['text1'], frame['text2'])]
    word = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2), sublinear_tf=True)
    train_word = word.fit_transform(word_text(train))
    dev_word = word.transform(word_text(dev))
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
    train_char = char.fit_transform([norm(a) + ' [SEP] ' + norm(b) for a, b in zip(train['text1'], train['text2'])])
    dev_char = char.transform([norm(a) + ' [SEP] ' + norm(b) for a, b in zip(dev['text1'], dev['text2'])])
    similarity = ((dev_word @ train_word.T) * .5 + (dev_char @ train_char.T) * .5).toarray()
    train_canonical = [canonical(r) for _, r in train.iterrows()]
    train_gold = train['gold'].to_numpy()
    train_correct = train['correct'].to_numpy()
    neighbors, minimal = [], []
    for i, row in dev.iterrows():
        # 排除相同或反向的文本对，它们另列在重叠审计中。
        target_canonical = canonical(row)
        eligible = np.asarray([pair != target_canonical for pair in train_canonical])
        order = np.argsort(-similarity[i], kind='stable')
        order = [j for j in order if eligible[j]]
        for rank, j in enumerate(order[:5], 1):
            other = train.iloc[j]
            neighbors.append({'dev_instance_id': row['instance_id'], 'rank': rank, 'similarity': similarity[i, j],
                              'dev_text1': row['text1'], 'dev_text2': row['text2'],
                              'dev_gold': row['gold'], 'dev_prediction': row['prediction'],
                              'train_instance_id': other['instance_id'], 'train_pair_id': other['pair_id'],
                              'train_text1': other['text1'], 'train_text2': other['text2'],
                              'train_gold': other['gold'], 'train_prediction': other['prediction'],
                              'train_correct': other['correct']})
        if row['correct']:
            continue
        # 找“同标签、Train 答对、Dev 答错”的最近自然类比，不能称受控最小改动实验。
        same_gold = [j for j in order if train_gold[j] == row['gold'] and train_correct[j]]
        j = same_gold[0]
        other = train.iloc[j]
        shared = set(key(row)) & set(key(other))
        minimal.append({'instance_id': row['instance_id'], 'gold': row['gold'],
                        'prediction': row['prediction'], 'pair_prediction': row['pair_prediction'],
                        'text1': row['text1'], 'text2': row['text2'],
                        'source_relations': row['source_relations'], 'similarity': similarity[i, j],
                        'exact_shared_phrase': ' | '.join(sorted(shared)),
                        'train_instance_id': other['instance_id'], 'train_text1': other['text1'],
                        'train_text2': other['text2'], 'train_gold': other['gold'],
                        'train_prediction': other['prediction'], 'train_max_probability': other['max_probability'],
                        'dev_max_probability': row['max_probability'], 'dev_unseen_words': row['unseen_words'],
                        'review_status': '自然类比候选，需人工核验，不证明因果'})
    save(pd.DataFrame(neighbors), 'dev_top5_train_neighbours.csv')
    minimal = pd.DataFrame(minimal).sort_values('similarity', ascending=False)
    save(minimal, 'train_correct_dev_wrong_analogues.csv')
    return minimal


def overlap_sensitivity(train, dev):
    rows = [{'subset': 'full_dev', **metrics(dev)}]
    exact = {(r['text1'], r['text2']) for _, r in train.iterrows()}
    normalized = {key(r) for _, r in train.iterrows()}
    for name, keys, make_key in [('exclude_exact_train_overlap', exact, lambda r: (r['text1'], r['text2'])),
                                  ('exclude_normalized_train_overlap', normalized, key)]:
        keep = [make_key(row) not in keys for _, row in dev.iterrows()]
        rows.append({'subset': name, **metrics(dev[keep])})
    save(pd.DataFrame(rows), 'overlap_sensitivity.csv')
    return rows


def slices(frame):
    records = []
    fields = ['gold', 'overlap_bin', 'both_phrases_seen_in_train', 'either_phrase_seen_in_train',
              'strict_word_subset', 'article_free_set_equal', 'different_numbers',
              'preposition_start_differs', 'pronoun_or_deictic', 'possessive_fragment',
              'source_relations', 'short_fragment']
    for field in fields:
        for value, group in frame.groupby(field, dropna=False):
            record = {'field': field, 'value': value, **metrics(group),
                      'pair_correct': int(group['pair_correct'].sum()),
                      'mean_max_probability': float(group['max_probability'].mean())}
            records.append(record)
    for value, group in frame.groupby(frame['unseen_word_count'] > 0):
        records.append({'field': 'has_unseen_words', 'value': bool(value), **metrics(group),
                        'pair_correct': int(group['pair_correct'].sum())})
    return pd.DataFrame(records)


def epoch_audit(dev):
    folder = ROOT / 'experiments/deberta_row_ce_frozen/outputs/run01_freeze6'
    history = pd.read_csv(folder / 'epoch_summary.csv')
    save(history, 'epoch_summary.csv')
    predictions = []
    for epoch in range(1, 11):
        values = pd.read_csv(folder / f'epoch_{epoch}/predictions_with_scores.csv', keep_default_na=False)
        values = values.set_index('instance_id').loc[dev['instance_id']]
        predictions.append(values['prediction'].to_numpy())
    predictions = np.asarray(predictions).T
    rows = dev[['instance_id', 'pair_id', 'text1', 'text2', 'gold']].copy()
    rows['correct_epochs'] = (predictions == dev['gold'].to_numpy()[:, None]).sum(1)
    rows['prediction_changes_across_epochs'] = (predictions[:, 1:] != predictions[:, :-1]).sum(1)
    rows['predictions_epoch1_to10'] = [' '.join(values) for values in predictions]
    save(rows, 'epoch_prediction_stability.csv')
    return {'wrong_all_10_epochs': int((rows['correct_epochs'] == 0).sum()),
            'correct_all_10_epochs': int((rows['correct_epochs'] == 10).sum()),
            'ever_correct': int((rows['correct_epochs'] > 0).sum()),
            'at_least_one_prediction_change': int((rows['prediction_changes_across_epochs'] > 0).sum())}


def probability_audit(frame):
    p = frame[[f'prob_{c}' for c in LABELS]].to_numpy()
    gold_indices = [LABELS.index(g) for g in frame['gold']]
    nll = float(-np.log(np.clip(p[np.arange(len(frame)), gold_indices], 1e-12, 1)).mean())
    bins = []
    for lower in np.arange(0, 1, .1):
        group = frame[(frame['max_probability'] > lower) & (frame['max_probability'] <= lower + .1 + 1e-10)]
        if len(group):
            bins.append({'lower': lower, 'rows': len(group), 'accuracy': float(group['correct'].mean()),
                         'mean_confidence': float(group['max_probability'].mean())})
    result = {'nll': nll, 'mean_confidence': float(frame['max_probability'].mean()),
              'wrong_confidence_above_0p9': int(((~frame['correct']) & (frame['max_probability'] > .9)).sum()),
              'ece_10_equal_width_bins': sum(b['rows'] * abs(b['accuracy'] - b['mean_confidence']) for b in bins) / len(frame),
              'N_roc_auc': roc_auc_score(frame['gold'] == 'N', p[:, 3]),
              'N_average_precision': average_precision_score(frame['gold'] == 'N', p[:, 3])}
    return result, pd.DataFrame(bins)


def main():
    train = pd.read_csv(OUT / 'train_predictions.csv', keep_default_na=False)
    dev = pd.read_csv(OUT / 'dev_predictions.csv', keep_default_na=False)
    summary = {'target': 'frozen lower6, Row CE only, epoch8, selected by independent Dev W-F1',
               'train': metrics(train), 'dev': metrics(dev),
               'train_pair_aware': metrics(train, 'pair_prediction'),
               'dev_pair_aware': metrics(dev, 'pair_prediction')}
    summary['duplicate_audit'] = audit_duplicates(train, dev)
    mapping, matches, summary['source_audit'] = source_audit(train, dev)
    combined = pd.concat([train.assign(split='train'), dev.assign(split='dev')], ignore_index=True)
    combined = combined.merge(mapping, on=['split', 'instance_id'], validate='one_to_one')
    train = features(combined[combined['split'] == 'train'].reset_index(drop=True), train)
    dev = features(combined[combined['split'] == 'dev'].reset_index(drop=True), train)
    save(train, 'train_enriched.csv')
    save(dev, 'dev_enriched.csv')
    save(train[~train['correct']], 'train_errors.csv')
    save(dev[~dev['correct']], 'dev_independent_errors.csv')
    save(dev[~dev['pair_correct']], 'dev_pair_aware_errors.csv')
    save(slices(train), 'train_slices.csv')
    save(slices(dev), 'dev_slices.csv')
    analogues = nearest_audit(train, dev)
    summary['natural_analogues'] = {
        'error_rows': len(analogues), 'same_gold_correct_train_similarity_above_0p45': int((analogues['similarity'] >= .45).sum()),
        'same_gold_correct_train_shares_exact_phrase': int((analogues['exact_shared_phrase'] != '').sum())}
    summary['overlap_sensitivity'] = overlap_sensitivity(train, dev)
    summary['semantic_graph'] = semantic_graph_audit(train, dev)
    summary['historical_comparison'] = historical_comparison(dev)
    summary['epoch_audit'] = epoch_audit(dev)
    for split, frame in [('train', train), ('dev', dev)]:
        summary[split + '_probabilities'], bins = probability_audit(frame)
        save(bins, split + '_calibration_bins.csv')
        summary[split + '_input'] = {'truncated_rows': int(frame['truncated'].sum()),
                                    'maximum_untruncated_tokens': int(frame['untruncated_length'].max()),
                                    'empty_text_rows': int(((frame['text1'].str.strip() == '') | (frame['text2'].str.strip() == '')).sum())}
        for column, suffix in [('prediction', 'independent'), ('pair_prediction', 'pair_aware')]:
            matrix = pd.DataFrame(confusion_matrix(frame['gold'], frame[column], labels=LABELS), index=LABELS, columns=LABELS)
            matrix.to_csv(OUT / f'{split}_{suffix}_confusion.csv', encoding='utf-8-sig', index_label='gold')
    changed = dev['prediction'] != dev['pair_prediction']
    summary['dev_decoding_changes'] = {'changed': int(changed.sum()),
                                      'fixed': int((~dev['correct'] & dev['pair_correct']).sum()),
                                      'broken': int((dev['correct'] & ~dev['pair_correct']).sum()),
                                      'changed_still_wrong': int((changed & ~dev['correct'] & ~dev['pair_correct']).sum())}
    # 供人工审核全部错误；不自动断言 gold 错误，也不自动修改标签。
    review = dev[~dev['correct']].merge(analogues[['instance_id', 'train_text1', 'train_text2', 'similarity']],
                                      on='instance_id', validate='one_to_one')
    for column in ['人工现象分类', '是否需要原句语境', '是否疑似标注冲突', '建议及依据']:
        review[column] = ''
    save(review, 'manual_review_all_errors.csv')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
