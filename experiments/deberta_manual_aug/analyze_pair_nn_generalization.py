"""只读run10及历史CSV：N稳定错误、Train类比、Train-only盲审材料；不训练。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent / 'outputs/run10_pair_nn'
OUT = RUN_DIR / 'generalization_analysis'
SOURCE_PATH = ROOT / 'experiments/deberta_generalization_audit/outputs/row_ce_frozen_epoch8/source_mapping.csv'
OOF_PATH = ROOT / 'experiments/deberta_lexical_mixed_pair/outputs/oof01_train_only/train_oof_scores.csv'
SHORT = {'FORWARD_ENTAILMENT': 'F', 'BACKWARD_ENTAILMENT': 'B',
         'EQUIVALENCE': 'E', 'NEGATIVE_OTHER': 'N'}


def read_prediction(path):
    frame = pd.read_csv(path, keep_default_na=False)
    frame['gold'] = frame.gold_label.map(SHORT)
    return frame


def pair_key(first, second):
    return tuple(sorted([' '.join(str(first).casefold().split()),
                         ' '.join(str(second).casefold().split())]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_PATH, keep_default_na=False)
    epoch_frames = []
    for epoch in range(1, 11):
        frame = read_prediction(RUN_DIR / f'epoch_{epoch}/after_pair/dev/predictions_with_scores.csv')
        frame['epoch'] = epoch
        epoch_frames.append(frame)
    epochs = pd.concat(epoch_frames, ignore_index=True)
    n = epochs[epochs.gold.eq('N')].copy()
    stability = n.groupby('instance_id').agg(
        text1=('text1', 'first'), text2=('text2', 'first'),
        independent_correct_epochs=('prediction', lambda x: int(x.eq('N').sum())),
        four_correct_epochs=('four_prediction', lambda x: int(x.eq('N').sum())),
    ).reset_index().merge(source[['instance_id', 'source_relations']], on='instance_id', validate='one_to_one')
    stability.to_csv(OUT / 'N_stability.csv', index=False, encoding='utf-8-sig')
    stability[stability.four_correct_epochs.eq(0)].to_csv(
        OUT / 'N_always_wrong.csv', index=False, encoding='utf-8-sig')

    # Train与Dev都固定使用第2轮Pair后的同一个checkpoint，避免把两轮混成泛化证据。
    train = read_prediction(RUN_DIR / 'epoch_2/after_pair/train/predictions_with_scores.csv')
    dev = read_prediction(RUN_DIR / 'epoch_2/after_pair/dev/predictions_with_scores.csv')
    summaries = []
    for split, frame in [('train', train), ('dev', dev)]:
        joined = frame[frame.gold.eq('N')].merge(
            source[['instance_id', 'source_relations']], on='instance_id', how='left', validate='one_to_one')
        joined['source_relations'] = joined.source_relations.fillna('manual_augmentation')
        for relation, group in joined.groupby('source_relations'):
            summaries.append({'split': split, 'source_relations': relation, 'rows': len(group),
                              'independent_correct': int(group.prediction.eq('N').sum()),
                              'four_correct': int(group.four_prediction.eq('N').sum())})
    pd.DataFrame(summaries).to_csv(OUT / 'N_source_summary.csv', index=False, encoding='utf-8-sig')

    official = train[train.instance_id.str.startswith('dico_')].reset_index(drop=True)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(official.text1 + ' || ' + official.text2)
    errors = dev[dev.gold.eq('N') & dev.four_prediction.ne('N')].reset_index(drop=True)
    similarities = (vectorizer.transform(errors.text1 + ' || ' + errors.text2) @ matrix.T).toarray()
    train_keys = [pair_key(row.text1, row.text2) for row in official.itertuples()]
    analogues = []
    for index, row in errors.iterrows():
        different_pair = np.array([key != pair_key(row.text1, row.text2) for key in train_keys])
        for kind, mask in [('N', official.gold.eq('N')), ('nonN', official.gold.ne('N'))]:
            candidates = np.where(mask & official.four_prediction.eq(official.gold) & different_pair)[0]
            ranked = candidates[np.argsort(-similarities[index, candidates])[:2]]
            for rank, selected in enumerate(ranked, 1):
                neighbour = official.iloc[selected]
                analogues.append({'dev_instance_id': row.instance_id, 'dev_text1': row.text1,
                                  'dev_text2': row.text2, 'dev_gold': row.gold,
                                  'dev_prediction': row.four_prediction, 'train_kind': kind, 'rank': rank,
                                  'train_instance_id': neighbour.instance_id,
                                  'train_text1': neighbour.text1, 'train_text2': neighbour.text2,
                                  'train_gold': neighbour.gold, 'train_prediction': neighbour.four_prediction,
                                  'char_similarity': float(similarities[index, selected]),
                                  'scope': 'Surface analogy only; not a controlled semantic edit or training recommendation'})
    pd.DataFrame(analogues).to_csv(OUT / 'N_train_analogues.csv', index=False, encoding='utf-8-sig')

    # 复用已有Train-only OOF分数挑审核题，不重新做OOF训练，也不据此重加权训练。
    # 这一步没有读取Dev文本/错误列表来筛选候选。
    oof = pd.read_csv(OOF_PATH, keep_default_na=False)
    oof['gold'] = oof.label.map(SHORT)
    oof['difficulty'] = 1.0 - oof.gold_probability
    oof = oof.merge(source[['instance_id', 'source_relations']], on='instance_id', validate='one_to_one')
    oof = oof[~oof.label_conflict].sort_values(['difficulty', 'instance_id'], ascending=[False, True])
    oof = oof.drop_duplicates('pair_id')
    masks = [('N_SIMI', oof.gold.eq('N') & oof.source_relations.eq('SIMI'), 8, 2),
             ('N_REL', oof.gold.eq('N') & oof.source_relations.eq('REL'), 4, 2),
             ('E', oof.gold.eq('E'), 6, 2),
             ('FB', oof.gold.isin(['F', 'B']), 6, 2)]
    parts = []
    for name, mask, hard_count, control_count in masks:
        part = oof[mask]
        hard = part[part.prediction.ne(part.gold)].head(hard_count).copy()
        control = part[part.prediction.eq(part.gold)].tail(control_count).copy()
        assert len(hard) == hard_count and len(control) == control_count
        hard['review_role'], control['review_role'] = 'OOF_wrong', 'OOF_correct_control'
        for frame in [hard, control]:
            frame['review_bucket'] = name
            parts.append(frame)
    review = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    review['review_id'] = [f'review_{i + 1:02d}' for i in range(len(review))]
    review[['review_id', 'instance_id', 'pair_id', 'text1', 'text2', 'gold', 'source_relations',
            'prediction', 'difficulty', 'review_role', 'review_bucket']].to_csv(
                OUT / 'train_review_key.csv', index=False, encoding='utf-8-sig')
    blind = review[['review_id', 'text1', 'text2']].copy()
    blind['your_label'] = ''
    blind['reason'] = ''
    blind['needs_context'] = ''
    blind.to_csv(OUT / 'train_review_blind.csv', index=False, encoding='utf-8-sig')
    facts = {'diagnostic_checkpoint': 'run10 epoch2 after_pair',
             'N_always_wrong': int(stability.four_correct_epochs.eq(0).sum()),
             'N_always_correct': int(stability.four_correct_epochs.eq(10).sum()),
             'N_correct_epoch_histogram': {int(k): int(v) for k, v in stability.four_correct_epochs.value_counts().sort_index().items()},
             'N_errors_at_epoch2': len(errors), 'analogy_rows': len(analogues),
             'review_groups': len(review), 'OOF_wrong_review_groups': int(review.review_role.eq('OOF_wrong').sum()),
             'OOF_correct_control_groups': int(review.review_role.eq('OOF_correct_control').sum()),
             'new_training_rows': 0, 'training_performed': False,
             'review_note': 'Original Train examples only; blind labels and reasoning are not yet adjudicated. Not an augmentation dataset.'}
    (OUT / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=2))
    print('只读分析完成：', OUT)


if __name__ == '__main__':
    main()
