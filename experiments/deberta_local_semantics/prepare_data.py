"""给既有Train行补局部标注；不生成新的训练文本。"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from experiments.deberta_local_semantics import config as cfg
from experiments.deberta_lexical.data import validate_groups


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span_offsets(text, span):
    # 明确检查标注短语的位置，防止把man匹配到woman内部。
    occurrences = list(re.finditer(r'(?<!\w)' + re.escape(span) + r'(?!\w)', text))
    if len(occurrences) != 1:
        raise ValueError(f'局部短语须唯一出现：{span!r}，原文：{text!r}')
    return occurrences[0].start(), occurrences[0].end()


def expand_annotations(official, source):
    records = []
    for row in source.itertuples(index=False):
        pair_id = f'dico_phrasis_{int(row.pair_number):07d}'
        group = official[official.pair_id.eq(pair_id)]
        candidates = group[
            group.text1.str.contains(re.escape(row.span1), regex=True) &
            group.text2.str.contains(re.escape(row.span2), regex=True)
        ]
        if len(candidates) != 1:
            raise ValueError(f'{pair_id}无法唯一确定源行，请检查两个span。')
        first = candidates.iloc[0]
        views = [(first, row.span1, row.span2, row.relation)]
        if len(group) == 2:
            second = group[group.instance_id.ne(first.instance_id)].iloc[0]
            assert second.text1 == first.text2 and second.text2 == first.text1
            views.append((second, row.span2, row.span1, cfg.INVERSE[row.relation]))
        for original, span1, span2, relation in views:
            start1, end1 = span_offsets(original.text1, span1)
            start2, end2 = span_offsets(original.text2, span2)
            records.append({
                'instance_id': original.instance_id, 'pair_id': pair_id,
                'split': row.split, 'text1': original.text1, 'text2': original.text2,
                'span1': span1, 'span1_start': start1, 'span1_end': end1,
                'span2': span2, 'span2_start': start2, 'span2_end': end2,
                'local_relation': relation, 'global_label': original.label,
                'family': row.family, 'reason': row.reason,
                'reading_assumption': row.reading_assumption,
                'annotation_status': 'assistant_annotated_draft_not_independently_adjudicated',
            })
    annotations = pd.DataFrame(records)
    assert annotations.instance_id.is_unique
    assert set(annotations.local_relation) <= set(cfg.LABELS)
    train_ids = set(annotations.loc[annotations.split.eq('train'), 'pair_id'])
    probe_ids = set(annotations.loc[annotations.split.eq('diagnostic'), 'pair_id'])
    assert train_ids.isdisjoint(probe_ids)
    return annotations


def normalized_pairs(frame):
    normalize = lambda value: ' '.join(str(value).casefold().split())
    return {tuple(sorted((normalize(a), normalize(b))))
            for a, b in zip(frame.text1, frame.text2)}


def prepare():
    train = pd.read_csv(cfg.BASE_TRAIN_PATH, keep_default_na=False)
    official = pd.read_csv(cfg.OFFICIAL_TRAIN_PATH, keep_default_na=False)
    reference = json.loads((cfg.REFERENCE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    canonical = hashlib.sha256(train.to_csv(index=False).encode('utf-8')).hexdigest()
    assert canonical == reference['augmentation']['effective_train_sha256']
    assert len(train) == 3290 and len(official) == 3042
    validate_groups(train)
    source = pd.read_csv(cfg.ANNOTATION_PATH, keep_default_na=False)
    annotations = expand_annotations(official, source)
    local_train = annotations[annotations.split.eq('train')].copy()
    diagnostic = annotations[annotations.split.eq('diagnostic')].copy()
    assert len(local_train) == 64 and len(diagnostic) == 16
    assert local_train.local_relation.value_counts().to_dict() == dict.fromkeys(cfg.LABELS, 16)
    # 不允许把增强行、Dev行或新造的反向N混进局部监督。
    assert set(annotations.instance_id) <= set(official.instance_id)
    assert set(annotations.instance_id) <= set(train.instance_id)

    # Dev只用于审计既有文本的交集，不用于选标注样本或确定局部标签。
    dev_text = pd.read_csv(cfg.DEV_PATH, usecols=['instance_id', 'text1', 'text2'], keep_default_na=False)
    assert set(annotations.instance_id).isdisjoint(dev_text.instance_id)
    shared_phrases = set(annotations.text1) | set(annotations.text2)
    dev_phrases = set(dev_text.text1) | set(dev_text.text2)
    assert not normalized_pairs(annotations) & normalized_pairs(dev_text), '新增局部监督不使用与Dev重复的完整文本对。'
    report = {
        'train_rows': len(train), 'official_train_rows': len(official),
        'new_text_rows': 0, 'local_train_rows': len(local_train),
        'local_diagnostic_rows': len(diagnostic),
        'local_train_relations': local_train.local_relation.value_counts().to_dict(),
        'local_diagnostic_relations': diagnostic.local_relation.value_counts().to_dict(),
        'whole_input_pair_overlap_with_dev': len(normalized_pairs(annotations) & normalized_pairs(dev_text)),
        'existing_whole_phrase_overlap_with_dev': len(shared_phrases & dev_phrases),
        'effective_train_sha256': canonical,
        'reference_train_sha256': reference['augmentation']['effective_train_sha256'],
        'train_equals_original_0p938_snapshot': True,
        'source_sha256': {str(path.relative_to(cfg.ROOT)): file_hash(path) for path in [
            cfg.BASE_TRAIN_PATH, cfg.OFFICIAL_TRAIN_PATH, cfg.ANNOTATION_PATH,
        ]},
        'annotation_source': 'Only existing official Train pairs; explicit span/relation annotations with reading assumptions. Dev text pairs are a rejection-only blacklist; Dev labels are not read for annotation.',
        'annotation_status': 'Assistant annotations, not independently adjudicated; not automatic semantic proofs.',
        'diagnostic_scope': 'These 16 inputs still receive ordinary Row CE because they are existing Train rows. Only their local annotations are held out. This is not an unseen-text generalization test.',
        'development': 'Method direction informed by earlier Dev analysis. Annotation examples selected by inspecting Train, without reading Dev labels or model predictions.',
        'isolation_scope': 'No new inputs are added. Existing Train may share vocabulary or a whole phrase with Dev. No claim of word-level isolation or hidden-Test non-overlap.',
    }
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    # 保留原snapshot的实际字节；这里不重新构造或打乱3290行。
    (cfg.PREPARED_DIR / 'training_data.csv').write_bytes(cfg.BASE_TRAIN_PATH.read_bytes())
    local_train.to_csv(cfg.PREPARED_DIR / 'local_train_annotations.csv', index=False, encoding='utf-8-sig')
    diagnostic.to_csv(cfg.PREPARED_DIR / 'local_diagnostic_annotations.csv', index=False, encoding='utf-8-sig')
    annotations.to_csv(cfg.PREPARED_DIR / 'all_local_annotations.csv', index=False, encoding='utf-8-sig')
    (cfg.PREPARED_DIR / 'annotation_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('整体Train：', len(train), '；新增文本：0')
    print('局部辅助监督：', len(local_train), '；局部标注留出诊断：', len(diagnostic))
    print('局部训练标签：', report['local_train_relations'])
    print('原0.938训练数据哈希一致：', canonical)
    return train, local_train, diagnostic, report


if __name__ == '__main__':
    prepare()
