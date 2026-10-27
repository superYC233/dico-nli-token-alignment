"""合并100组旧标注与1660组新增说明，覆盖全Train；不读取Dev。"""
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from experiments.deberta_lexical_mixed_pair import config_reason as cfg


PILOT_PATH = cfg.EXPERIMENT_DIR / 'reason_annotations_pilot100.csv'
REVIEW_PATH = cfg.EXPERIMENT_DIR / 'reason_semantic_review.tsv'
WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*")
FUNCTION = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being',
            'of', 'on', 'in', 'at', 'to', 'for', 'from', 'by', 'with', 'its', 'his', 'her'}


def cue_phrases(first, second):
    """依语义说明定位表面变化；文字理由已在CSV/TSV逐组撰写。"""
    words = [list(WORD.finditer(first)), list(WORD.finditer(second))]
    keys = [[m.group(0).casefold() for m in items] for items in words]
    selected = [[], []]
    for tag, a, b, c, d in SequenceMatcher(None, *keys, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        for side, text, start, end in [(0, first, a, b), (1, second, c, d)]:
            if end > start:
                phrase = text[words[side][start].start():words[side][end - 1].end()]
                if text.count(phrase) == 1 and phrase.casefold() not in {'a', 'an', 'the'}:
                    selected[side].append(phrase)
    if not any(selected):
        for side, text in enumerate([first, second]):
            selected[side] = [m.group(0) for m in words[side]
                              if m.group(0).casefold() not in FUNCTION and text.count(m.group(0)) == 1]
    if not any(selected):
        selected = [[first] if first else [], [second] if second else []]
    return ['|'.join(dict.fromkeys(items)) for items in selected]


def main():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    if not PILOT_PATH.exists():
        PILOT_PATH.write_bytes(cfg.ANNOTATION_PATH.read_bytes())
    pilot = pd.read_csv(PILOT_PATH, keep_default_na=False)
    by_id = train.set_index('instance_id')
    pilots = {by_id.loc[a.instance_id, 'pair_id']: a for a in pilot.itertuples()}
    reviews = pd.read_csv(REVIEW_PATH, sep='\t', dtype=str, keep_default_na=False)
    assert reviews.pair_id.is_unique
    reviews = reviews.set_index('pair_id')
    covered = set(pilots) | {'dico_phrasis_' + value for value in reviews.index}
    assert covered == set(train.pair_id), '每个Train组都必须有逐组说明。'
    records = []
    for pair_id, group in train.groupby('pair_id', sort=False):
        p = pilots.get(pair_id)
        row = by_id.loc[p.instance_id] if p is not None else group.iloc[0]
        instance_id = p.instance_id if p is not None else row.instance_id
        if p is not None:
            status, operation, reason = 'supported', p.operation, p.reason
            cue1, cue2 = p.evidence_text1, p.evidence_text2
            evidence_source = 'assistant_selected'
        else:
            reviewed = reviews.loc[pair_id.rsplit('_', 1)[1]]
            status, operation, reason = reviewed.annotation_status, reviewed.operation, reviewed.reason
            cue1, cue2 = cue_phrases(row.text1, row.text2)
            evidence_source = 'surface_difference_draft'
        records.append({
            'instance_id': instance_id, 'evidence_text1': cue1, 'evidence_text2': cue2,
            'operation': operation, 'reason': reason, 'annotation_status': status,
            'annotation_source': 'assistant_authored_train_only_draft', 'evidence_source': evidence_source,
        })
    label_sets = defaultdict(set)
    group_keys = defaultdict(set)
    for row in train.itertuples():
        key = (' '.join(row.text1.casefold().split()), ' '.join(row.text2.casefold().split()))
        label_sets[key].add(row.label)
        group_keys[row.pair_id].add(key)
    conflicts = {key for key, labels in label_sets.items() if len(labels) > 1}
    for record in records:
        pair_id = by_id.loc[record['instance_id'], 'pair_id']
        if group_keys[pair_id] & conflicts:
            record['annotation_status'] = 'label_question'
            record['reason'] += ' This ordered input has conflicting labels elsewhere in Train; auxiliary evidence is withheld.'
    frame = pd.DataFrame(records)
    assert len(frame) == 1760 and frame.instance_id.is_unique and frame.reason.str.strip().ne('').all()
    frame.to_csv(cfg.ANNOTATION_PATH, index=False, encoding='utf-8-sig')
    info = {
        'groups': len(frame), 'group_status_counts': frame.annotation_status.value_counts().to_dict(),
        'reason_source': '100 original assistant-authored groups + 1660 additional assistant-authored group notes',
        'new_evidence_source': 'surface differences in original text; draft, not independently adjudicated',
        'selection_inputs': ['official Train text and labels', PILOT_PATH.name, REVIEW_PATH.name],
        'dev_or_test_read': False, 'independent_human_adjudication': False,
    }
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.PREPARED_DIR / 'completion_report.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
