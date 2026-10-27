"""严格筛选官方Train原样或副本，再比较等权融合；不训练。"""
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evaluation_functions import score_files
from experiments.model_ensemble.analyze_all_models import OUTPUT as SOURCE, DEV_PATH, FULL, LABELS, EnsembleSearch
from experiments.deberta_lexical_mixed_pair.analyze_fusion import decode, metrics


OUTPUT = Path(__file__).resolve().parent / 'outputs/run02_official_or_copy'


def audit_members(members):
    path = ROOT / 'data/train/dico_nli_train_track1_participant_labeled.csv'
    original = pd.read_csv(path, keep_default_na=False)
    columns = ['text1', 'text2', 'label']
    original_rows = set(original[columns].itertuples(index=False, name=None))
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for index, member in enumerate(members):
        folder = Path(member['prediction_file']).parent
        while folder != ROOT and not (folder / 'run_config.json').is_file():
            folder = folder.parent
        cfg = json.loads((folder / 'run_config.json').read_text(encoding='utf-8')) if folder != ROOT else {}
        evidence = float(cfg.get('evidence_weight', 0)) > 0 or float(cfg.get('reason_weight', 0)) > 0
        record = {'id': member['id'], 'name': member['name'], 'index': index,
                  'checkpoint_found': member['checkpoint_found'], 'extra_evidence_annotations': evidence,
                  'eligible_texts': False, 'source': '', 'new_text_label_triplets': None}
        snapshot = next((folder / name for name in ['training_data.csv', 'train.csv', 'train_features.csv'] if (folder / name).is_file()), None)
        if snapshot is not None:
            table = pd.read_csv(snapshot, keep_default_na=False)
            if set(columns).issubset(table):
                rows = set(table[columns].itertuples(index=False, name=None))
                new = rows - original_rows
                record.update({'eligible_texts': len(new) == 0, 'source': str(snapshot),
                               'new_text_label_triplets': len(new), 'actual_train_rows': len(table)})
        elif cfg.get('train_rows') == 3042 and cfg.get('train_sha256') == original_hash:
            record.update({'eligible_texts': True, 'source': 'Saved Train SHA256 equals official Train.',
                           'new_text_label_triplets': 0, 'actual_train_rows': 3042})
        else:
            record['source'] = 'No verified original-or-copy snapshot; excluded.'
        records.append(record)
    return pd.DataFrame(records)


def main():
    if (OUTPUT / 'RESULTS.md').exists():
        raise FileExistsError('本次报告已存在；请修改输出RUN_NAME保留旧结果。')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    members = json.loads((SOURCE / 'members.json').read_text(encoding='utf-8'))
    values = np.load(SOURCE / 'aligned_scores.npz')['log_probability']
    for member, lp in zip(members, values):
        member['log_probability'] = lp
    audit = audit_members(members)
    audit.to_csv(OUTPUT / 'data_audit.csv', index=False, encoding='utf-8-sig')
    dev = pd.read_csv(DEV_PATH, keep_default_na=False).reset_index(drop=True)
    gold = dev.label.map(dict(zip(FULL, range(4)))).to_numpy()
    pairs = np.array([g.index.to_numpy() for _, g in dev.groupby('pair_id', sort=False) if len(g) == 2])
    recommendations = []
    for cohort in ['original_labels_only', 'allow_evidence_annotations']:
        eligible = audit.eligible_texts & audit.checkpoint_found
        if cohort == 'original_labels_only':
            eligible &= ~audit.extra_evidence_annotations
        indices = audit.loc[eligible, 'index'].tolist()
        search = EnsembleSearch(members, gold, pairs)
        print(cohort, 'eligible members:', len(indices), flush=True)
        with (OUTPUT / f'{cohort}_all_recipes.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=search.fields)
            writer.writeheader()
            for mode in ['probability', 'log_probability', 'hier_0930']:
                for size in [1, 2, 3]:
                    search.evaluate(iter(combinations(indices, size)), mode, writer)
                for size in [4, 5, 6]:
                    seeds = {item[2] for key, queue in search.heaps.items() if key[0] == mode and key[1] == size - 1 for item in queue}
                    extensions = {tuple(sorted((*seed, extra))) for seed in seeds for extra in indices if extra not in seed}
                    search.evaluate(iter(sorted(extensions)), mode, writer)
                print(cohort, mode, 'done', flush=True)
        for mode in ['probability', 'log_probability', 'hier_0930']:
            for objective in ['independent', 'pair', 'balanced']:
                queue = [item for key, q in search.heaps.items() if key[0] == mode and key[2] == 'weights_found' and key[3] == objective for item in q if item[3]['model_count'] >= 2]
                item = max(queue, key=lambda x: (x[0], -x[3]['model_count']))
                row = item[3].copy()
                row.update({'cohort': cohort, 'objective': objective,
                            'direction_names': ' | '.join(members[i]['name'] for i in item[2])})
                recommendations.append(row)

    recipes = {}
    for i, row in enumerate(recommendations, 1):
        code = f'C{i:02d}'
        row['code'] = code
        indices = [int(x[1:]) - 1 for x in row['members'].split('+')]
        search = EnsembleSearch(members, gold, pairs)
        logits = search.combine(np.array([indices]), row['mode'])[0]
        for mode in ['independent', 'pair_aware']:
            predicted = decode(logits, pairs, mode)
            directory = OUTPUT / 'official_checks' / code / mode
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'predictions.csv'
            pd.DataFrame({'instance_id': dev.instance_id, 'label': np.array(FULL)[predicted]}).to_csv(path, index=False)
            report = score_files(ROOT / 'data/dev/dico_nli_dev_track1_reference.csv', path, output_dir=directory)
            expected = row['independent_f1'] if mode == 'independent' else row['pair_f1']
            assert abs(report.weighted_f1 - expected) < 1e-12
        all_indices = sorted(set(indices) | ({search.gates[row['mode']]} if row['mode'].startswith('hier_') else set()))
        assert audit.iloc[all_indices].eligible_texts.all()
        if row['cohort'] == 'original_labels_only':
            assert not audit.iloc[all_indices].extra_evidence_annotations.any()
        recipes[code] = {'recipe': row, 'members': [{k: v for k, v in members[j].items() if k != 'log_probability'} for j in all_indices]}
    table = pd.DataFrame(recommendations)
    table.to_csv(OUTPUT / 'recommendations.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'recipes.json').write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 仅官方Train或原样复制：多模型融合', '',
             '筛选标准为有序(text1,text2,label)必须已在官方Train出现；允许重复，不能交换N再称原样复制。',
             '所有代表方案的已有权重文件均找到；未找到权重的B/D复制实验仍在data_audit.csv列出，但不进入推荐。',
             'original_labels_only进一步排除reason/evidence额外辅助标注；allow_evidence_annotations允许这类标注，但仍不新增训练文本。',
             '普通平均=四类softmax后等权平均；几何平均=log-prob等权平均；分层仅允许官方数据训练的0.9306作为N门，禁止增强数据的0.938作为N门。', '',
             '| 编号 | 监督范围 | 融合方式 | 优先目标 | 模型数 | 独立F1 | pair-aware F1 |',
             '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for row in recommendations:
        lines.append(f"| {row['code']} | {row['cohort']} | {row['mode']} | {row['objective']} | {row['model_count']} | {row['independent_f1']:.6f} | {row['pair_f1']:.6f} |")
    for code, recipe in recipes.items():
        lines += ['', f'## {code}', '', f"配方：{recipe['recipe']['mode']}，方向/投票成员={recipe['recipe']['members']}，N门={recipe['recipe']['gate'] or '无独立N门'}。", '']
        for member in recipe['members']:
            lines.append(f"- {member['id']}：{member['name']}；行数={audit.loc[audit.id.eq(member['id']), 'actual_train_rows'].iloc[0]}。")
            lines.append(f"  权重：`{' ; '.join(member['checkpoint_candidates_existing'])}`")
    lines += ['', '## 解释边界', '',
              '1—3成员穷举，较大组合beam扩展。只计算已有预测，不重新训练或运行模型，不强制单例N。',
              '最高值仍是大量Dev选择后的探索性结果，不等于隐藏Test成绩。原始标签限制不排除公开NLI预训练初始化、手工Lexical特征、OOF重加权或不同架构。',
              '名字带copy但底座含scope200/relation48的旧模型不合格，例如local128_copy、family768_copy。是否合格以实际训练快照和官方Train内容比较为准。']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(table[['code', 'cohort', 'mode', 'objective', 'model_count', 'independent_f1', 'pair_f1', 'members', 'gate']].to_string(index=False))


if __name__ == '__main__':
    main()
