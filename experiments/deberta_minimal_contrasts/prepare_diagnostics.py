"""从 Train 来源建立待审核诊断集；不写回 Train，也不自动批准新标签。"""

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from experiments.deberta_minimal_contrasts import config as cfg


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def pair_key(text1, text2):
    # 一起检查两个方向，防止变体碰到已知 Dev 的反向。
    return tuple(sorted([normalize(text1), normalize(text2)]))


def build_examples(train):
    lookup = train.set_index('instance_id')
    rows = []
    groups = []

    def add_pair(family, source, kind, text1, text2, label, reason, observed=False):
        pair_id = f'contrast_{family}_{kind}'
        reverse = {'F': 'B', 'B': 'F', 'E': 'E'}
        versions = [(text1, text2, label)]
        if label != 'N':
            versions.append((text2, text1, reverse[label]))
        # N 仅作为一条独立四分类题，不假定官方对 N 评反转一致性。
        for direction, (first, second, proposed) in enumerate(versions):
            rows.append({
                'instance_id': pair_id + f'__{direction}', 'pair_id': pair_id,
                'family_id': family, 'source_instance_id': source.name,
                'kind': kind, 'text1': first, 'text2': second,
                'proposed_label': proposed, 'label_reason': reason,
                'reference_status': 'observed_train' if observed else 'proposed_unreviewed',
                'purpose': 'diagnostic_only',
            })

    for family, suffix, noun, modifier, foil, predicate, person in cfg.FAMILIES:
        source_id = 'dico_phrasis_' + suffix
        source = lookup.loc[source_id]
        if source['label'] != 'FORWARD_ENTAILMENT':
            raise ValueError(f'{source_id} 不是预期的 Train F 来源。')
        narrow, broad = source['text1'], source['text2']
        determiner = narrow.split()[0]
        if determiner.lower() not in ['a', 'an', 'the']:
            raise ValueError('来源不符合本次名词短语模板：' + narrow)
        # an orange ball 展开为 a ball that is orange，要同步修正 a/an。
        expanded_determiner = determiner
        if determiner.lower() == 'an':
            expanded_determiner = 'A' if determiner[0].isupper() else 'a'
        relative = 'who' if person else 'that'
        expanded = f'{expanded_determiner} {noun} {relative} {predicate}'
        foil_narrow = f'a {modifier} {foil}'
        foil_broad = f'a {foil}'
        add_pair(family, source, 'anchor', narrow, broad, 'F',
                 '直接沿用 Train F 及其 B 反向，仅作已见样本对照。', observed=True)
        add_pair(family, source, 'restriction_rephrase', expanded, broad, 'F',
                 '建议：限定属性展开为定语从句，仍比普通中心词范围窄；需审核词义和指称。')
        add_pair(family, source, 'equivalence_rephrase', narrow, expanded, 'E',
                 '建议：属性前置与同一属性的从句展开保义；文字更长不自动增加限制。')
        add_pair(family, source, 'modifier_scope', narrow + ' and ' + foil_narrow,
                 narrow + ' and ' + foil_broad, 'F',
                 '建议：左侧第二个对象也受修饰，右侧不要求该属性；不从第一个并列项传播修饰。')
        add_pair(family, source, 'core_object_change', narrow, foil_narrow, 'N',
                 '建议：保留属性但改变中心对象；所选对象互不构成一般集合包含，不代表所有换中心词都为 N。')
        # 同样的词，属性归属不同：白狗+猫，与狗+白猫不能互相推出。
        # 这道题与下面的并列换序 E 对照，不能靠词袋是否相同决定标签。
        bare_first = f'{expanded_determiner} {noun}'
        add_pair(family, source, 'modifier_binding', narrow + ' and ' + foil_broad,
                 bare_first + ' and ' + foil_narrow, 'N',
                 '建议：属性从第一个对象移到第二个对象；保留词汇计数，但信息不同且无单向包含。')
        lower_narrow = narrow[0].lower() + narrow[1:]
        add_pair(family, source, 'conjunct_order', narrow + ' and ' + foil_broad,
                 foil_broad + ' and ' + lower_narrow, 'E',
                 '建议：两个名词短语仅交换并列顺序；无时间先后关系，原修饰词仍跟随原对象。')
        groups.append({'family_id': family, 'source_instance_id': source_id,
                       'source_text1': narrow, 'source_text2': broad,
                       'noun': noun, 'modifier': modifier, 'foil_noun': foil,
                       'expanded_phrase': expanded})
    return pd.DataFrame(rows), pd.DataFrame(groups)


def save_review_focus(examples):
    # 浓缩范围/归属/换序的对照，不展示模型答案；正式审核状态仍写 review.csv。
    path = cfg.DATA_DIR / 'review_focus.csv'
    if path.exists():
        return
    focus = examples[examples.kind.isin(['modifier_scope', 'modifier_binding', 'conjunct_order'])]
    focus = focus.groupby(['family_id', 'kind'], sort=False).head(1)
    focus[['family_id', 'kind', 'text1', 'text2', 'proposed_label', 'label_reason']].to_csv(
        path, index=False, encoding='utf-8-sig')


def main():
    train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
    dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
    manifest = {'train_sha256': file_hash(cfg.TRAIN_PATH), 'dev_sha256': file_hash(cfg.DEV_PATH),
                'families': cfg.FAMILIES, 'format_version': 2,
                'purpose': 'Train-derived diagnostic candidates, not training augmentation'}
    # JSON 往返后 tuple 变 list，统一后再比较。
    manifest = json.loads(json.dumps(manifest))
    path = cfg.DATA_DIR / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != manifest:
            raise ValueError('来源或模板已变化，请改 RUN_NAME，避免覆盖已审核文件。')
        for name in ['examples.csv', 'review.csv', 'families.csv', 'overlap_audit.csv']:
            if not (cfg.DATA_DIR / name).exists():
                raise FileNotFoundError('已有准备目录不完整：' + name)
        save_review_focus(pd.read_csv(cfg.DATA_DIR / 'examples.csv', keep_default_na=False))
        print('复用诊断集，保留你对 review.csv 的修改：', cfg.DATA_DIR)
        return

    examples, families = build_examples(train)
    train_keys = {pair_key(r.text1, r.text2) for r in train.itertuples()}
    dev_keys = {pair_key(r.text1, r.text2) for r in dev.itertuples()}
    train_phrases = {normalize(t) for t in list(train.text1) + list(train.text2)}
    examples['seen_pair_in_train'] = [pair_key(a, b) in train_keys for a, b in zip(examples.text1, examples.text2)]
    examples['overlaps_dev'] = [pair_key(a, b) in dev_keys for a, b in zip(examples.text1, examples.text2)]
    examples['both_phrases_seen_in_train'] = [normalize(a) in train_phrases and normalize(b) in train_phrases
                                               for a, b in zip(examples.text1, examples.text2)]
    # Dev 只用于重叠排除，不用来提供标签或筛选模型错误。
    # 一行碰到 Dev 时，整个反转 pair 一起排除；保留该家族的其他探针。
    blocked_pairs = set(examples.loc[examples.overlaps_dev, 'pair_id'])
    examples['eligible'] = ~examples.pair_id.isin(blocked_pairs)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    examples.to_csv(cfg.DATA_DIR / 'examples.csv', index=False, encoding='utf-8-sig')
    families.to_csv(cfg.DATA_DIR / 'families.csv', index=False, encoding='utf-8-sig')
    examples[['instance_id', 'pair_id', 'family_id', 'kind', 'seen_pair_in_train',
              'overlaps_dev', 'eligible']].to_csv(cfg.DATA_DIR / 'overlap_audit.csv', index=False, encoding='utf-8-sig')
    review = examples.copy()
    review['review_status'] = ['observed_train' if status == 'observed_train' else 'pending'
                               for status in review.reference_status]
    review['review_label'] = ['' for _ in range(len(review))]
    review['review_note'] = ['' for _ in range(len(review))]
    review.to_csv(cfg.DATA_DIR / 'review.csv', index=False, encoding='utf-8-sig')
    save_review_focus(examples)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('语义家族：', len(families), '；候选行：', len(examples))
    print('与 Dev 重叠而排除的行：', int((~examples.eligible).sum()))
    print('新探针的标签尚待审核；没有添加任何训练数据。')
    print('审核文件：', cfg.DATA_DIR / 'review.csv')


if __name__ == '__main__':
    main()
