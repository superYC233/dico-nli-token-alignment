"""保留原 20 维，新增 12 维 WordNet 辅助证据；不从词典直接给任务标签。"""
from functools import lru_cache

from experiments.deberta_wordnet import config as cfg
from experiments.deberta_wordnet.lexical_features import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    build_lexical_features, tokenize_words,
)
from experiments.deberta_wordnet.wordnet_resource import ensure_wordnet


FEATURE_VERSION = 'lexical_v1_wordnet_v1'
WORDNET_FEATURE_NAMES = [
    'wn_lookup_coverage_1',
    'wn_lookup_coverage_2',
    'wn_residual_fraction_1',
    'wn_residual_fraction_2',
    'wn_synonym_coverage_1',
    'wn_synonym_coverage_2',
    'wn_specific_1_to_general_2',
    'wn_specific_2_to_general_1',
    'wn_antonym_coverage_1',
    'wn_antonym_coverage_2',
    'wn_relation_coverage_1',
    'wn_relation_coverage_2',
]
FEATURE_NAMES = list(BASE_FEATURE_NAMES) + WORDNET_FEATURE_NAMES

# 只限制 WordNet 查询范围；不修改旧 20 维特征，也不修改 DeBERTa 的输入。
FUNCTION_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'his', 'her', 'its', 'their', 'our',
    'of', 'in', 'on', 'at', 'to', 'for', 'from', 'with', 'by', 'as', 'into',
    'over', 'under', 'after', 'before', 'between', 'through', 'during', 'without',
    'and', 'or', 'but', 'if', 'than', 'no', 'not', 'never', 'neither', 'nor',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 's',
}


def candidate_words(text):
    return sorted({
        word for word in tokenize_words(text)
        if word not in FUNCTION_WORDS and not word[0].isdigit() and len(word) > 1
    })


@lru_cache(maxsize=50000)
def word_senses(word):
    reader = ensure_wordnet()
    selected = []
    # 无额外 POS 标注器：每种词性最多取第一个词义，避免遍历全部罕见词义。
    # 这不是语境消歧，仍可能选错词义，所以只作为软特征。
    for pos in ('n', 'v', 'a', 'r'):
        senses = reader.synsets(word, pos=pos)
        if senses:
            selected.append(senses[0].name())
    return tuple(dict.fromkeys(selected))


@lru_cache(maxsize=50000)
def hypernym_names(synset_name):
    reader = ensure_wordnet()
    current = [reader.synset(synset_name)]
    seen = {synset_name}
    ancestors = set()
    # 只沿“是某一类”的方向向上走，不把共享祖先误当成蕴含。
    for _ in range(cfg.WORDNET_MAX_HYPERNYM_DEPTH):
        next_level = []
        for synset in current:
            for parent in synset.hypernyms() + synset.instance_hypernyms():
                name = parent.name()
                if name not in seen:
                    seen.add(name)
                    ancestors.add(name)
                    next_level.append(parent)
        current = next_level
    return frozenset(ancestors)


@lru_cache(maxsize=50000)
def antonym_names(synset_name):
    reader = ensure_wordnet()
    names = set()
    for lemma in reader.synset(synset_name).lemmas():
        for opposite in lemma.antonyms():
            names.add(opposite.synset().name())
    return frozenset(names)


@lru_cache(maxsize=100000)
def word_relations(word1, word2):
    senses1 = set(word_senses(word1))
    senses2 = set(word_senses(word2))
    synonym = bool(senses1 & senses2)
    specific1 = any(senses2 & hypernym_names(name) for name in senses1)
    specific2 = any(senses1 & hypernym_names(name) for name in senses2)
    opposite = (
        any(senses2 & antonym_names(name) for name in senses1)
        or any(senses1 & antonym_names(name) for name in senses2)
    )
    return synonym, specific1, specific2, opposite


def relation_coverage(source, target):
    totals = [0, 0, 0, 0]  # 同义、source 更具体、反义、任一种已知关系。
    evidence = []
    for word1 in source:
        found = [False, False, False, False]
        for word2 in target:
            synonym, specific1, specific2, opposite = word_relations(word1, word2)
            found[0] |= synonym
            found[1] |= specific1
            found[2] |= opposite
            found[3] |= synonym or specific1 or specific2 or opposite
            if (synonym or specific1 or specific2 or opposite) and len(evidence) < 12:
                evidence.append({
                    'source': word1, 'target': word2,
                    'synonym': synonym, 'source_more_specific': specific1,
                    'target_more_specific': specific2, 'antonym': opposite,
                })
        totals = [total + int(flag) for total, flag in zip(totals, found)]
    values = [total / len(source) if source else 0.0 for total in totals]
    return values, evidence


def build_wordnet_features(text1, text2):
    words1, words2 = candidate_words(text1), candidate_words(text2)
    # 先去掉共有表面词，聚焦 car/vehicle 等差异，避免共同的 black 掩盖它们。
    shared = set(words1) & set(words2)
    residual1 = [word for word in words1 if word not in shared]
    residual2 = [word for word in words2 if word not in shared]
    known1 = [word for word in residual1 if word_senses(word)]
    known2 = [word for word in residual2 if word_senses(word)]
    forward, evidence = relation_coverage(residual1, residual2)
    reverse, _ = relation_coverage(residual2, residual1)
    values = [
        len(known1) / len(residual1) if residual1 else 0.0,
        len(known2) / len(residual2) if residual2 else 0.0,
        len(residual1) / len(words1) if words1 else 0.0,
        len(residual2) / len(words2) if words2 else 0.0,
        forward[0], reverse[0], forward[1], reverse[1],
        forward[2], reverse[2], forward[3], reverse[3],
    ]
    audit = {
        'residual_words_1': residual1, 'residual_words_2': residual2,
        'known_words_1': known1, 'known_words_2': known2,
        'selected_senses_1': {word: list(word_senses(word)) for word in residual1},
        'selected_senses_2': {word: list(word_senses(word)) for word in residual2},
        'evidence': evidence,
    }
    return values, audit


def build_features(text1, text2, use_wordnet_features=True):
    base = build_lexical_features(text1, text2)
    if use_wordnet_features:
        extra, audit = build_wordnet_features(text1, text2)
    else:
        extra = [0.0] * len(WORDNET_FEATURE_NAMES)
        audit = {}
    return base + extra, audit
