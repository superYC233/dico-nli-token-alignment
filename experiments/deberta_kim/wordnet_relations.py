"""沿用旧 WordNet 分支的查询策略；不新增词义消歧或关系种类。"""
from functools import lru_cache

from experiments.deberta_kim import config as cfg
from experiments.deberta_kim.lexical_features import tokenize_words
from experiments.deberta_kim.wordnet_resource import ensure_wordnet


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
