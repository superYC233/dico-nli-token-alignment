"""词对关系 -> 对应的子词位置；只接收文本和 tokenizer，不接收任务标签。"""
import torch

from experiments.deberta_kim.lexical_features import TOKEN_PATTERN, tokenize_words
from experiments.deberta_kim.wordnet_relations import candidate_words, word_relations


RELATION_NAMES = ['synonym', 'source_more_specific', 'target_more_specific', 'antonym']
RELATION_VERSION = 'kim_local_residual_wordnet_v1'
REVERSE_CHANNELS = [0, 2, 1, 3]


def word_positions(text, encoded, side):
    """用原文字符偏移找子词。跨词 token 或被截断的词不用于词典证据。"""
    sequence_ids = encoded.sequence_ids()
    offsets = encoded['offset_mapping']
    spans = []
    for match in TOKEN_PATTERN.finditer(text):
        words = tokenize_words(match.group())
        if len(words) == 1:
            spans.append((match.start(), match.end(), words[0]))

    positions = {}
    for start, end, word in spans:
        indices = []
        covered = set()
        for index, (left, right) in enumerate(offsets):
            if sequence_ids[index] != side or right <= left:
                continue
            overlapping = [span for span in spans if left < span[1] and right > span[0]]
            if len(overlapping) == 1 and overlapping[0] == (start, end, word):
                indices.append(index)
                covered.update(range(max(start, left), min(end, right)))
        if indices and covered == set(range(start, end)):
            positions.setdefault(word, []).extend(indices)
    return positions


def build_relation_edges(text1, text2, encoded):
    first = set(candidate_words(text1))
    second = set(candidate_words(text2))
    shared = first & second
    first = sorted(first - shared)
    second = sorted(second - shared)
    positions1 = word_positions(text1, encoded, 0)
    positions2 = word_positions(text2, encoded, 1)
    edges = []
    evidence = []
    for word1 in first:
        if word1 not in positions1:
            continue
        for word2 in second:
            if word2 not in positions2:
                continue
            relation = tuple(int(flag) for flag in word_relations(word1, word2))
            if not any(relation):
                continue
            for i in positions1[word1]:
                for j in positions2[word2]:
                    edges.append((i, j, relation))
            evidence.append({'word1': word1, 'word2': word2, 'relation': relation})
    return edges, evidence


def relation_tensor(edges, length):
    # 只存 Text1 -> Text2；另一方向由模型转置并交换上下位通道得到。
    # uint8 节省 CPU/传输内存，进入计算时再转为浮点数。
    result = torch.zeros(length, length, len(RELATION_NAMES), dtype=torch.uint8)
    for i, j, relation in edges:
        result[i, j] = torch.tensor(relation, dtype=torch.uint8)
    return result
