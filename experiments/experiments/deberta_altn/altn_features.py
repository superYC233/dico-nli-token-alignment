"""词面 LCS 对齐后的 8 维差异位置特征；不使用任务标签或语料统计。"""
from experiments.deberta_altn.lexical_features import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    FEATURE_VERSION as BASE_FEATURE_VERSION,
    build_lexical_features, tokenize_words,
)


POSITION_FEATURE_NAMES = [
    'unmatched_runs_1_ratio',
    'longest_unmatched_run_1_ratio',
    'first_unmatched_position_1_ratio',
    'last_unmatched_position_1_ratio',
    'unmatched_runs_2_ratio',
    'longest_unmatched_run_2_ratio',
    'first_unmatched_position_2_ratio',
    'last_unmatched_position_2_ratio',
]
FEATURE_NAMES = list(BASE_FEATURE_NAMES) + POSITION_FEATURE_NAMES
FEATURE_VERSION = BASE_FEATURE_VERSION + '_altn_lcs_position_v1'
ALIGNMENT_METHOD = 'canonical-order exact-token longest common subsequence'


def lcs_matches(first, second):
    """顺序不交叉、一对一的相同词匹配；重复词不能一词多次匹配。"""
    n, m = len(first), len(second)
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if first[i] == second[j]:
                lengths[i][j] = 1 + lengths[i + 1][j + 1]
            else:
                lengths[i][j] = max(lengths[i + 1][j], lengths[i][j + 1])

    matches = []
    i, j = 0, 0
    while i < n and j < m:
        if first[i] == second[j]:
            matches.append((i, j))
            i += 1
            j += 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return matches


def align_words(first, second):
    # 多个 LCS 同样长时，固定使用规范顺序打破平局。
    # 因而交换文本会准确交换两侧掩码，不产生“总是偏向 Text1”的额外偏置。
    if tuple(first) <= tuple(second):
        matches = lcs_matches(first, second)
    else:
        matches = [(j, i) for i, j in lcs_matches(second, first)]
    matched1 = [False] * len(first)
    matched2 = [False] * len(second)
    for i, j in matches:
        matched1[i] = True
        matched2[j] = True
    return matched1, matched2, matches


def unmatched_spans(matched):
    """返回未匹配片段的半开区间 [start, end)，仅表示词面差异。"""
    spans = []
    start = None
    for index, is_matched in enumerate(matched):
        if not is_matched and start is None:
            start = index
        elif is_matched and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(matched)))
    return spans


def side_features(matched):
    n = len(matched)
    spans = unmatched_spans(matched)
    if n == 0 or not spans:
        return [0.0, 0.0, 0.0, 0.0]
    # 位置使用从 1 开始的词号 / 词数；0 专门表示没有差异或空文本。
    return [
        len(spans) / n,
        max(end - start for start, end in spans) / n,
        (spans[0][0] + 1) / n,
        spans[-1][1] / n,
    ]


def build_position_features(text1, text2):
    words1, words2 = tokenize_words(text1), tokenize_words(text2)
    matched1, matched2, _ = align_words(words1, words2)
    return side_features(matched1) + side_features(matched2)


def build_features(text1, text2):
    # 原 20 维逐项保留；不删除冠词/介词/否定词，不改 DeBERTa 原始输入。
    return build_lexical_features(text1, text2) + build_position_features(text1, text2)


def alignment_audit(text1, text2):
    words1, words2 = tokenize_words(text1), tokenize_words(text2)
    matched1, matched2, matches = align_words(words1, words2)
    return {
        'words1': words1, 'words2': words2,
        'matched1': matched1, 'matched2': matched2,
        'matches_zero_based': matches,
        'unmatched_spans1': unmatched_spans(matched1),
        'unmatched_spans2': unmatched_spans(matched2),
    }
