"""只从当前两个短语提取特征，不读标签、pair_id、语料统计或外部模型。"""
import math
import re
import unicodedata
from decimal import Decimal


FEATURE_VERSION = 'lexical_v1'

# 只在“去冠词覆盖率”里忽略这三个词。原文本和 DeBERTa 输入完全保留。
# 不删除介词、否定词或数量词，它们可能改变关系。
ARTICLES = {'a', 'an', 'the'}
NEGATIONS = {'no', 'not', 'never', 'neither', 'nor', 'without', 'cannot'}
NUMBER_WORDS = {
    word: str(number) for number, word in enumerate([
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
        'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
        'nineteen', 'twenty',
    ])
}

# 同时保留 3.4、1,000、3m 和 don't 这样的形式。
TOKEN_PATTERN = re.compile(
    r"\d+(?:,\d{3})*(?:\.\d+)?(?:%|[a-z]+)?|[^\W\d_]+(?:'[^\W\d_]+)*"
)

# 固定顺序，保存模型时也会保存这份名单。共 20 维。
FEATURE_NAMES = [
    'word_coverage_1_by_2',
    'word_coverage_2_by_1',
    'word_jaccard',
    'article_free_coverage_1_by_2',
    'article_free_coverage_2_by_1',
    'unique_words_1_log',
    'unique_words_2_log',
    'word_count_1_log',
    'word_count_2_log',
    'signed_length_difference',
    'article_free_1_subset_of_2',
    'article_free_2_subset_of_1',
    'has_number_1',
    'has_number_2',
    'number_jaccard',
    'different_nonempty_number_sets',
    'has_negation_1',
    'has_negation_2',
    'negation_presence_differs',
    'article_free_word_sets_equal',
]


def tokenize_words(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = text.replace('’', "'").replace('‘', "'")
    return TOKEN_PATTERN.findall(text)


def coverage(source, target):
    # source 的词有多少能在 target 中找到，方向不能交换。
    return len(source & target) / len(source) if source else 0.0


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def log_count(count):
    # 固定缩放，不在 Train/Dev 上拟合标准化器。
    return math.log1p(count) / math.log1p(128)


def number_set(tokens):
    numbers = set()
    for token in tokens:
        if token in NUMBER_WORDS:
            numbers.add(NUMBER_WORDS[token])
        elif token[0].isdigit():
            # 只规范化纯数字：3、3.0 视为相同；3m 仍保留单位形式。
            raw = token.replace(',', '')
            if re.fullmatch(r'\d+(?:\.\d+)?', raw):
                raw = format(Decimal(raw).normalize(), 'f')
            numbers.add(raw)
    return numbers


def has_negation(tokens):
    return any(token in NEGATIONS or token.endswith("n't") for token in tokens)


def build_lexical_features(text1, text2):
    tokens1 = tokenize_words(text1)
    tokens2 = tokenize_words(text2)
    words1, words2 = set(tokens1), set(tokens2)
    content1 = words1 - ARTICLES
    content2 = words2 - ARTICLES
    numbers1, numbers2 = number_set(tokens1), number_set(tokens2)
    negative1, negative2 = has_negation(tokens1), has_negation(tokens2)

    values = [
        coverage(words1, words2),
        coverage(words2, words1),
        jaccard(words1, words2),
        coverage(content1, content2),
        coverage(content2, content1),
        log_count(len(content1 - content2)),
        log_count(len(content2 - content1)),
        log_count(len(tokens1)),
        log_count(len(tokens2)),
        (len(tokens1) - len(tokens2)) / max(1, len(tokens1) + len(tokens2)),
        bool(content1) and content1.issubset(content2),
        bool(content2) and content2.issubset(content1),
        bool(numbers1),
        bool(numbers2),
        jaccard(numbers1, numbers2),
        bool(numbers1) and bool(numbers2) and numbers1 != numbers2,
        negative1,
        negative2,
        negative1 != negative2,
        bool(content1) and content1 == content2,
    ]
    return [float(value) for value in values]
