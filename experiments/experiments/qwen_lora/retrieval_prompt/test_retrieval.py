"""CPU 检查：重复排除、方向、Dev 标签隔离、非法输出与官方评分。"""
import unittest

import numpy as np
import pandas as pd

import config as cfg
from retrieval_utils import clean_candidate_pool, eligible_indices, pick_examples, render_prompt


def sample(instance, pair, text1, text2, short):
    inverse = {short: full for full, short in cfg.LABEL_MAP.items()}
    return {'instance_id': instance, 'pair_id': pair, 'text1': text1,
            'text2': text2, 'label': inverse[short]}


class RetrievalTests(unittest.TestCase):
    def test_conflicts_include_reversed_direction(self):
        train = pd.DataFrame([
            sample('a', 'p1', 'cat', 'animal', 'F'),
            sample('b', 'p1', 'animal', 'cat', 'B'),
            sample('c', 'p2', 'cat', 'animal', 'N'),
            sample('d', 'p3', 'dog', 'canine', 'E'),
        ])
        pool, excluded = clean_candidate_pool(train)
        self.assertEqual(pool['instance_id'].tolist(), ['d'])
        self.assertEqual(set(excluded['instance_id']), {'a', 'b', 'c'})

    def test_same_and_reverse_duplicates_are_excluded_as_groups(self):
        pool = pd.DataFrame([
            sample('a', 'p1', 'a dog', 'an animal', 'F'),
            sample('b', 'p1', 'an animal', 'a dog', 'B'),
            sample('c', 'p2', 'other text', 'other phrase', 'N'),
            sample('d', 'p3', 'a cat', 'an animal', 'F'),
        ])
        query = {'instance_id':'dev1', 'pair_id':'p2', 'text1':' AN ANIMAL ', 'text2':'A DOG'}
        self.assertEqual(eligible_indices(pool, query), [3])
        # 修改 Dev 标签，不应改变候选资格。
        self.assertEqual(eligible_indices(pool, {**query, 'label':'arbitrary secret'}), [3])

    def test_orientation_and_one_example_per_pair(self):
        pool = pd.DataFrame([
            sample('a', 'p1', 'dog', 'animal', 'F'),
            sample('b', 'p1', 'animal', 'dog', 'B'),
            sample('c', 'p2', 'cat', 'animal', 'F'),
            sample('d', 'p3', 'dog', 'cat', 'N'),
        ])
        scores = np.array([0.6, 0.9, 0.8, 0.2])
        selected = pick_examples(pool, list(range(4)), scores, 2, 'retrieval', 42)
        self.assertEqual(selected, [1, 2])
        self.assertEqual(cfg.LABEL_MAP[pool.iloc[selected[0]]['label']], 'B')
        first = pick_examples(pool, list(range(4)), scores, 2, 'random', 42)
        self.assertEqual(first, pick_examples(pool, list(range(4)), scores, 2, 'random', 42))
        self.assertEqual(len(set(pool.iloc[first]['pair_id'])), 2)

    def test_prompt_uses_only_source_labels(self):
        template = 'Instructions.\n\nText 1: {text1}\nText 2: {text2}\nAnswer:\n'
        query = {'text1':'query {with braces}', 'text2':'another query', 'label':'DO_NOT_COPY_DEV_LABEL'}
        example = sample('source', 'pair', 'source one', 'source two', 'N')
        baseline = render_prompt(template, query, [])
        self.assertEqual(baseline, template.format(text1=query['text1'], text2=query['text2']))
        prompt = render_prompt(template, query, [example])
        self.assertNotIn('DO_NOT_COPY_DEV_LABEL', prompt)
        self.assertIn('Answer: N', prompt)
        self.assertTrue(prompt.endswith('Text 2: another query\nAnswer:\n'))

    def test_illegal_generation_keeps_candidate_scores(self):
        import torch
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        from evaluate_retrieval import predict_batch

        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH, local_files_only=True)
        label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in cfg.LABELS]
        invalid = tokenizer.encode('Class', add_special_tokens=False)[0]

        class FakeModel:
            def generate(self, input_ids, **kwargs):
                scores = torch.full((len(input_ids), len(tokenizer)), -5.0)
                scores[:, label_ids[1]] = 5
                scores[:, invalid] = 10
                generated = torch.full((len(input_ids), 1), invalid, dtype=torch.long)
                return SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1), scores=(scores,))

        raw, scores = predict_batch(['Short\nAnswer:\n', 'A longer question\nAnswer:\n'],
                                     FakeModel(), tokenizer, 'cpu', 128)
        self.assertTrue(all(text not in cfg.LABELS for text in raw))
        self.assertEqual(scores.argmax(dim=1).tolist(), [1, 1])
        self.assertEqual(scores.shape, (2, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)

