"""不加载模型：检查来源、反转、重叠与人工审核是否被正确处理。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from experiments.deberta_minimal_contrasts import config as cfg
from experiments.deberta_minimal_contrasts.prepare_diagnostics import build_examples, pair_key, normalize
from experiments.deberta_minimal_contrasts.evaluate_diagnostics import read_review, save_summaries


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(cfg.TRAIN_PATH, keep_default_na=False)
        cls.dev = pd.read_csv(cfg.DEV_PATH, keep_default_na=False)
        cls.examples, cls.families = build_examples(cls.train)

    def test_source_and_reversal(self):
        self.assertEqual(len(self.families), 20)
        self.assertEqual(len(self.examples), 240)
        lookup = self.train.set_index('instance_id')
        inverse = {'F': 'B', 'B': 'F', 'E': 'E'}
        for _, group in self.examples.groupby('pair_id'):
            first = group.iloc[0]
            if first.proposed_label == 'N':
                self.assertEqual(len(group), 1)
            else:
                self.assertEqual(len(group), 2)
                second = group.iloc[1]
                self.assertEqual(first.text1, second.text2)
                self.assertEqual(first.text2, second.text1)
                self.assertEqual(inverse[first.proposed_label], second.proposed_label)
            if first.kind == 'anchor':
                original = lookup.loc[first.source_instance_id]
                self.assertEqual((first.text1, first.text2), (original.text1, original.text2))

    def test_binding_and_order_have_same_word_counts(self):
        # E 的顺序对照和 N 的属性归属对照都保持词袋，不能用词袋相同作答案。
        for row in self.examples[self.examples.kind.isin(['modifier_binding', 'conjunct_order'])].itertuples():
            self.assertEqual(Counter(normalize(row.text1).split()), Counter(normalize(row.text2).split()))
        labels_per_input = self.examples.groupby(['text1', 'text2']).proposed_label.nunique()
        self.assertTrue((labels_per_input == 1).all())

    def test_prepared_dev_exclusions(self):
        prepared = pd.read_csv(cfg.DATA_DIR / 'examples.csv', keep_default_na=False)
        dev_keys = {pair_key(r.text1, r.text2) for r in self.dev.itertuples()}
        for row in prepared[prepared.eligible].itertuples():
            self.assertNotIn(pair_key(row.text1, row.text2), dev_keys)
        self.assertEqual(prepared.eligible.sum(), 238)
        self.assertTrue(prepared[prepared.kind != 'anchor'].eligible.all())

    def test_review_requires_real_label_and_preserves_text(self):
        examples = pd.read_csv(cfg.DATA_DIR / 'examples.csv', keep_default_na=False)
        review = pd.read_csv(cfg.DATA_DIR / 'review.csv', keep_default_na=False)
        index = review.index[review.kind != 'anchor'][0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            examples.to_csv(path / 'examples.csv', index=False)
            with patch.object(cfg, 'DATA_DIR', path):
                review.loc[index, 'review_status'] = 'approved'
                review.loc[index, 'review_label'] = ''
                review.to_csv(path / 'review.csv', index=False)
                with self.assertRaises(ValueError):
                    read_review()
                review.loc[index, 'review_label'] = 'N'
                review.to_csv(path / 'review.csv', index=False)
                self.assertEqual(read_review().loc[index, 'review_label'], 'N')
                review.loc[index, 'text1'] = 'silently changed input'
                review.to_csv(path / 'review.csv', index=False)
                with self.assertRaises(ValueError):
                    read_review()

    def test_pending_never_counts_as_approved_gold(self):
        result = pd.read_csv(cfg.DATA_DIR / 'review.csv', keep_default_na=False).iloc[:12].copy()
        result['prediction'] = result.proposed_label
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch.object(cfg, 'OUTPUT_DIR', path), contextlib.redirect_stdout(io.StringIO()):
                save_summaries(result.copy())
                summary = json.loads((path / 'approved_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['approved_rows'], 0)
                self.assertNotIn('accuracy', summary)
                # 审核人改标签时，必须以改后的标签评分，而不是沿用建议标签。
                index = result.index[result.kind != 'anchor'][0]
                result.loc[index, 'review_status'] = 'approved'
                result.loc[index, 'review_label'] = 'N'
                save_summaries(result.copy())
                summary = json.loads((path / 'approved_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['approved_rows'], 1)
                self.assertEqual(summary['correct'], 0)


if __name__ == '__main__':
    unittest.main()
