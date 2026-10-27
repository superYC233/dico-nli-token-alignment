"""CPU 单元检查；不加载真实 Qwen，不进行实际训练。"""
import math
import unittest

import torch
from datasets import Dataset

from contrastive_data import add_contrast_groups
from contrastive_trainer import supervised_contrastive_loss


class ContrastiveTests(unittest.TestCase):
    def test_loss_matches_small_hand_calculation(self):
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 0])
        groups = torch.tensor([0, 0, 1, 2])
        loss, valid = supervised_contrastive_loss(features, labels, groups, temperature=0.1)
        expected = (2 * math.log1p(math.exp(-10)) + math.log(2 + math.exp(-10))) / 3
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(valid.item(), 0.75)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_duplicates_alone_do_not_create_positive_examples(self):
        features = torch.randn(3, 8, requires_grad=True)
        loss, valid = supervised_contrastive_loss(
            features, torch.tensor([3, 3, 3]), torch.tensor([7, 7, 7]), 0.1
        )
        self.assertEqual(loss.item(), 0)
        self.assertEqual(valid.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features)))

    def test_no_positive_and_single_sample_are_safe(self):
        for size in [1, 4]:
            features = torch.randn(size, 8, requires_grad=True)
            loss, valid = supervised_contrastive_loss(features, torch.arange(size), torch.arange(size), 0.1)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(valid.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    def test_class_separation_is_rewarded(self):
        labels = torch.tensor([0, 0, 1, 1])
        groups = torch.arange(4)
        good = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        bad = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        good_loss, _ = supervised_contrastive_loss(good, labels, groups, 0.1)
        bad_loss, _ = supervised_contrastive_loss(bad, labels, groups, 0.1)
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_reversed_source_receives_same_group(self):
        data = Dataset.from_dict({
            'text1': ['A dog', 'an animal', 'a cat'],
            'text2': ['an animal', ' a DOG ', 'an animal'],
            'label': ['F', 'B', 'F'],
        })
        result = add_contrast_groups(data)
        self.assertEqual(len(data), len(result))
        self.assertEqual(result['contrast_group'], [0, 0, 1])
        self.assertEqual(result['label'], data['label'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

