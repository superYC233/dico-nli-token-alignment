"""CPU 检查：显式 tokenizer、非法生成、补齐恢复和历史结构解码。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import smol_config as cfg
from smol_utils import get_label_ids, predict_with_scores, pair_aware_decode


def tiny_tokenizer():
    backend = Tokenizer(WordLevel(
        {'<unk>': 0, '<eos>': 1, '<pad>': 2, 'F': 3, 'B': 4, 'E': 5, 'N': 6, 'Class': 7},
        unk_token='<unk>',
    ))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='<unk>', eos_token='<eos>', pad_token='<pad>',
    )


class FakeModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = tokenizer

    def generate(self, input_ids, attention_mask, generation_config, **kwargs):
        self.last_inputs = input_ids
        self.last_mask = attention_mask
        self.last_training = self.training
        scores = torch.zeros((len(input_ids), len(self.tokenizer)))
        scores[:, get_label_ids(self.tokenizer)[3]] = 5
        illegal = self.tokenizer.encode('Class', add_special_tokens=False)[0]
        scores[:, illegal] = 10
        output = torch.full((len(input_ids), 1), illegal, dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat([input_ids, output], dim=1), scores=(scores,))


class SmolTests(unittest.TestCase):
    def test_passed_tokenizer_and_illegal_generation(self):
        tokenizer = tiny_tokenizer()
        tokenizer.padding_side = 'right'
        model = FakeModel(tokenizer)
        model.train()
        rows = Dataset.from_dict({
            'instance_id': ['x', 'y'], 'pair_id': ['x', 'y'],
            'prompt': ['short', 'a much longer prompt'],
            'label': ['NEGATIVE_OTHER', 'NEGATIVE_OTHER'],
        })
        frame = predict_with_scores(model, tokenizer, rows)
        self.assertEqual(frame['prediction'].tolist(), ['N', 'N'])
        self.assertEqual(frame['generated_prediction'].tolist(), ['Class', 'Class'])
        self.assertEqual(frame['logit_N'].tolist(), [5, 5])
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertTrue(model.training)
        self.assertFalse(model.last_training)
        self.assertEqual(model.last_mask[0, -1].item(), 1)
        self.assertEqual(model.last_mask[0, 0].item(), 0)

    def test_gold_is_not_used_for_readout(self):
        tokenizer = tiny_tokenizer()
        model = FakeModel(tokenizer)
        rows = Dataset.from_dict({
            'instance_id': ['x'], 'pair_id': ['x'], 'prompt': ['same input'], 'label': ['SECRET'],
        })
        frame = predict_with_scores(model, tokenizer, rows)
        self.assertEqual(frame.iloc[0]['prediction'], 'N')
        self.assertNotIn('SECRET', tokenizer.decode(model.last_inputs[0]))

    def test_pair_decoder_reproduces_all_historical_answers(self):
        path = cfg.QWEN_DIR / 'outputs/qwen3_0p6b_abc/run04_r16/C/predictions_with_scores.csv'
        original = pd.read_csv(path, keep_default_na=False)
        decoded = pair_aware_decode(original.drop(columns='pair_prediction'))
        self.assertTrue(np.array_equal(decoded['pair_prediction'], original['pair_prediction']))
        singleton = original.groupby('pair_id')['instance_id'].transform('size') == 1
        self.assertTrue(decoded.loc[singleton, 'prediction'].equals(decoded.loc[singleton, 'pair_prediction']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
