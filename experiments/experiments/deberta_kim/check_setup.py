"""只检查数据、分词和局部知识，不加载真实 DeBERTa 主干，不训练。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_kim import config as cfg
from experiments.deberta_kim.data import read_data, build_feature_tensor, DiCoDataset, PairDataset
from experiments.deberta_kim.wordnet_resource import ensure_wordnet, resource_info
from experiments.deberta_kim.relation_features import word_positions, RELATION_NAMES


def main():
    ensure_wordnet()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), local_files_only=True,
    )
    assert tokenizer.is_fast
    train, dev = read_data()
    datasets = []
    for name, frame in [('Train', train), ('Dev', dev)]:
        features = build_feature_tensor(frame)
        dataset = DiCoDataset(frame, tokenizer, features, cfg.MAX_LENGTH)
        covered = sum(bool(row['word_pair_count']) for row in dataset.audit)
        # 提前编码的 token 必须与旧 Dataset 的输入完全相同。
        for index, row in enumerate(frame.to_dict('records')):
            old = tokenizer(row['text1'], row['text2'], truncation=True,
                            padding='max_length', max_length=cfg.MAX_LENGTH, return_tensors='pt')
            ids, mask, edges = dataset.examples[index]
            assert torch.equal(ids, old['input_ids'][0])
            assert torch.equal(mask, old['attention_mask'][0])
            first_sep = (ids == tokenizer.sep_token_id).nonzero()[0].item()
            last_sep = (ids == tokenizer.sep_token_id).nonzero()[-1].item()
            assert all(0 < i < first_sep < j < last_sep for i, j, _ in edges)
        print(f'{name}：{len(frame)} 行，20 维原特征，{covered} 行含局部词典关系。')
        datasets.append(dataset)
    pairs = PairDataset(train, tokenizer, datasets[0].features, cfg.MAX_LENGTH,
                        row_dataset=datasets[0])
    assert len(pairs) == 1282
    print('Real Pair：', len(pairs))
    encoded = tokenizer('A black car', 'A black vehicle', return_offsets_mapping=True)
    print('关系通道：', RELATION_NAMES)
    print('car 的 token 位置：', word_positions('A black car', encoded, 0)['car'])
    print('WordNet：', resource_info())
    print('检查通过。正式训练请运行 train_deberta_kim.py。')
    return tokenizer, datasets


if __name__ == '__main__':
    main()
