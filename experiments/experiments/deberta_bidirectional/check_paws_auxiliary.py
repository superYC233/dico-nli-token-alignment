"""离线CPU检查PAWS筛选、二元损失与梯度合并；不执行参数更新。"""
import argparse
import io
import shutil
import sys
import tarfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project_paths import get_model_cache_dir
from experiments.deberta_bidirectional.check_alignment_joint import AlignmentToyTokenizer, toy_backbone
from experiments.deberta_bidirectional.check_bidirectional import toy_frame
from experiments.deberta_bidirectional.data import BidirectionalDataset, model_inputs
from experiments.deberta_bidirectional.alignment_joint_model import (
    AlignmentJointDeberta, build_model, save_weights, load_weights,
)
from experiments.deberta_bidirectional.prepare_paws_data import (
    filter_candidates, select_balanced, overlap_report, write_train_tsv, read_train_file,
)
from experiments.deberta_bidirectional.paws_auxiliary import (
    PawsDataset, equivalence_loss, interleaved_batches, train_epoch,
)
from experiments.deberta_bidirectional import train_deberta_paws as cfg


class LengthTokenizer:
    def __call__(self, first, second, **kwargs):
        return {'input_ids': [[1] * (len(a.split()) + len(b.split()) + 3)
                              for a, b in zip(first, second)]}


class GradientRecorder:
    """只记录梯度，step不改动任何权重。"""
    def __init__(self, model):
        self.model = model
        self.param_groups = [{'lr': cfg.LEARNING_RATE}]
        self.states, self.gradients = [], []

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)
        self.states.append(torch.get_rng_state().clone())

    def step(self):
        self.gradients.append({name: parameter.grad.detach().clone()
                               for name, parameter in self.model.named_parameters() if parameter.grad is not None})


def paws_frame():
    return pd.DataFrame({
        'source_id': ['positive', 'negative'], 'source_split': ['train', 'train'],
        'text1': ['a bright lantern', 'a guide follows a singer'],
        'text2': ['a lantern that is bright', 'a singer follows a guide'],
        'paws_label': [1, 0],
    })


def check_preparation(folder):
    examples = [
        ('a bright lantern', 'a lantern that is bright', 1),
        ('a lantern that is bright', 'a bright lantern', 1),
        ('a guide follows a singer', 'a singer follows a guide', 0),
        ('a quick arrival', 'a fast arrival', 1),
        ('people help birds', 'birds help people', 0),
        ('KNOWN PHRASE', 'new words', 0),
        ('held-out sentence', 'other words', 1),
        ('same', 'same', 1),
        ('conflict one', 'conflict two', 0),
        ('conflict two', 'conflict one', 1),
        (' '.join(['very'] * 200), 'long text', 0),
        ('', 'empty text', 1),
        ('a singer follows a guide', 'a guide follows a singer', 0),
    ]
    raw = pd.DataFrame(examples, columns=['sentence1', 'sentence2', 'label'])
    raw.insert(0, 'id', [str(index) for index in range(len(raw))])
    train = pd.DataFrame({'text1': ['Known phrase'], 'text2': ['training wording']})
    dev = pd.DataFrame({'text1': ['Held-out sentence'], 'text2': ['held-out wording']})
    candidates, rejected = filter_candidates(raw, LengthTokenizer(), train, dev)
    assert len(candidates) == 4 and len(rejected) == 9
    assert rejected.reason.value_counts().to_dict() == {
        'duplicate_pair': 2, 'conflicting_paws_labels': 2, 'dico_train_overlap': 1,
        'dico_dev_overlap': 1, 'identical_texts': 1, 'empty_text': 1, 'exceeds_max_length': 1,
    }
    chosen = select_balanced(candidates, per_class=2, seed=42)
    pd.testing.assert_frame_equal(chosen, select_balanced(candidates, per_class=2, seed=42))
    assert chosen.paws_label.value_counts().to_dict() == {0: 2, 1: 2}
    assert chosen.source_split.eq('train').all()
    assert all(value == 0 for value in overlap_report(chosen, train, dev).values())

    # 标注包含Dev/Test时，只取train的内容，不能把文件拼起来用。
    train_bytes = raw.to_csv(index=False, sep='\t').encode('utf-8')
    archive_path = folder / 'fixture.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, content in [('final/train.tsv', train_bytes), ('final/dev.tsv', b'NEVER READ'),
                              ('final/test.tsv', b'NEVER READ')]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    write_train_tsv(archive_path, folder / 'train.tsv')
    assert (folder / 'train.tsv').read_bytes() == train_bytes
    raw.to_parquet(folder / 'train.parquet', index=False)
    pd.testing.assert_frame_equal(read_train_file(folder / 'train.tsv'), read_train_file(folder / 'train.parquet'))
    print('筛选检查通过：只取train、正反去重、标签冲突、DiCo文本隔离、长度过滤、平衡抽样。')


def check_loss():
    logits = torch.tensor([[.1, -.2, .3, .2], [-.2, .5, .1, .0]], requires_grad=True)
    target = torch.tensor([1, 0])
    loss = equivalence_loss(logits, target)
    expected = F.binary_cross_entropy(torch.softmax(logits, dim=1)[:, 2], target.float())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 2] < 0 and (logits.grad[0, [0, 1, 3]] > 0).all()
    assert logits.grad[1, 2] > 0 and (logits.grad[1, [0, 1, 3]] < 0).all()
    swapped = logits.detach()[:, [1, 0, 2, 3]]
    torch.testing.assert_close(loss.detach(), equivalence_loss(swapped, target))
    extreme = torch.tensor([[1000., -1000., -1000., 0.], [0., 0., 1000., -1000.]], requires_grad=True)
    large_loss = equivalence_loss(extreme, target)
    large_loss.backward()
    assert torch.isfinite(large_loss) and torch.isfinite(extreme.grad).all()
    print('损失检查通过：等价概率BCE、稳定logsumexp、交换一致；负例没有被指定为N。')


def check_schedule(folder):
    schedule = list(interleaved_batches(list(range(12)), ['a', 'b'], every=4))
    assert [(step, auxiliary) for step, _, auxiliary in schedule if auxiliary is not None] == [(4, 'a'), (8, 'b'), (12, 'a')]
    class ForbiddenLoader:
        def __iter__(self):
            raise AssertionError('权重0时不应读取PAWS。')
    assert all(auxiliary is None for _, _, auxiliary in interleaved_batches(range(8), ForbiddenLoader(), enabled=False))

    tokenizer = AlignmentToyTokenizer()
    model = AlignmentJointDeberta(toy_backbone(), tokenizer, dimension=12)
    dico = BidirectionalDataset(toy_frame(), tokenizer, 16)
    dico_batch = cfg.base_cfg.collate_original_units([dico[i] for i in range(len(dico))])
    auxiliary_dataset = PawsDataset(paws_frame(), tokenizer, 16)
    assert set(auxiliary_dataset.labels.tolist()) == {0, 1}
    paws_loader = DataLoader(auxiliary_dataset, batch_size=2, shuffle=False,
                             collate_fn=cfg.base_cfg.collate_original_units,
                             generator=torch.Generator().manual_seed(100))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg.ROW_WEIGHTS))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    recorder = GradientRecorder(model)
    totals, exposure = train_epoch(model, [dico_batch] * 5, paws_loader, recorder, criterion,
                                   torch.device('cpu'), paws_weight=.1, paws_every=4)
    assert len(recorder.gradients) == 5 and totals['optimizer_steps'] == 5
    assert totals['row_ce_rows'] == 30 and totals['paws_rows'] == 2 and totals['paws_batches'] == 1
    assert exposure[0]['dico_step'] == 4
    assert all(torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters())

    # 核对第4次更新是两项梯度之和，没有在中间zero_grad或单独step。
    auxiliary = next(iter(paws_loader))
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(recorder.states[3])
    criterion(model(**model_inputs(dico_batch, torch.device('cpu'))), dico_batch['labels']).backward()
    (.1 * equivalence_loss(model(**model_inputs(auxiliary, torch.device('cpu'))), auxiliary['labels'])).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, recorder.gradients[3][name], rtol=1e-5, atol=1e-6)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        expected = model(**model_inputs(dico_batch, torch.device('cpu')))
    save_weights(model, folder / 'model.pth')
    restored = AlignmentJointDeberta(toy_backbone(), tokenizer, dimension=12).eval()
    load_weights(restored, folder / 'model.pth')
    with torch.inference_mode():
        actual = restored(**model_inputs(dico_batch, torch.device('cpu')))
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert all((ROOT / name).is_file() for name in cfg.source_hashes())
    print('训练路径检查通过：DiCo曝光不变、4:1取样、同一步合并梯度、没有参数更新、原结构完整保存重载。')


def check_real():
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, cache_dir=get_model_cache_dir(), revision=cfg.BASE_REVISION, local_files_only=True,
    )
    model = build_model(cfg.MODEL_NAME, cfg.BASE_REVISION, tokenizer, cfg.DIMENSION, cfg.HEAD_DROPOUT)
    dataset = PawsDataset(paws_frame(), tokenizer, cfg.MAX_LENGTH)
    batch = cfg.base_cfg.collate_original_units([dataset[i] for i in range(len(dataset))])
    model.train()
    loss = equivalence_loss(model(**model_inputs(batch, torch.device('cpu'))), batch['labels'])
    loss.backward()
    assert torch.isfinite(loss) and model.align_projection.weight.grad is not None
    assert model.deberta_model.embeddings.word_embeddings.weight.grad is not None
    print('真实NLI模型PAWS二元前向/梯度检查通过；使用的是测试夹具，未下载真实PAWS，也未更新参数。')


def main(real=False):
    torch.set_num_threads(4)
    artifacts = (ROOT / 'artifacts').resolve()
    artifacts.mkdir(exist_ok=True)
    folder = artifacts / f'paws_auxiliary_check_{uuid.uuid4().hex}'
    folder.mkdir()
    try:
        check_preparation(folder)
        check_loss()
        check_schedule(folder)
        if real:
            check_real()
    finally:
        target = folder.resolve()
        assert target.parent == artifacts and target.name.startswith('paws_auxiliary_check_')
        shutil.rmtree(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true')
    args = parser.parse_args()
    main(args.real)
