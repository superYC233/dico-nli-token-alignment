"""PAWS的非等价不等于N：只监督E与F/B/N合并后的概率。"""
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.deberta_bidirectional.data import LABELS, BidirectionalDataset, model_inputs


class PawsDataset(BidirectionalDataset):
    def __init__(self, frame, tokenizer, max_length=128):
        if not frame.paws_label.isin([0, 1]).all() or not frame.source_split.eq('train').all():
            raise ValueError('PAWS辅助样本必须来自train，标签只能为二元0/1。')
        super().__init__(frame[['text1', 'text2']], tokenizer, max_length)
        self.frame = frame.reset_index(drop=True)
        # 这里的labels是PAWS二元标签，绝不经过DiCo四类标签映射。
        self.labels = torch.tensor(frame.paws_label.tolist(), dtype=torch.long)


def equivalence_loss(logits, binary_labels):
    # logits顺序F/B/E/N。把F/B/N合并成“非等价”，而非把负例指定为N。
    not_equivalent = torch.logsumexp(logits[:, [0, 1, 3]].float(), dim=-1)
    equivalent = logits[:, 2].float()
    binary_logits = torch.stack([not_equivalent, equivalent], dim=-1)
    return F.cross_entropy(binary_logits, binary_labels.long())


def interleaved_batches(dico_loader, paws_loader, every=4, enabled=True):
    paws_iterator = None
    for step, batch in enumerate(dico_loader, start=1):
        auxiliary = None
        if enabled and step % every == 0:
            if paws_iterator is None:
                paws_iterator = iter(paws_loader)
            try:
                auxiliary = next(paws_iterator)
            except StopIteration:
                paws_iterator = iter(paws_loader)
                auxiliary = next(paws_iterator)
        yield step, batch, auxiliary


def train_epoch(model, dico_loader, paws_loader, optimizer, criterion, device,
                paws_weight=0.1, paws_every=4):
    model.train()
    started = time.perf_counter()
    row_sum, row_weight_sum, auxiliary_sum, objective_sum = 0.0, 0.0, 0.0, 0.0
    row_counts = torch.zeros(4, dtype=torch.long)
    auxiliary_counts = torch.zeros(2, dtype=torch.long)
    row_views, auxiliary_views, auxiliary_batches = 0, 0, 0
    exposure = []
    sequence = interleaved_batches(dico_loader, paws_loader, paws_every, paws_weight > 0)
    for step, batch, auxiliary in tqdm(sequence, total=len(dico_loader), desc='DiCo四类CE + PAWS等价监督'):
        optimizer.zero_grad(set_to_none=True)
        logits = model(**model_inputs(batch, device))
        labels = batch['labels'].to(device)
        row_loss = criterion(logits, labels)
        row_value = row_loss.item()
        weight = criterion.weight[labels].sum().item()
        row_loss.backward()
        row_sum += row_value * weight
        row_weight_sum += weight
        row_counts += torch.bincount(batch['labels'], minlength=4)
        row_views += len(batch['input_ids'])
        del logits, row_loss, labels

        auxiliary_value = 0.0
        if auxiliary is not None:
            # Row图先释放，再跑PAWS，避免同时保留两份encoder激活。
            # 两种梯度合入同一次AdamW更新，没有仅PAWS的优化器更新。
            logits = model(**model_inputs(auxiliary, device))
            auxiliary_loss = equivalence_loss(logits, auxiliary['labels'].to(device))
            auxiliary_value = auxiliary_loss.item()
            (paws_weight * auxiliary_loss).backward()
            auxiliary_sum += auxiliary_value * len(auxiliary['labels'])
            auxiliary_counts += torch.bincount(auxiliary['labels'], minlength=2)
            auxiliary_views += len(auxiliary['input_ids'])
            auxiliary_batches += 1
            exposure.append({'dico_step': step, 'paws_source_ids': '|'.join(
                str(value) for value in paws_loader.dataset.frame.iloc[
                    auxiliary['source_rows'].tolist()
                ].source_id.tolist()
            ), 'paws_equivalent': int(auxiliary['labels'].eq(1).sum()),
                'paws_not_equivalent': int(auxiliary['labels'].eq(0).sum())})
            del logits, auxiliary_loss
        optimizer.step()
        objective_sum += row_value + paws_weight * auxiliary_value

    auxiliary_rows = int(auxiliary_counts.sum())
    record = {
        'row_loss': row_sum / row_weight_sum,
        'paws_loss': auxiliary_sum / auxiliary_rows if auxiliary_rows else 0.0,
        'mean_update_objective': objective_sum / len(dico_loader),
        'row_ce_rows': int(row_counts.sum()), 'paws_rows': auxiliary_rows,
        'paws_batches': auxiliary_batches, 'paws_weight': paws_weight,
        'paws_equivalent_rows': int(auxiliary_counts[1]), 'paws_not_equivalent_rows': int(auxiliary_counts[0]),
        'dico_encoded_views': row_views, 'paws_encoded_views': auxiliary_views,
        'optimizer_steps': len(dico_loader), 'learning_rate': optimizer.param_groups[0]['lr'],
        'train_seconds': time.perf_counter() - started,
    }
    record.update({f'seen_{label}': count for label, count in zip(LABELS, row_counts.tolist())})
    return record, exposure
