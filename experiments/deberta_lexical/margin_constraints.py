"""逐样本margin损坏约束；自包含的同时原始—对偶更新。"""
import torch
import torch.nn.functional as F


def classification_margins(logits, labels):
    own = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(labels, num_classes=4).bool(), -torch.inf)
    return own - other.max(dim=1).values


class MarginProtection:
    def __init__(self, reference_logits, labels, device,
                 margin_cap=1.0, initial_dual=0.25, dual_lr=0.1):
        # 只接收本轮Row结束后的Train输出；不备份完整模型参数。
        self.labels = labels.detach().cpu().clone()
        self.reference_logits = reference_logits.detach().float().cpu().clone()
        self.reference_margins = classification_margins(self.reference_logits, self.labels)
        self.row_correct = self.reference_logits.argmax(1).eq(self.labels)
        # 排除恰好同分的情况，保证被保护样本的参考下限严格大于0。
        self.protected = self.row_correct & self.reference_margins.gt(0)
        self.floors = torch.where(
            self.protected, self.reference_margins.clamp(max=margin_cap), 0.0,
        )
        self.device = device
        self.multipliers = torch.full((4,), initial_dual, device=device)
        self.dual_lr = dual_lr

    def counts(self, indices):
        indices = indices.cpu()
        selected = self.labels[indices][self.protected[indices]]
        return torch.bincount(selected, minlength=4).to(self.device).float()

    def constraint_parts(self, logits, indices, denominators):
        # denominators来自整个保护批次，不能对每个microbatch分别取平均。
        indices = indices.cpu()
        labels = self.labels[indices].to(logits.device)
        active = self.protected[indices].to(logits.device)
        floors = self.floors[indices].to(logits.device)
        current = classification_margins(logits.float(), labels)
        damage = (floors - current).clamp_min(0) * active
        totals = torch.zeros(4, device=logits.device).scatter_add(0, labels, damage)
        return totals / denominators.clamp_min(1)

    def primal_term(self, violations):
        # 本次模型更新始终使用lambda_t，dual_step在optimizer.step之后调用。
        return (self.multipliers * violations).sum()

    @torch.no_grad()
    def dual_step(self, violations):
        # g_c是逐样本非负损坏的类别均值，预算为0，改善不能抵消损坏。
        self.multipliers.add_(violations, alpha=self.dual_lr).clamp_min_(0)

    def audit(self, logits):
        logits = logits.detach().float().cpu()
        correct = logits.argmax(1).eq(self.labels)
        current = classification_margins(logits, self.labels)
        damage = (self.floors - current).clamp_min(0) * self.protected
        records = []
        for k, label in enumerate('FBEN'):
            mask = self.labels.eq(k)
            protected = mask & self.protected
            n = int(protected.sum())
            records.append({
                'class': label, 'train_rows': int(mask.sum()),
                'row_correct': int((mask & self.row_correct).sum()),
                'protected': n,
                'pair_correct': int((mask & correct).sum()),
                'protected_broken': int((protected & ~correct).sum()),
                'protected_retained': int((protected & correct).sum()),
                'previously_wrong_fixed': int((mask & ~self.row_correct & correct).sum()),
                'margin_violations': int((protected & damage.gt(1e-6)).sum()),
                'mean_margin_damage': float(damage[protected].mean()) if n else 0.0,
                'max_margin_damage': float(damage[protected].max()) if n else 0.0,
                'final_multiplier': float(self.multipliers[k]),
            })
        return records
