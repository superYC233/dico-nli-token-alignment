"""SiFT适配：归一化embedding、一步扰动搜索、四类分布正则。

Based on Microsoft DeBERTa/DeBERTa/sift/sift.py (MIT).
Copyright (c) Microsoft Corporation. See SIFT_LICENSE.txt.
Source: https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/sift/sift.py
"""
from contextlib import contextmanager
import weakref

import torch
import torch.nn.functional as F


def capture_rng(device):
    state = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


@contextmanager
def replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda'], device)
        yield


def symmetric_kl(first, second):
    # 与官方双向stop-gradient交叉熵具有相同梯度，扣除无梯度熵常数便于读日志。
    log_first = F.log_softmax(first.float(), dim=-1)
    log_second = F.log_softmax(second.float(), dim=-1)
    p, q = log_first.exp().detach(), log_second.exp().detach()
    return ((p * (log_first.detach() - log_second)).sum(-1)
            + (q * (log_second.detach() - log_first)).sum(-1)).mean()


class SiFT:
    def __init__(self, model, init_perturbation=1e-2, step_size=1e-4, noise_seed=142,
                 normalization_eps=1e-7):
        self.model_ref = weakref.ref(model)
        self.init_perturbation = init_perturbation
        self.step_size = step_size
        self.noise_seed = noise_seed
        self.normalization_eps = normalization_eps
        self.generator = None
        self.active = False
        self.cached = None
        self.delta = None
        self.last_stats = {}
        self.handle = model.deberta_model.embeddings.LayerNorm.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if self.active:
            # 与官方实现一致：扰动支路从缓存的归一化embedding开始；该输入detach。
            value = (self.cached + self.delta).to(inputs[0].dtype)
        else:
            value = F.layer_norm(inputs[0].float(), (inputs[0].shape[-1],),
                                 eps=self.normalization_eps).to(inputs[0].dtype)
            self.cached = value.detach()
        return (value,) + inputs[1:]

    def clear(self):
        self.active = False
        self.cached = None
        self.delta = None

    @contextmanager
    def search_only_delta(self):
        # 搜索扰动时只求delta梯度；结束后恢复原冻结状态，不改变正式训练参数范围。
        parameters = list(self.model_ref().parameters())
        flags = [p.requires_grad for p in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)

    def loss(self, clean_logits, forward, attention_mask, rng_state):
        device = clean_logits.device
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.noise_seed)
        mask = attention_mask.unsqueeze(-1).to(self.cached.dtype)
        noise = torch.randn(self.cached.shape, generator=self.generator, device=device,
                            dtype=torch.float32) * self.init_perturbation
        noise = noise.clamp(-2 * self.init_perturbation, 2 * self.init_perturbation) * mask
        self.delta = noise.detach().requires_grad_(True)
        self.active = True
        try:
            with self.search_only_delta():
                with replay_rng(rng_state, device):
                    search_logits = forward()
                search_loss = F.kl_div(
                    F.log_softmax(search_logits.float(), dim=-1),
                    F.softmax(clean_logits.detach().float(), dim=-1), reduction='batchmean',
                )
                gradient = torch.autograd.grad(search_loss, self.delta)[0]
            # 官方按每个token的最大绝对梯度缩放，做一次上升；不加入额外PGD循环。
            direction = gradient / (gradient.abs().amax(dim=-1, keepdim=True) + 1e-6)
            self.delta = (noise + self.step_size * direction * mask).detach()
            self.last_stats = {'search_kl': search_loss.detach().item(),
                               'delta_max': self.delta.abs().max().item(),
                               'delta_gradient_norm': gradient.norm().item()}
            with replay_rng(rng_state, device):
                perturbed_logits = forward()
            return symmetric_kl(perturbed_logits, clean_logits)
        finally:
            self.clear()

    def remove(self):
        self.clear()
        self.handle.remove()
