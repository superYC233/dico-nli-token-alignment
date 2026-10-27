"""普通L2 SAM的临时参数扰动；独立实现，不包装/替换AdamW。

公式：https://arxiv.org/abs/2010.01412
PyTorch实现参考：https://github.com/davda54/sam/blob/main/sam.py
"""
from contextlib import contextmanager

import torch


def capture_rng(device):
    states = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        states['cuda'] = torch.cuda.get_rng_state(device)
    return states


@contextmanager
def replay_rng(states, device):
    # 第二次前向复用第一次dropout，退出后恢复到第一次前向后的随机状态。
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(states['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(states['cuda'], device)
        yield


class SAMParameters:
    def __init__(self, parameters, rho=0.05, eps=1e-12):
        self.parameters = list(parameters)
        self.rho = rho
        self.eps = eps
        self.original = []

    @torch.no_grad()
    def perturb(self):
        # 全模型共用一个L2范数，不是每个tensor各扰动rho。
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.grad)
            for parameter in self.parameters if parameter.grad is not None
        ])).item()
        scale = self.rho / (norm + self.eps)
        for parameter in self.parameters:
            if parameter.grad is not None:
                self.original.append((parameter, parameter.detach().clone()))
                parameter.add_(parameter.grad, alpha=scale)
        return {'clean_grad_norm': norm, 'sam_radius': norm * scale}

    @torch.no_grad()
    def restore(self):
        # 恢复所有被扰动的参数，即使第二次反向中某个参数grad变成None。
        # copy_精确恢复，不用减去扰动，以免留下浮点误差。
        for parameter, original in self.original:
            parameter.copy_(original)
        self.original.clear()
