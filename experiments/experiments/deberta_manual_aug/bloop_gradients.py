"""BLOOP更新的独立PyTorch实现：Row梯度EMA + 全局Pair正交投影。

方法：Hsieh et al., ICML2024, https://arxiv.org/abs/2402.02998
对应作者实现的init='grad', rescale=False, use_ratio_only=False。
"""
import math

import torch


class BloopGradients:
    def __init__(self, parameters, ema_decay=0.9, eps=1e-7):
        self.parameters = list(parameters)
        self.ema = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.ema_decay = ema_decay
        self.eps = eps
        self.updates = 0

    @torch.no_grad()
    def update_ema(self, row_gradients):
        # 第一批直接初始化为主梯度。None相当于该参数梯度为0。
        for average, gradient in zip(self.ema, row_gradients):
            if self.updates == 0:
                if gradient is not None:
                    average.copy_(gradient)
            else:
                average.mul_(self.ema_decay)
                if gradient is not None:
                    average.add_(gradient, alpha=1.0 - self.ema_decay)
        self.updates += 1

    @torch.no_grad()
    def assign(self, row_gradients, pair_gradients, mode='bloop'):
        """Pair梯度已经乘0.5；取得这些梯度的所有权，原地合成以节约显存。"""
        self.update_ema(row_gradients)
        # Row平方、Pair平方、EMA平方、Row·Pair、EMA·Pair、Row·EMA。
        values = torch.zeros(6, device=self.parameters[0].device, dtype=torch.float64)
        for row, pair, average in zip(row_gradients, pair_gradients, self.ema):
            values[2] += torch.linalg.vector_norm(average).double().square()
            if row is not None:
                values[0] += torch.linalg.vector_norm(row).double().square()
                values[5] += torch.dot(row.reshape(-1), average.reshape(-1)).double()
            if pair is not None:
                values[1] += torch.linalg.vector_norm(pair).double().square()
                values[4] += torch.dot(average.reshape(-1), pair.reshape(-1)).double()
                if row is not None:
                    values[3] += torch.dot(row.reshape(-1), pair.reshape(-1)).double()
        r2, p2, m2, rp, mp, rm = values.cpu().tolist()
        candidate = mp / (m2 + self.eps)
        coefficient = candidate if mode == 'bloop' else 0.0

        # g = 当前Row梯度 + 加权Pair梯度 - c*主梯度EMA。
        # 无论内积正负都投影；不能把最后一项误写成c*当前Row梯度。
        for parameter, row, pair, average in zip(
            self.parameters, row_gradients, pair_gradients, self.ema,
        ):
            gradient = row if row is not None else pair
            if gradient is not None:
                if row is not None and pair is not None:
                    gradient.add_(pair)
                if coefficient != 0:
                    gradient.add_(average, alpha=-coefficient)
            elif coefficient != 0:
                # 全局投影也可能在本批原梯度为0、但EMA非0的参数上产生分量。
                gradient = average.mul(-coefficient)
            parameter.grad = gradient

        projected_square = max(p2 - 2 * coefficient * mp + coefficient**2 * m2, 0.0)
        return {
            'ema_updates': self.updates,
            'row_grad_norm': math.sqrt(r2), 'weighted_pair_grad_norm': math.sqrt(p2),
            'row_ema_norm': math.sqrt(m2), 'row_pair_dot': rp, 'ema_pair_dot': mp,
            'row_pair_cosine': rp / math.sqrt(r2 * p2) if r2 > 0 and p2 > 0 else None,
            'ema_pair_cosine': mp / math.sqrt(m2 * p2) if m2 > 0 and p2 > 0 else None,
            'ema_row_cosine': rm / math.sqrt(m2 * r2) if m2 > 0 and r2 > 0 else None,
            'projection_coefficient': coefficient,
            'projection_applied': mode == 'bloop' and coefficient != 0,
            'projected_pair_norm': math.sqrt(projected_square),
            'ema_dot_projected_pair': mp - coefficient * m2,
            # 这些是原梯度内积，不是AdamW实际参数位移的保证。
            'row_dot_assigned_gradient': r2 + rp - coefficient * rm,
        }
