"""Row参数锚定：AdamW候选更新后的平方距离近端步，独立实现。"""
import math

import torch


class RowParameterAnchor:
    def __init__(self, model):
        # 每轮完整Row结束后新建。备份detach且独立存储，不从旧模型/Dev读取。
        self.references = {
            id(parameter): (name, parameter.detach().clone())
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }

    @torch.no_grad()
    def apply(self, optimizer, strength):
        """u是AdamW候选；theta=(u+lr*strength*anchor)/(1+lr*strength)。"""
        device = optimizer.param_groups[0]['params'][0].device
        squares = torch.zeros(2, 2, dtype=torch.float64, device=device)
        rates = {}
        for group in optimizer.param_groups:
            lr = group['lr']
            retention = 1.0 / (1.0 + lr * strength)
            for parameter in group['params']:
                name, reference = self.references[id(parameter)]
                index = int(name.startswith('lexical_output.'))
                rates[index] = (lr, retention)
                squares[index, 0] += torch.linalg.vector_norm(parameter - reference).double().square()
                if strength > 0:
                    # 对所有锚定参数回拉，即使该批Pair对某个参数没有梯度。
                    # lerp_在parameter==reference时保持不变，避免乘加重构的舍入漂移。
                    parameter.lerp_(reference, 1.0 - retention)
                squares[index, 1] += torch.linalg.vector_norm(parameter - reference).double().square()
        values = squares.cpu().tolist()
        records = {'anchor_strength': strength}
        for index, prefix in enumerate(['original', 'lexical']):
            before, after = values[index]
            lr, retention = rates[index]
            records.update({
                f'{prefix}_lr': lr, f'{prefix}_retention': retention,
                f'{prefix}_distance_before_pull': math.sqrt(before),
                f'{prefix}_distance_after_pull': math.sqrt(after),
            })
        before = sum(item[0] for item in values)
        after = sum(item[1] for item in values)
        records.update(anchor_distance_before_pull=math.sqrt(before),
                       anchor_distance_after_pull=math.sqrt(after),
                       anchor_penalty_before_pull=0.5 * strength * before,
                       anchor_penalty_after_pull=0.5 * strength * after)
        return records
