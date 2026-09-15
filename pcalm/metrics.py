from __future__ import annotations

import torch


def mse_ce_accuracy(logits: torch.Tensor, y: torch.Tensor):
    mse = 0.5 * torch.mean(torch.sum((logits - y) ** 2, dim=-1))
    ce = -torch.mean(torch.sum(y * torch.log_softmax(logits, dim=-1), dim=-1))
    acc = torch.mean((torch.argmax(logits, dim=-1) == torch.argmax(y, dim=-1)).float())
    return mse, ce, acc


def tree_l2(tree: list[torch.Tensor]) -> torch.Tensor:
    if not tree:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack([torch.sum(x * x) for x in tree]).sum())


def tree_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    if not a:
        return torch.tensor(0.0)
    return torch.stack([torch.sum(x * y) for x, y in zip(a, b, strict=True)]).sum()


def tree_cos(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    denominator = torch.clamp(tree_l2(a) * tree_l2(b), min=1e-30)
    return tree_dot(a, b) / denominator
