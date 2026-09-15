from __future__ import annotations

import torch

from .model import Params


def make_adam(params: Params, learning_rate: float) -> torch.optim.Adam:
    return torch.optim.Adam(params, lr=learning_rate)
