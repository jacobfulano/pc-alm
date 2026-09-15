from __future__ import annotations

import math
from collections.abc import Callable

import torch

Params = list[torch.Tensor]


def activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "linear":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    raise ValueError(f"unknown activation: {name}")


def model_scales(width: int, depth: int, input_dim: int) -> list[float]:
    if depth < 2:
        raise ValueError(
            "depth must include at least one hidden layer and one output layer"
        )
    return (
        [1.0 / math.sqrt(input_dim)]
        + [1.0 / math.sqrt(width * depth)] * (depth - 2)
        + [1.0 / width]
    )


def skip_mask(depth: int) -> tuple[bool, ...]:
    return tuple([False] + [True] * (depth - 2) + [False])


def init_params(
    generator: torch.Generator,
    *,
    depth: int,
    width: int,
    input_dim: int,
    output_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Params:
    layers: Params = []
    for layer_ix in range(depth):
        in_dim = input_dim if layer_ix == 0 else width
        out_dim = output_dim if layer_ix == depth - 1 else width
        weight = torch.randn(
            (out_dim, in_dim),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        layers.append(weight.requires_grad_(True))
    return layers


def block_pred(
    weight: torch.Tensor,
    scale: float,
    skip: bool,
    z_prev: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
    *,
    is_first: bool,
) -> torch.Tensor:
    inp = z_prev if is_first else phi(z_prev)
    pred = scale * (inp @ weight.T)
    if skip:
        pred = pred + z_prev
    return pred


def forward(
    params: Params,
    scales: list[float],
    skips: tuple[bool, ...],
    x: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    acts = []
    z_prev = x
    for layer_ix, weight in enumerate(params):
        z = block_pred(
            weight,
            scales[layer_ix],
            skips[layer_ix],
            z_prev,
            phi,
            is_first=(layer_ix == 0),
        )
        acts.append(z)
        z_prev = z
    return acts


def logits(
    params: Params,
    scales: list[float],
    skips: tuple[bool, ...],
    x: torch.Tensor,
    phi: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return forward(params, scales, skips, x, phi)[-1]
