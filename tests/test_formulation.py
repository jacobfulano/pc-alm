from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from pcalm.inference import (
    Schedule,
    bp_loss,
    constraint_residuals,
    method_grad,
    run_pc,
    run_pcalm,
)
from pcalm.model import activation_fn, forward, init_params, model_scales, skip_mask
from pcalm.training import adam_learning_rate

DEVICE = torch.device(os.environ.get("PCALM_TEST_DEVICE", "cpu"))


def small_case():
    generator = torch.Generator(device=DEVICE.type).manual_seed(0)
    params = init_params(
        generator, depth=4, width=5, input_dim=3, output_dim=2, device=DEVICE
    )
    scales = model_scales(width=5, depth=4, input_dim=3)
    skips = skip_mask(4)
    phi = activation_fn("tanh")
    data_generator = torch.Generator(device=DEVICE.type).manual_seed(1)
    x = torch.randn((7, 3), generator=data_generator, device=DEVICE)
    y = torch.nn.functional.one_hot(torch.arange(7, device=DEVICE) % 2, 2).float()
    return params, scales, skips, phi, x, y


def assert_trees_close(actual, expected, *, atol=2e-5, rtol=2e-4):
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


def test_configured_test_device_is_available():
    if DEVICE.type == "cuda":
        assert torch.cuda.is_available()


def test_constraints_are_hidden_edges_only():
    params, scales, skips, phi, x, _ = small_case()
    free = [torch.zeros((x.shape[0], 5), device=DEVICE) for _ in range(3)]
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    assert len(residuals) == len(params) - 1
    assert all(residual.shape == (x.shape[0], 5) for residual in residuals)
    assert all(residual.device.type == DEVICE.type for residual in residuals)


def test_pc_has_zero_duals():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pc(
        params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=2, phi=phi
    )
    assert all(torch.allclose(dual, torch.zeros_like(dual)) for dual in duals)


def test_pcalm_alpha_zero_matches_pc_gradient():
    params, scales, skips, phi, x, y = small_case()
    pc_schedule = Schedule(family="pc", budget=3)
    alm_schedule = Schedule(family="pcalm", budget=3, alpha=0.0)
    g_pc = method_grad(
        params, scales, skips, x, y, pc_schedule, state_lr=0.1, rho=1.0, phi=phi
    )
    g_alm = method_grad(
        params, scales, skips, x, y, alm_schedule, state_lr=0.1, rho=1.0, phi=phi
    )
    assert_trees_close(g_pc, g_alm, atol=1e-5, rtol=1e-5)


def test_pcalm_duals_update_hidden_edges():
    params, scales, skips, phi, x, y = small_case()
    _, duals = run_pcalm(
        params,
        scales,
        skips,
        x,
        y,
        state_lr=0.1,
        rho=1.0,
        alpha=1.0,
        budget=2,
        inner_steps=1,
        weight_credit_timing="post_dual_energy",
        phi=phi,
    )
    assert len(duals) == len(params) - 1
    assert any(float(torch.linalg.vector_norm(dual)) > 0.0 for dual in duals)


def test_default_adam_lr_uses_width_depth_scaling():
    assert (
        adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=None)
        == 2e-3
    )
    assert (
        adam_learning_rate(width=64, depth=16, eta0=1e-3, gamma0=1.0, explicit_lr=5e-4)
        == 5e-4
    )


def test_inference_is_per_sample_batch_invariant():
    params, scales, skips, phi, x, y = small_case()
    kwargs = {
        "state_lr": 0.1,
        "rho": 1.0,
        "alpha": 1.0,
        "budget": 3,
        "inner_steps": 1,
        "weight_credit_timing": "pre_dual_energy",
        "phi": phi,
    }
    free_single, _ = run_pcalm(params, scales, skips, x[:1], y[:1], **kwargs)
    free_batch, _ = run_pcalm(params, scales, skips, x, y, **kwargs)
    for single, batch in zip(free_single, free_batch, strict=True):
        torch.testing.assert_close(single[0], batch[0], atol=1e-5, rtol=1e-5)


def test_torch_formulation_matches_pinned_jax_reference():
    fixture_path = Path(__file__).parent / "fixtures" / "jax_reference.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["upstream_commit"] == "660747f61a8a7e547c0ecd2c48c8883380a7d1f6"

    params = [
        torch.tensor(value, device=DEVICE, requires_grad=True)
        for value in fixture["params"]
    ]
    x = torch.tensor(fixture["x"], device=DEVICE)
    y = torch.tensor(fixture["y"], device=DEVICE)
    scales = model_scales(width=2, depth=3, input_dim=2)
    skips = skip_mask(3)
    phi = activation_fn("tanh")

    expected_forward = [
        torch.tensor(value, device=DEVICE) for value in fixture["forward"]
    ]
    assert_trees_close(forward(params, scales, skips, x, phi), expected_forward)
    torch.testing.assert_close(
        bp_loss(params, scales, skips, x, y, phi),
        torch.tensor(fixture["bp_loss"], device=DEVICE),
        atol=2e-5,
        rtol=2e-4,
    )

    schedules = {
        "bp": Schedule("bp", 0),
        "pc": Schedule("pc", 3),
        "pcalm": Schedule("pcalm", 3, alpha=1.0),
    }
    for name, schedule in schedules.items():
        actual = method_grad(
            params, scales, skips, x, y, schedule, state_lr=0.1, rho=1.0, phi=phi
        )
        expected = [
            torch.tensor(value, device=DEVICE) for value in fixture[f"{name}_grads"]
        ]
        assert_trees_close(actual, expected)

    pc_free, pc_duals = run_pc(
        params, scales, skips, x, y, state_lr=0.1, rho=1.0, steps=3, phi=phi
    )
    assert_trees_close(
        pc_free, [torch.tensor(value, device=DEVICE) for value in fixture["pc_free"]]
    )
    assert_trees_close(
        pc_duals, [torch.tensor(value, device=DEVICE) for value in fixture["pc_duals"]]
    )
    pcalm_free, pcalm_duals = run_pcalm(
        params,
        scales,
        skips,
        x,
        y,
        state_lr=0.1,
        rho=1.0,
        alpha=1.0,
        budget=3,
        inner_steps=1,
        weight_credit_timing="pre_dual_energy",
        phi=phi,
    )
    assert_trees_close(
        pcalm_free,
        [torch.tensor(value, device=DEVICE) for value in fixture["pcalm_free"]],
    )
    assert_trees_close(
        pcalm_duals,
        [torch.tensor(value, device=DEVICE) for value in fixture["pcalm_duals"]],
    )
