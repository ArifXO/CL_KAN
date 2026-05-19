"""Tests for ResidualFastKANWarp (Stage 5)."""

import pytest
import torch
import torch.nn.functional as F

from src.models.kan.residual_warp import ResidualFastKANWarp


# --- fixtures ---

@pytest.fixture
def warp() -> ResidualFastKANWarp:
    return ResidualFastKANWarp(
        input_dim=16,
        hidden_dim=12,
        num_layers=2,
        num_centers=5,
        alpha_init=0.1,
        learnable_alpha=True,
        clamp_alpha=True,
        clamp_max=0.2,
    )


# --- shape and output contract ---

def test_output_shape(warp: ResidualFastKANWarp) -> None:
    z = torch.randn(8, 16)
    out = warp(z)
    assert out.shape == z.shape, f"Expected shape {z.shape}, got {out.shape}"


def test_output_unit_norm(warp: ResidualFastKANWarp) -> None:
    torch.manual_seed(0)
    z = torch.randn(8, 16)
    out = warp(z)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-6), (
        f"Output rows must be unit-norm; norms: {norms}"
    )


def test_output_finite(warp: ResidualFastKANWarp) -> None:
    torch.manual_seed(1)
    z = torch.randn(16, 16)
    assert torch.isfinite(warp(z)).all(), "Forward output must be finite"


# --- identity: alpha=0 → normalize(z) ---

def test_identity_at_alpha_zero() -> None:
    """Non-unit-norm input with alpha=0 must equal normalize(z) exactly."""
    warp_fixed = ResidualFastKANWarp(
        input_dim=16,
        hidden_dim=12,
        alpha_init=0.0,
        learnable_alpha=False,
        clamp_alpha=True,
    )
    torch.manual_seed(2)
    # Deliberately not unit-norm — tests that normalization is the only transform.
    z = torch.randn(6, 16) * 3.0
    out = warp_fixed(z)
    expected = F.normalize(z, dim=-1)
    assert torch.allclose(out, expected, atol=1e-6), (
        "With alpha=0 output must equal normalize(z)"
    )


# --- gradient flow ---

def test_gradient_flows_through_z(warp: ResidualFastKANWarp) -> None:
    torch.manual_seed(3)
    z = torch.randn(8, 16, requires_grad=True)
    warp(z).sum().backward()
    assert z.grad is not None, "Gradient must reach input z"
    assert torch.isfinite(z.grad).all(), "z.grad must be finite"


def test_gradient_flows_through_kan_params(warp: ResidualFastKANWarp) -> None:
    """With alpha > 0, KAN parameters must receive non-zero gradients."""
    torch.manual_seed(4)
    z = torch.randn(8, 16)
    warp(z).sum().backward()
    rbf_grad = warp.kan.layers[0].rbf_weight.grad
    assert rbf_grad is not None, "KAN rbf_weight must have a gradient"
    assert torch.isfinite(rbf_grad).all(), "KAN gradient must be finite"
    assert rbf_grad.abs().max() > 0, "KAN gradient must be non-zero (alpha > 0)"


def test_gradient_flows_through_learnable_alpha(warp: ResidualFastKANWarp) -> None:
    torch.manual_seed(5)
    z = torch.randn(8, 16)
    warp(z).sum().backward()
    assert warp.alpha.grad is not None, "Learnable alpha must receive a gradient"
    assert torch.isfinite(warp.alpha.grad), "alpha.grad must be finite"


# --- alpha: learnable vs fixed ---

def test_learnable_alpha_in_parameters() -> None:
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=6, alpha_init=0.0, learnable_alpha=True
    )
    param_names = [n for n, _ in warp.named_parameters()]
    assert "alpha" in param_names, "alpha must be an nn.Parameter when learnable=True"


def test_fixed_alpha_not_in_parameters() -> None:
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=6, alpha_init=0.0, learnable_alpha=False
    )
    param_names = [n for n, _ in warp.named_parameters()]
    assert "alpha" not in param_names, "alpha must NOT be a parameter when learnable=False"
    buffer_names = [n for n, _ in warp.named_buffers()]
    assert "alpha" in buffer_names, "alpha must be a buffer when learnable=False"


# --- clamp behaviour ---

def test_clamp_caps_effective_alpha() -> None:
    """Setting alpha beyond clamp_max must produce same result as alpha=clamp_max."""
    torch.manual_seed(6)
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=6, alpha_init=0.0,
        learnable_alpha=True, clamp_alpha=True, clamp_max=0.2,
    )
    z = torch.randn(4, 8)

    # Manually push alpha above the clamp ceiling
    warp.alpha.data.fill_(0.5)
    out_over = warp(z)

    warp.alpha.data.fill_(0.2)
    out_at_max = warp(z)

    assert torch.allclose(out_over, out_at_max, atol=1e-6), (
        "With clamp_alpha=True, alpha=0.5 should produce the same result as alpha=0.2"
    )


def test_no_clamp_allows_large_alpha() -> None:
    """Without clamping, alpha=0.5 must differ from alpha=0.2."""
    torch.manual_seed(7)
    warp = ResidualFastKANWarp(
        input_dim=8, hidden_dim=6, alpha_init=0.0,
        learnable_alpha=True, clamp_alpha=False,
    )
    z = torch.randn(4, 8)

    warp.alpha.data.fill_(0.5)
    out_large = warp(z)

    warp.alpha.data.fill_(0.2)
    out_small = warp(z)

    assert not torch.allclose(out_large, out_small, atol=1e-4), (
        "Without clamp, alpha=0.5 and alpha=0.2 must produce different outputs"
    )


# --- validation errors (Rule 9) ---

def test_wrong_ndim_raises(warp: ResidualFastKANWarp) -> None:
    with pytest.raises(ValueError, match="2D"):
        warp(torch.randn(4, 16, 1))


def test_wrong_input_dim_raises(warp: ResidualFastKANWarp) -> None:
    with pytest.raises(ValueError, match="input_dim"):
        warp(torch.randn(4, 8))  # warp expects 16


def test_negative_alpha_init_raises() -> None:
    with pytest.raises(ValueError, match="alpha_init"):
        ResidualFastKANWarp(input_dim=8, hidden_dim=6, alpha_init=-0.1)


def test_alpha_init_exceeds_clamp_max_raises() -> None:
    with pytest.raises(ValueError, match="clamp_max"):
        ResidualFastKANWarp(
            input_dim=8, hidden_dim=6,
            alpha_init=0.5, clamp_alpha=True, clamp_max=0.2,
        )


def test_invalid_input_dim_raises() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        ResidualFastKANWarp(input_dim=0, hidden_dim=6)


def test_invalid_clamp_max_raises() -> None:
    with pytest.raises(ValueError, match="clamp_max"):
        ResidualFastKANWarp(input_dim=8, hidden_dim=6, clamp_max=0.0)
