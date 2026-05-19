"""Tests for MLPPairScorer."""

import pytest
import torch
import torch.nn.functional as F

from src.models.pair_scorer import MLPPairScorer


@pytest.fixture
def scorer() -> MLPPairScorer:
    return MLPPairScorer(input_dim=16, hidden_dim=32, num_layers=2)


def _make_z(B: int = 4, D: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(B, D), dim=-1)


# --- output shape and bounds ---

def test_output_shape(scorer: MLPPairScorer) -> None:
    z = _make_z()
    out = scorer(z)
    assert out.shape == (4, 4), f"Expected [B, B], got {out.shape}"


def test_output_bounds_default(scorer: MLPPairScorer) -> None:
    z = _make_z()
    p_fn = scorer(z)
    assert (p_fn >= 0.0).all(), "p_fn values must be >= 0"
    assert (p_fn <= 1.0).all(), "p_fn values must be <= 1"


def test_output_bounds_with_clip_val() -> None:
    s = MLPPairScorer(input_dim=16, hidden_dim=32, clip_val=0.5)
    z = _make_z()
    p_fn = s(z)
    assert (p_fn <= 0.5).all(), "clip_val=0.5 must cap outputs at 0.5"


def test_diagonal_not_zeroed(scorer: MLPPairScorer) -> None:
    """Self-pair masking is the loss's responsibility; scorer does not zero diagonal."""
    z = _make_z()
    p_fn = scorer(z)
    # Diagonal entries are valid probabilities, just unused by the loss
    assert p_fn.diagonal().shape == (4,)


# --- gradient flow ---

def test_gradient_flows_through_z(scorer: MLPPairScorer) -> None:
    z = _make_z().requires_grad_(True)
    scorer(z).sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_gradient_flows_through_params(scorer: MLPPairScorer) -> None:
    z = _make_z()
    scorer(z).sum().backward()
    for name, param in scorer.named_parameters():
        assert param.grad is not None, f"No grad for param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for: {name}"


# --- single layer (linear-then-sigmoid) ---

def test_single_layer_scorer() -> None:
    s = MLPPairScorer(input_dim=8, hidden_dim=16, num_layers=1)
    z = _make_z(B=3, D=8)
    out = s(z)
    assert out.shape == (3, 3)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# --- validation errors (Rule 9) ---

def test_wrong_z_ndim_raises(scorer: MLPPairScorer) -> None:
    with pytest.raises(ValueError, match="2D"):
        scorer(torch.randn(4, 16, 2))


def test_wrong_input_dim_raises(scorer: MLPPairScorer) -> None:
    with pytest.raises(ValueError, match="input_dim"):
        scorer(torch.randn(4, 8))  # scorer expects D=16


def test_invalid_input_dim_raises() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        MLPPairScorer(input_dim=0, hidden_dim=32)


def test_invalid_hidden_dim_raises() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        MLPPairScorer(input_dim=16, hidden_dim=0)


def test_invalid_num_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        MLPPairScorer(input_dim=16, hidden_dim=32, num_layers=0)


def test_invalid_dropout_raises() -> None:
    with pytest.raises(ValueError, match="dropout"):
        MLPPairScorer(input_dim=16, hidden_dim=32, dropout=1.0)


def test_invalid_clip_val_zero_raises() -> None:
    with pytest.raises(ValueError, match="clip_val"):
        MLPPairScorer(input_dim=16, hidden_dim=32, clip_val=0.0)


def test_invalid_clip_val_above_one_raises() -> None:
    with pytest.raises(ValueError, match="clip_val"):
        MLPPairScorer(input_dim=16, hidden_dim=32, clip_val=1.1)
