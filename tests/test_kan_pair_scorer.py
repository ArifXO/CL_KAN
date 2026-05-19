"""Tests for KANPairScorer (Stage 7).

Mirrors test_pair_scorer.py so MLP and KAN scorers are verified against the
same contract. Additional tests cover:
  - parameter count comparison vs MLPPairScorer (Rule 1)
  - interchangeability inside FNWeightedInfoNCELoss (acceptance criterion)
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.pair_scorer import KANPairScorer, MLPPairScorer


@pytest.fixture
def scorer() -> KANPairScorer:
    return KANPairScorer(input_dim=16, hidden_dim=4, num_layers=2, num_centers=8)


def _make_z(B: int = 4, D: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(B, D), dim=-1)


# --- output shape and bounds ---

def test_output_shape(scorer: KANPairScorer) -> None:
    z = _make_z()
    out = scorer(z)
    assert out.shape == (4, 4), f"Expected [B, B], got {out.shape}"


def test_output_bounds_default(scorer: KANPairScorer) -> None:
    z = _make_z()
    p_fn = scorer(z)
    assert (p_fn >= 0.0).all(), "p_fn values must be >= 0"
    assert (p_fn <= 1.0).all(), "p_fn values must be <= 1"


def test_output_bounds_with_clip_val() -> None:
    s = KANPairScorer(input_dim=16, hidden_dim=4, clip_val=0.5)
    z = _make_z()
    p_fn = s(z)
    assert (p_fn <= 0.5).all(), "clip_val=0.5 must cap outputs at 0.5"


def test_diagonal_not_zeroed(scorer: KANPairScorer) -> None:
    """Self-pair masking is the loss's responsibility; scorer does not zero diagonal."""
    z = _make_z()
    p_fn = scorer(z)
    assert p_fn.diagonal().shape == (4,)


# --- gradient flow ---

def test_gradient_flows_through_z(scorer: KANPairScorer) -> None:
    z = _make_z().requires_grad_(True)
    scorer(z).sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_gradient_flows_through_params(scorer: KANPairScorer) -> None:
    z = _make_z()
    scorer(z).sum().backward()
    for name, param in scorer.named_parameters():
        assert param.grad is not None, f"No grad for param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for: {name}"


# --- single layer (single KAN layer + sigmoid) ---

def test_single_layer_scorer() -> None:
    s = KANPairScorer(input_dim=8, hidden_dim=4, num_layers=1)
    z = _make_z(B=3, D=8)
    out = s(z)
    assert out.shape == (3, 3)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# --- validation errors (Rule 9) ---

def test_wrong_z_ndim_raises(scorer: KANPairScorer) -> None:
    with pytest.raises(ValueError, match="2D"):
        scorer(torch.randn(4, 16, 2))


def test_wrong_input_dim_raises(scorer: KANPairScorer) -> None:
    with pytest.raises(ValueError, match="input_dim"):
        scorer(torch.randn(4, 8))  # scorer expects D=16


def test_invalid_input_dim_raises() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        KANPairScorer(input_dim=0, hidden_dim=4)


def test_invalid_hidden_dim_raises() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        KANPairScorer(input_dim=16, hidden_dim=0)


def test_invalid_num_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        KANPairScorer(input_dim=16, hidden_dim=4, num_layers=0)


def test_invalid_clip_val_zero_raises() -> None:
    with pytest.raises(ValueError, match="clip_val"):
        KANPairScorer(input_dim=16, hidden_dim=4, clip_val=0.0)


def test_invalid_clip_val_above_one_raises() -> None:
    with pytest.raises(ValueError, match="clip_val"):
        KANPairScorer(input_dim=16, hidden_dim=4, clip_val=1.1)


# --- Rule 1: parameter count comparison with MLPPairScorer ---

def test_parameter_count_comparison() -> None:
    """Rule 1: KAN and MLP scorers must be parameter-matched within 15 %.

    Canonical matched pair at D=16, L=2:
        MLPPairScorer(H=32)          = 1 089 params
        KANPairScorer(H=4, K=8)      = 1 193 params  (~10 % more)
    """
    mlp = MLPPairScorer(input_dim=16, hidden_dim=32, num_layers=2)
    kan = KANPairScorer(input_dim=16, hidden_dim=4, num_layers=2, num_centers=8)

    mlp_params = sum(p.numel() for p in mlp.parameters())
    kan_params = sum(p.numel() for p in kan.parameters())

    ratio = abs(mlp_params - kan_params) / mlp_params
    assert ratio < 0.15, (
        f"KAN ({kan_params}) and MLP ({mlp_params}) param counts differ by "
        f"{ratio:.1%} — must be within 15 % for Rule 1 parity"
    )


# --- interchangeability: KAN scorer inside FNWeightedInfoNCELoss ---

def test_interchangeable_with_fn_loss_gradient() -> None:
    """KAN scorer plugged into FNWeightedInfoNCELoss must produce finite grads."""
    from src.losses.fn_weighted_infonce import FNWeightedInfoNCELoss

    torch.manual_seed(99)
    B, D = 4, 16
    scorer = KANPairScorer(input_dim=D, hidden_dim=4, num_layers=2, num_centers=8)
    fn_loss = FNWeightedInfoNCELoss(temperature=0.1)

    z_leaf = torch.randn(2 * B, D).requires_grad_(True)
    p_fn = scorer(z_leaf[:B])           # in-graph; view-1 only
    fn_loss(z_leaf, p_fn)["loss"].backward()

    assert z_leaf.grad is not None
    for name, param in scorer.named_parameters():
        assert param.grad is not None, f"No grad for scorer param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for: {name}"


def test_kan_and_mlp_scorers_same_output_contract() -> None:
    """Both scorers fed into the same loss must produce finite scalar losses."""
    from src.losses.fn_weighted_infonce import FNWeightedInfoNCELoss

    torch.manual_seed(42)
    B, D = 4, 16
    fn_loss = FNWeightedInfoNCELoss(temperature=0.1)
    z = torch.randn(2 * B, D)

    mlp_scorer = MLPPairScorer(input_dim=D, hidden_dim=32)
    kan_scorer = KANPairScorer(input_dim=D, hidden_dim=4, num_centers=8)

    for name, scorer in [("MLP", mlp_scorer), ("KAN", kan_scorer)]:
        p_fn = scorer(z[:B])
        out = fn_loss(z, p_fn)
        assert torch.isfinite(out["loss"]), f"{name} scorer produced non-finite loss"
        assert out["loss"].ndim == 0, f"{name} scorer loss must be a scalar"
