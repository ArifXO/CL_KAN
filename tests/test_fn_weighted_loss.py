"""Tests for FNWeightedInfoNCELoss (Rule 3: positive, negative, FN cases).

Rule 3 coverage:
  positive case — p_fn=0 → loss == standard InfoNCE
  negative case — p_fn=1 → all negatives removed → loss collapses to 0
  FN case       — mixed p_fn changes loss vs. baseline in expected direction
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses.fn_weighted_infonce import FNWeightedInfoNCELoss
from src.losses.infonce import InfoNCELoss
from src.models.pair_scorer import MLPPairScorer

_REQUIRED_KEYS = {
    "loss",
    "pos_sim_mean",
    "neg_sim_mean",
    "p_fn_mean",
    "p_fn_max",
    "downweighted_fraction",
    "temperature",
}


# --- fixtures ---

@pytest.fixture
def loss_fn() -> FNWeightedInfoNCELoss:
    return FNWeightedInfoNCELoss(temperature=0.1)


def _make_batch(B: int = 4, D: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    p_fn = torch.zeros(B, B)
    return z, p_fn


# --- output contract (Rule 7) ---

def test_output_is_dict(loss_fn: FNWeightedInfoNCELoss) -> None:
    z, p_fn = _make_batch()
    assert isinstance(loss_fn(z, p_fn), dict)


def test_output_has_required_keys(loss_fn: FNWeightedInfoNCELoss) -> None:
    z, p_fn = _make_batch()
    missing = _REQUIRED_KEYS - set(loss_fn(z, p_fn).keys())
    assert not missing, f"Missing keys: {missing}"


def test_loss_is_scalar(loss_fn: FNWeightedInfoNCELoss) -> None:
    z, p_fn = _make_batch()
    assert loss_fn(z, p_fn)["loss"].ndim == 0


def test_output_finite(loss_fn: FNWeightedInfoNCELoss) -> None:
    z, p_fn = _make_batch(seed=1)
    out = loss_fn(z, p_fn)
    for key in ("loss", "pos_sim_mean", "neg_sim_mean"):
        assert torch.isfinite(out[key]), f"{key} must be finite"


# --- Rule 3: positive case — p_fn = 0 recovers standard InfoNCE ---

def test_equivalence_with_zero_p_fn() -> None:
    """Rule 3 positive case: p_fn=zeros must reproduce InfoNCELoss exactly."""
    torch.manual_seed(0)
    B, D = 4, 32
    z = torch.randn(2 * B, D)
    p_fn = torch.zeros(B, B)

    tau = 0.1
    infonce_out = InfoNCELoss(temperature=tau)(z)
    fn_out = FNWeightedInfoNCELoss(temperature=tau)(z, p_fn)

    assert torch.allclose(fn_out["loss"], infonce_out["loss"], atol=1e-5), (
        f"FNWeighted(p_fn=0) loss={fn_out['loss'].item():.6f} != "
        f"InfoNCE loss={infonce_out['loss'].item():.6f}"
    )


# --- Rule 3: negative case — p_fn = 1 removes all negatives ---

def test_all_fn_weight_one_collapses_loss() -> None:
    """Rule 3 negative case: p_fn=1 for all pairs → denominator = positive only → loss ≈ 0."""
    torch.manual_seed(2)
    B, D = 4, 16
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    p_fn = torch.ones(B, B)

    out = FNWeightedInfoNCELoss(temperature=0.1)(z, p_fn)

    assert abs(out["loss"].item()) < 1e-4, (
        f"p_fn=1 removes all negatives; loss should be ~0, got {out['loss'].item():.6f}"
    )


# --- Rule 3: FN case — mixed p_fn changes loss in expected direction ---

def test_nonzero_p_fn_reduces_denominator() -> None:
    """Rule 3 FN case: higher p_fn should produce lower loss (weaker negatives)."""
    torch.manual_seed(3)
    B, D = 4, 16
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    tau = 0.1

    out_zero = FNWeightedInfoNCELoss(temperature=tau)(z, torch.zeros(B, B))
    out_half = FNWeightedInfoNCELoss(temperature=tau)(z, torch.full((B, B), 0.5))

    # Removing half the negative weight makes the loss easier → loss should be lower
    assert out_half["loss"].item() < out_zero["loss"].item(), (
        "FN downweighting (p_fn=0.5) should reduce loss vs p_fn=0"
    )


def test_p_fn_monotone_loss_reduction() -> None:
    """Increasing p_fn monotonically reduces loss (denominator shrinks)."""
    torch.manual_seed(4)
    B, D = 4, 16
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    tau = 0.1
    fn_loss = FNWeightedInfoNCELoss(temperature=tau)

    losses = [
        fn_loss(z, torch.full((B, B), v))["loss"].item()
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    for i in range(len(losses) - 1):
        assert losses[i] >= losses[i + 1], (
            f"Loss not monotonically non-increasing: {losses}"
        )


# --- gradient flow ---

def test_gradient_flows_through_z(loss_fn: FNWeightedInfoNCELoss) -> None:
    torch.manual_seed(5)
    B, D = 4, 16
    z_leaf = F.normalize(torch.randn(2 * B, D), dim=-1).detach().requires_grad_(True)
    p_fn = torch.zeros(B, B)
    loss_fn(z_leaf, p_fn)["loss"].backward()
    assert z_leaf.grad is not None
    assert torch.isfinite(z_leaf.grad).all()


def test_gradient_flows_through_scorer_params() -> None:
    """Grad must reach scorer params when p_fn is wired in-graph from scorer output."""
    torch.manual_seed(6)
    B, D = 4, 16
    scorer = MLPPairScorer(input_dim=D, hidden_dim=32)
    fn_loss = FNWeightedInfoNCELoss(temperature=0.1)

    z_leaf = torch.randn(2 * B, D).requires_grad_(True)
    p_fn = scorer(z_leaf[:B])          # in-graph; view-1 only
    fn_loss(z_leaf, p_fn)["loss"].backward()

    assert z_leaf.grad is not None
    for name, param in scorer.named_parameters():
        assert param.grad is not None, f"No grad for scorer param: {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for: {name}"


# --- diagnostics ---

def test_p_fn_mean_with_constant_matrix(loss_fn: FNWeightedInfoNCELoss) -> None:
    """p_fn_mean must equal the constant value when p_fn is uniform over negatives."""
    torch.manual_seed(7)
    B, D = 4, 16
    z = torch.randn(2 * B, D)
    val = 0.3
    p_fn = torch.full((B, B), val)
    out = loss_fn(z, p_fn)
    assert abs(out["p_fn_mean"].item() - val) < 1e-5, (
        f"Expected p_fn_mean={val}, got {out['p_fn_mean'].item()}"
    )


def test_downweighted_fraction_all_above_threshold(loss_fn: FNWeightedInfoNCELoss) -> None:
    """When all p_fn > 0.5, downweighted_fraction must be 1.0."""
    torch.manual_seed(8)
    B, D = 4, 16
    z = torch.randn(2 * B, D)
    p_fn = torch.full((B, B), 0.8)
    out = loss_fn(z, p_fn)
    assert abs(out["downweighted_fraction"].item() - 1.0) < 1e-5


def test_downweighted_fraction_none_above_threshold(loss_fn: FNWeightedInfoNCELoss) -> None:
    """When all p_fn <= 0.5, downweighted_fraction must be 0.0."""
    torch.manual_seed(9)
    B, D = 4, 16
    z = torch.randn(2 * B, D)
    p_fn = torch.full((B, B), 0.2)
    out = loss_fn(z, p_fn)
    assert abs(out["downweighted_fraction"].item() - 0.0) < 1e-5


def test_temperature_in_output_matches_init() -> None:
    fn = FNWeightedInfoNCELoss(temperature=0.2)
    z, p_fn = _make_batch()
    out = fn(z, p_fn)
    assert out["temperature"].item() == pytest.approx(0.2)


def test_max_fn_weight_caps_p_fn_effect() -> None:
    """max_fn_weight=0.5 should give same result as capping p_fn at 0.5."""
    torch.manual_seed(10)
    B, D = 4, 16
    z = torch.randn(2 * B, D)
    p_fn_high = torch.ones(B, B)  # p_fn=1

    # max_fn_weight=0.5 → effectively p_fn clamped to 0.5
    out_capped = FNWeightedInfoNCELoss(temperature=0.1, max_fn_weight=0.5)(z, p_fn_high)
    out_half = FNWeightedInfoNCELoss(temperature=0.1)(z, torch.full((B, B), 0.5))

    assert torch.allclose(out_capped["loss"], out_half["loss"], atol=1e-5), (
        "max_fn_weight=0.5 with p_fn=1 should equal p_fn=0.5 with max_fn_weight=1"
    )


# --- validation errors (Rule 9) ---

def test_wrong_z_ndim_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    with pytest.raises(ValueError, match="2D"):
        loss_fn(torch.randn(4, 16, 2), torch.zeros(2, 2))


def test_odd_batch_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    with pytest.raises(ValueError, match="even"):
        loss_fn(torch.randn(5, 16), torch.zeros(5, 5))


def test_too_small_batch_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    with pytest.raises(ValueError, match="B >= 2"):
        loss_fn(torch.randn(2, 16), torch.zeros(1, 1))


def test_wrong_p_fn_ndim_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    with pytest.raises(ValueError, match="2D"):
        loss_fn(torch.randn(8, 16), torch.zeros(4))


def test_p_fn_shape_mismatch_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        loss_fn(torch.randn(8, 16), torch.zeros(3, 3))


def test_p_fn_nan_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    p_fn = torch.zeros(4, 4)
    p_fn[0, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        loss_fn(torch.randn(8, 16), p_fn)


def test_p_fn_negative_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    p_fn = torch.zeros(4, 4)
    p_fn[0, 1] = -0.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        loss_fn(torch.randn(8, 16), p_fn)


def test_p_fn_above_one_raises(loss_fn: FNWeightedInfoNCELoss) -> None:
    p_fn = torch.zeros(4, 4)
    p_fn[0, 1] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        loss_fn(torch.randn(8, 16), p_fn)


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        FNWeightedInfoNCELoss(temperature=0.0)


def test_invalid_max_fn_weight_raises() -> None:
    with pytest.raises(ValueError, match="max_fn_weight"):
        FNWeightedInfoNCELoss(max_fn_weight=0.0)
    with pytest.raises(ValueError, match="max_fn_weight"):
        FNWeightedInfoNCELoss(max_fn_weight=1.1)
