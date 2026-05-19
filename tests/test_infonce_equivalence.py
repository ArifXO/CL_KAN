"""InfoNCE loss: dict contract, numerical stability, gradient flow, hand-computed equivalence."""

import math

import pytest
import torch

from src.losses.infonce import InfoNCELoss


REQUIRED_KEYS = {"loss", "pos_sim_mean", "neg_sim_mean", "temperature"}


@pytest.fixture
def loss_fn() -> InfoNCELoss:
    return InfoNCELoss(temperature=0.1, normalize_embeddings=True)


# --- dict contract (Rule 7) ---

def test_returns_dict(loss_fn: InfoNCELoss) -> None:
    torch.manual_seed(0)
    z = torch.randn(8, 64)
    out = loss_fn(z)
    assert isinstance(out, dict), "forward() must return dict (Rule 7)"
    missing = REQUIRED_KEYS - out.keys()
    assert not missing, f"missing keys: {missing}"


def test_loss_is_scalar(loss_fn: InfoNCELoss) -> None:
    torch.manual_seed(0)
    z = torch.randn(8, 64)
    out = loss_fn(z)
    assert out["loss"].shape == torch.Size([]), "loss must be a 0-dim scalar"


# --- numerical stability ---

def test_loss_finite(loss_fn: InfoNCELoss) -> None:
    torch.manual_seed(1)
    z = torch.randn(16, 128)
    assert torch.isfinite(loss_fn(z)["loss"])


@pytest.mark.parametrize("tau", [0.05, 0.07, 0.1, 0.5])
def test_finite_across_temperature_range(tau: float) -> None:
    fn = InfoNCELoss(temperature=tau)
    torch.manual_seed(2)
    z = torch.randn(16, 64)
    assert torch.isfinite(fn(z)["loss"]), f"loss not finite at τ={tau}"


# --- validation errors (Rule 9) ---

def test_temperature_zero_raises() -> None:
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=0.0)


def test_negative_temperature_raises() -> None:
    with pytest.raises(ValueError):
        InfoNCELoss(temperature=-0.1)


def test_odd_batch_raises(loss_fn: InfoNCELoss) -> None:
    with pytest.raises(ValueError):
        loss_fn(torch.randn(5, 64))


def test_single_pair_raises(loss_fn: InfoNCELoss) -> None:
    """B=1 has no negatives; neg_sim_mean would be nan — must raise (Rule 9)."""
    with pytest.raises(ValueError):
        loss_fn(torch.randn(2, 64))


def test_wrong_ndim_raises(loss_fn: InfoNCELoss) -> None:
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4, 64, 1))


# --- gradient flow ---

def test_gradient_flows() -> None:
    fn = InfoNCELoss(temperature=0.1)
    torch.manual_seed(3)
    z = torch.randn(8, 64, requires_grad=True)
    fn(z)["loss"].backward()
    assert z.grad is not None, "gradient must reach input z"
    assert torch.isfinite(z.grad).all(), "gradients must be finite"


# --- self-pair exclusion ---

def test_self_pairs_excluded() -> None:
    """Loss must be > 0 even when self-similarity is the highest in each row."""
    fn = InfoNCELoss(temperature=0.1, normalize_embeddings=False)
    # Without diagonal masking, argmax of each row would be itself → trivial loss ≈ 0.
    z = torch.eye(4, 4) * 10.0
    out = fn(z)
    assert out["loss"].item() > 0, "self-pairs not excluded from denominator"


# --- hand-computed equivalence ---

def test_hand_computed_equivalence() -> None:
    """Pin the loss to a manual logsumexp calculation.

    Setup: B=2, τ=0.5
      view1 = [e1, e2]   (standard basis vectors, already unit-norm)
      view2 = [e1, e2]   (identical copies → perfect positive pairs)
      negatives are orthogonal: cos_sim(view1[i], view1[j≠i]) = 0

    For each of the 4 rows the denominator sum is:
      exp(-inf) [self] + exp(0/τ) [neg1] + exp(1/τ) [pos] + exp(0/τ) [neg2]
      = 0 + 1 + e^(1/τ) + 1 = 2 + e^(1/τ)

    Per-row loss = -(1/τ) + log(2 + e^(1/τ))
    Total loss   = same (all rows identical)

    A sign flip in the implementation would change -(1/τ) to +(1/τ), shifting
    the expected value by 2/τ = 4.0 — well outside the 1e-5 tolerance.
    """
    tau = 0.5
    z = torch.zeros(4, 4)
    z[0, 0] = 1.0  # view1[0] = e1
    z[1, 1] = 1.0  # view1[1] = e2
    z[2, 0] = 1.0  # view2[0] = e1
    z[3, 1] = 1.0  # view2[1] = e2

    fn = InfoNCELoss(temperature=tau, normalize_embeddings=False)
    out = fn(z)

    expected = -(1.0 / tau) + math.log(2.0 + math.exp(1.0 / tau))
    assert abs(out["loss"].item() - expected) < 1e-5, (
        f"loss={out['loss'].item():.6f}, expected={expected:.6f}"
    )


def test_monitoring_sims_are_raw_cosine() -> None:
    """pos_sim_mean and neg_sim_mean must not depend on τ (they are pre-τ cosine sims)."""
    torch.manual_seed(4)
    z = torch.randn(8, 32)
    out_low = InfoNCELoss(temperature=0.07)(z)
    out_high = InfoNCELoss(temperature=0.5)(z)
    # Monitoring values must be τ-independent
    assert torch.allclose(out_low["pos_sim_mean"], out_high["pos_sim_mean"], atol=1e-6)
    assert torch.allclose(out_low["neg_sim_mean"], out_high["neg_sim_mean"], atol=1e-6)
    # But the loss itself must differ
    assert not torch.allclose(out_low["loss"], out_high["loss"])
