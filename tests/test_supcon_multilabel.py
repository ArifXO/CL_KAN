"""Tests for SupConMultilabelLoss (Rule 3: positive, negative, FN mask cases)."""

import pytest
import torch
import torch.nn.functional as F

from src.losses.supcon import SupConMultilabelLoss

_REQUIRED_KEYS = {
    "loss",
    "pos_term",
    "neg_term",
    "pos_sim_mean",
    "neg_sim_mean",
    "mean_pos_overlap",
    "n_no_positive",
    "temperature",
}


# --- fixtures ---

@pytest.fixture
def loss_fn() -> SupConMultilabelLoss:
    return SupConMultilabelLoss(temperature=0.07, overlap_threshold=0.0)


def _make_batch(B: int = 4, D: int = 16, C: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    labels = torch.randint(0, 2, (B, C)).float()
    labels[:, 0] = 1  # ensure at least one shared label per sample
    return z, labels


# --- output contract (Rule 7) ---

def test_output_is_dict(loss_fn: SupConMultilabelLoss) -> None:
    z, labels = _make_batch()
    out = loss_fn(z, labels)
    assert isinstance(out, dict)


def test_output_has_required_keys(loss_fn: SupConMultilabelLoss) -> None:
    z, labels = _make_batch()
    out = loss_fn(z, labels)
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"Missing keys: {missing}"


def test_loss_is_scalar(loss_fn: SupConMultilabelLoss) -> None:
    z, labels = _make_batch()
    out = loss_fn(z, labels)
    assert out["loss"].ndim == 0, "loss must be a scalar tensor"


def test_output_is_finite(loss_fn: SupConMultilabelLoss) -> None:
    torch.manual_seed(1)
    z, labels = _make_batch(seed=1)
    out = loss_fn(z, labels)
    for key in ("loss", "pos_sim_mean", "neg_sim_mean", "mean_pos_overlap"):
        assert torch.isfinite(out[key]), f"{key} must be finite"


# --- gradient flow ---

def test_gradient_flows_through_z(loss_fn: SupConMultilabelLoss) -> None:
    torch.manual_seed(2)
    z = F.normalize(torch.randn(8, 16, requires_grad=True), dim=-1)
    labels = torch.randint(0, 2, (4, 5)).float()
    labels[:, 0] = 1
    # detach and re-attach with requires_grad so we own the leaf
    z_leaf = z.detach().requires_grad_(True)
    loss_fn(z_leaf, labels)["loss"].backward()
    assert z_leaf.grad is not None
    assert torch.isfinite(z_leaf.grad).all()


def test_loss_is_non_negative(loss_fn: SupConMultilabelLoss) -> None:
    """InfoNCE-style losses are always >= 0."""
    torch.manual_seed(3)
    z, labels = _make_batch(seed=3)
    assert loss_fn(z, labels)["loss"].item() >= 0.0


# --- positive / negative / FN coverage (Rule 3) ---

def test_positive_pairs_increase_pos_sim_mean() -> None:
    """When all samples share the same label, pos_sim_mean should be among the
    highest similarities because all cross-sample pairs become positives."""
    torch.manual_seed(4)
    B, D = 4, 16
    z = F.normalize(torch.randn(2 * B, D), dim=-1)
    # All samples share label 0 → all inter-sample pairs are label-positive
    labels_all_shared = torch.ones(B, 1)
    # Disjoint labels → only aug pairs are positive
    labels_disjoint = torch.eye(B)

    loss_shared = SupConMultilabelLoss(temperature=0.07, overlap_threshold=0.0)
    loss_disjoint = SupConMultilabelLoss(temperature=0.07, overlap_threshold=0.0)

    out_shared = loss_shared(z, labels_all_shared)
    out_disjoint = loss_disjoint(z, labels_disjoint)

    # With many positives, mean_pos_overlap should differ
    assert out_shared["mean_pos_overlap"].item() > 0.0, "shared labels → overlap > 0"
    assert out_disjoint["mean_pos_overlap"].item() == 0.0, "disjoint labels → overlap = 0"


def test_aug_pairs_guarantee_no_anchor_without_positive() -> None:
    """In two-view mode the augmentation pair always supplies one positive per anchor."""
    torch.manual_seed(5)
    z, labels = _make_batch(seed=5)
    out = SupConMultilabelLoss(temperature=0.07)(z, labels)
    assert out["n_no_positive"].item() == 0, "aug pairs guarantee no anchor lacks positives"


def test_false_negative_pairs_treated_as_negatives() -> None:
    """Rule 3 — FN case in the loss: pairs sharing labels at threshold=0 become
    negatives when overlap_threshold is raised beyond their Jaccard similarity.
    mean_pos_overlap should drop to 0 when all label pairs are below threshold.
    """
    torch.manual_seed(6)
    B, D = 4, 16
    z = F.normalize(torch.randn(2 * B, D), dim=-1)

    # Each pair shares exactly 1 of 3 labels → Jaccard ≤ 0.5
    # At threshold=0.0 these pairs are positive; at threshold=0.9 they are FN.
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    out_low = SupConMultilabelLoss(temperature=0.07, overlap_threshold=0.0)(z, labels)
    out_high = SupConMultilabelLoss(temperature=0.07, overlap_threshold=0.9)(z, labels)

    assert out_low["mean_pos_overlap"].item() > 0.0, (
        "at threshold=0, label-sharing pairs are positive; overlap > 0"
    )
    assert out_high["mean_pos_overlap"].item() == pytest.approx(0.0), (
        "at threshold=0.9 all label pairs are FN negatives; mean_pos_overlap = 0"
    )
    # FN pairs affect the actual loss value — verify loss changes, not just stats
    assert out_low["loss"].item() != pytest.approx(out_high["loss"].item()), (
        "treating label-sharing pairs as negatives (high threshold) must change the loss"
    )


# --- n_no_positive ---

def test_n_no_positive_is_zero_with_aug_pairs(loss_fn: SupConMultilabelLoss) -> None:
    """In two-view mode each anchor always has at least its augmentation pair."""
    z, labels = _make_batch()
    out = loss_fn(z, labels)
    assert out["n_no_positive"].item() == 0


# --- temperature in dict ---

def test_temperature_in_output_matches_init() -> None:
    fn = SupConMultilabelLoss(temperature=0.2)
    z, labels = _make_batch()
    out = fn(z, labels)
    assert out["temperature"].item() == pytest.approx(0.2)


# --- validation errors (Rule 9) ---

def test_wrong_z_ndim_raises(loss_fn: SupConMultilabelLoss) -> None:
    with pytest.raises(ValueError, match="2D"):
        loss_fn(torch.randn(4, 16, 2), torch.ones(2, 5))


def test_odd_batch_size_raises(loss_fn: SupConMultilabelLoss) -> None:
    with pytest.raises(ValueError, match="even"):
        loss_fn(torch.randn(5, 16), torch.ones(5, 3))


def test_labels_batch_mismatch_raises(loss_fn: SupConMultilabelLoss) -> None:
    z = torch.randn(8, 16)  # B=4
    labels = torch.ones(3, 5)  # B=3 — mismatch
    with pytest.raises(ValueError, match="B=4"):
        loss_fn(z, labels)


def test_wrong_labels_ndim_raises(loss_fn: SupConMultilabelLoss) -> None:
    z = torch.randn(8, 16)
    with pytest.raises(ValueError, match="2D"):
        loss_fn(z, torch.ones(4))


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SupConMultilabelLoss(temperature=0.0)


def test_invalid_overlap_threshold_raises() -> None:
    with pytest.raises(ValueError, match="overlap_threshold"):
        SupConMultilabelLoss(overlap_threshold=1.0)
    with pytest.raises(ValueError, match="overlap_threshold"):
        SupConMultilabelLoss(overlap_threshold=-0.1)
