"""Tests for multi-label overlap utilities (Rule 3: positive, negative, FN cases)."""

import pytest
import torch

from src.losses.multilabel_overlap import (
    build_multilabel_positive_mask,
    expand_positive_mask_to_two_view,
    jaccard_overlap_matrix,
    label_overlap_matrix,
)


# --- label_overlap_matrix ---

def test_label_overlap_shape() -> None:
    y = torch.randint(0, 2, (5, 4)).float()
    assert label_overlap_matrix(y).shape == (5, 5)


def test_label_overlap_all_same_labels() -> None:
    y = torch.ones(3, 4)
    overlap = label_overlap_matrix(y)
    assert (overlap == 4.0).all(), "all samples share all 4 labels"


def test_label_overlap_no_shared_labels() -> None:
    y = torch.eye(4)  # each sample has a unique single label
    overlap = label_overlap_matrix(y)
    assert (overlap.fill_diagonal_(0) == 0).all(), "disjoint labels → zero off-diagonal"


def test_label_overlap_known_value() -> None:
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    overlap = label_overlap_matrix(y)
    assert overlap[0, 1].item() == pytest.approx(1.0), "exactly 1 shared label"
    assert overlap[0, 0].item() == pytest.approx(2.0), "diagonal = self-label count"


def test_label_overlap_invalid_input_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        label_overlap_matrix(torch.randn(3, 4, 5))


# --- jaccard_overlap_matrix ---

def test_jaccard_shape() -> None:
    y = torch.randint(0, 2, (6, 5)).float()
    assert jaccard_overlap_matrix(y).shape == (6, 6)


def test_jaccard_diagonal_is_one() -> None:
    y = torch.randint(0, 2, (4, 3)).float()
    # Rows with at least one label: diagonal = 1.0
    row_any = y.sum(dim=-1) > 0
    j = jaccard_overlap_matrix(y)
    for i in range(4):
        if row_any[i]:
            assert j[i, i].item() == pytest.approx(1.0), f"diagonal[{i}] must be 1"


def test_jaccard_disjoint_is_zero() -> None:
    y = torch.eye(4)
    j = jaccard_overlap_matrix(y)
    off_diag = j.clone().fill_diagonal_(0)
    assert (off_diag == 0.0).all(), "disjoint labels → zero off-diagonal Jaccard"


def test_jaccard_identical_is_one() -> None:
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    j = jaccard_overlap_matrix(y)
    assert j[0, 1].item() == pytest.approx(1.0)


def test_jaccard_known_value() -> None:
    # y[0] = [1, 1, 0], y[1] = [1, 0, 1]
    # intersection = 1, union = 3, Jaccard = 1/3
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    j = jaccard_overlap_matrix(y)
    assert j[0, 1].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert j[1, 0].item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_jaccard_no_label_row_is_zero_not_nan() -> None:
    """Sample with no active labels must produce 0 overlap, not NaN."""
    y = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    j = jaccard_overlap_matrix(y)
    assert torch.isfinite(j).all(), "Jaccard must be finite even for empty label rows"
    assert j[0, 1].item() == pytest.approx(0.0)
    assert j[0, 0].item() == pytest.approx(0.0)


def test_jaccard_invalid_input_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        jaccard_overlap_matrix(torch.randn(3))


# --- build_multilabel_positive_mask ---

def test_mask_shape() -> None:
    y = torch.randint(0, 2, (5, 3)).float()
    assert build_multilabel_positive_mask(y).shape == (5, 5)


def test_mask_excludes_self() -> None:
    y = torch.ones(4, 3)  # all pairs share all labels
    mask = build_multilabel_positive_mask(y, threshold=0.0)
    assert not mask.diagonal().any(), "self-pairs must never be positive"


def test_mask_positive_case() -> None:
    """Rule 3 — positive case: samples with shared labels above threshold."""
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])  # identical → Jaccard=1.0
    mask = build_multilabel_positive_mask(y, threshold=0.5)
    assert mask[0, 1], "identical label pairs must be positive at any threshold < 1"
    assert mask[1, 0]


def test_mask_negative_case() -> None:
    """Rule 3 — negative case: samples with no shared labels are always negative."""
    y = torch.eye(4)  # each sample has a unique label, all Jaccard = 0
    for thr in [0.0, 0.3, 0.9]:
        mask = build_multilabel_positive_mask(y, threshold=thr)
        off = mask.clone().fill_diagonal_(False)
        assert not off.any(), f"disjoint labels → all negative at threshold={thr}"


def test_mask_false_negative_case() -> None:
    """Rule 3 — FN case: pair shares label (Jaccard > 0) but below threshold.

    y[0] = [1, 1, 0], y[1] = [1, 0, 0] → intersection=1, union=2, Jaccard=0.5.
    With threshold=0.5 the pair is not positive (0.5 is NOT > 0.5).
    At threshold=0.0, same pair IS positive.
    """
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # threshold=0.5: Jaccard(0,1)=0.5 → not > 0.5 → treated as FN
    mask_strict = build_multilabel_positive_mask(y, threshold=0.5)
    assert not mask_strict[0, 1], "Jaccard=0.5 is not > 0.5 → false negative at this threshold"
    assert not mask_strict[1, 0]

    # threshold=0.0: same pair IS positive
    mask_loose = build_multilabel_positive_mask(y, threshold=0.0)
    assert mask_loose[0, 1], "at threshold=0, label-sharing pair must be positive"
    assert mask_loose[1, 0]


def test_mask_no_positives_row() -> None:
    """A sample with a unique label set has no label-based positives."""
    y = torch.eye(4)  # each sample has exactly one unique label
    mask = build_multilabel_positive_mask(y, threshold=0.0)
    for i in range(4):
        assert mask[i].sum().item() == 0, f"row {i} should have no label positives"


def test_mask_invalid_threshold_raises() -> None:
    y = torch.ones(3, 2)
    with pytest.raises(ValueError, match="threshold"):
        build_multilabel_positive_mask(y, threshold=1.0)
    with pytest.raises(ValueError, match="threshold"):
        build_multilabel_positive_mask(y, threshold=-0.1)


# --- expand_positive_mask_to_two_view ---

def test_expand_shape() -> None:
    mask = torch.zeros(4, 4, dtype=torch.bool)
    assert expand_positive_mask_to_two_view(mask).shape == (8, 8)


def test_expand_all_quadrants_match() -> None:
    """All four [B, B] quadrants of the [2B, 2B] output must equal mask_bb."""
    torch.manual_seed(0)
    B = 4
    mask_bb = torch.randint(0, 2, (B, B)).bool()
    mask_bb.fill_diagonal_(False)
    out = expand_positive_mask_to_two_view(mask_bb)

    assert (out[:B, :B] == mask_bb).all(), "top-left quadrant mismatch"
    assert (out[:B, B:] == mask_bb).all(), "top-right quadrant mismatch"
    assert (out[B:, :B] == mask_bb).all(), "bottom-left quadrant mismatch"
    assert (out[B:, B:] == mask_bb).all(), "bottom-right quadrant mismatch"


def test_expand_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        expand_positive_mask_to_two_view(torch.zeros(3, 4, dtype=torch.bool))
