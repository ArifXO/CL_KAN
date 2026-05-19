"""Tests for SimCLR positive/self mask construction (Rule 3)."""

import pytest
import torch

from src.losses.masks import build_positive_mask, build_self_mask


# --- build_positive_mask ---

@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_positive_mask_shape(B: int) -> None:
    mask = build_positive_mask(B)
    assert mask.shape == (2 * B, 2 * B)


@pytest.mark.parametrize("B", [2, 4, 8])
def test_positive_mask_no_diagonal(B: int) -> None:
    mask = build_positive_mask(B)
    assert not mask.diagonal().any(), "positive mask must never include self-pairs"


@pytest.mark.parametrize("B", [2, 4, 8])
def test_positive_mask_total_count(B: int) -> None:
    mask = build_positive_mask(B)
    assert mask.sum().item() == 2 * B, "exactly one positive per row (2*B total)"


@pytest.mark.parametrize("B", [2, 4, 8])
def test_positive_mask_correct_pairs(B: int) -> None:
    mask = build_positive_mask(B)
    for i in range(B):
        assert mask[i, i + B].item(), f"expected mask[{i}, {i + B}] = True"
        assert mask[i + B, i].item(), f"expected mask[{i + B}, {i}] = True"
        assert mask[i].sum().item() == 1, f"row {i} must have exactly 1 positive"
        assert mask[i + B].sum().item() == 1, f"row {i + B} must have exactly 1 positive"


def test_positive_mask_no_false_positives() -> None:
    """No True values outside the (i, i+B)/(i+B, i) off-block-diagonal."""
    B = 4
    mask = build_positive_mask(B)
    reference = torch.zeros(2 * B, 2 * B, dtype=torch.bool)
    for i in range(B):
        reference[i, i + B] = True
        reference[i + B, i] = True
    assert (mask == reference).all()


def test_positive_mask_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        build_positive_mask(0)


# --- build_self_mask ---

@pytest.mark.parametrize("n", [2, 8, 16])
def test_self_mask_shape(n: int) -> None:
    mask = build_self_mask(n)
    assert mask.shape == (n, n)


@pytest.mark.parametrize("n", [2, 8, 16])
def test_self_mask_is_identity(n: int) -> None:
    mask = build_self_mask(n)
    assert (mask == torch.eye(n, dtype=torch.bool)).all()


@pytest.mark.parametrize("n", [2, 8, 16])
def test_self_mask_off_diagonal_false(n: int) -> None:
    mask = build_self_mask(n)
    assert (~mask).sum().item() == n * n - n, "only diagonal entries must be True"


def test_self_mask_invalid_n() -> None:
    with pytest.raises(ValueError):
        build_self_mask(0)
