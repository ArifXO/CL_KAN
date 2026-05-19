"""Tests for representation geometry metrics."""

import pytest
import torch
from omegaconf import OmegaConf

from src.metrics.geometry import (
    alignment,
    compute_geometry_metrics,
    covariance_off_diagonal_energy,
    effective_rank,
    embedding_norm_statistics,
    geometry_regularizers,
    per_dimension_variance,
    uniformity,
)


def _embeddings(batch_size: int = 8, dim: int = 6) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)


def test_alignment_is_finite_for_two_view_batch() -> None:
    z = _embeddings()

    value = alignment(z)

    assert value.shape == torch.Size([])
    assert torch.isfinite(value)


def test_alignment_accepts_explicit_positive_embeddings() -> None:
    z = _embeddings(batch_size=4)
    positives = z + 0.01 * torch.randn_like(z)

    value = alignment(z, positives=positives)

    assert value.shape == torch.Size([])
    assert torch.isfinite(value)


def test_uniformity_is_finite() -> None:
    value = uniformity(_embeddings(), t=2.0)

    assert value.shape == torch.Size([])
    assert torch.isfinite(value)


def test_effective_rank_is_finite() -> None:
    value = effective_rank(_embeddings())

    assert value.shape == torch.Size([])
    assert torch.isfinite(value)
    assert value >= 0


def test_per_dimension_variance_shape_and_finite_values() -> None:
    variances = per_dimension_variance(_embeddings(batch_size=5, dim=7))

    assert variances.shape == (7,)
    assert torch.isfinite(variances).all()
    assert (variances >= 0).all()


def test_covariance_off_diagonal_energy_is_finite() -> None:
    value = covariance_off_diagonal_energy(_embeddings())

    assert value.shape == torch.Size([])
    assert torch.isfinite(value)
    assert value >= 0


def test_embedding_norm_statistics_are_finite_scalars() -> None:
    stats = embedding_norm_statistics(_embeddings())

    assert set(stats) == {"norm_mean", "norm_std", "norm_min", "norm_max"}
    for value in stats.values():
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)


def test_compute_geometry_metrics_returns_expected_finite_outputs() -> None:
    metrics = compute_geometry_metrics(_embeddings())

    expected = {
        "alignment",
        "uniformity",
        "effective_rank",
        "per_dim_variance",
        "variance_mean",
        "variance_min",
        "variance_max",
        "covariance_offdiag_energy",
        "norm_mean",
        "norm_std",
        "norm_min",
        "norm_max",
    }
    assert set(metrics) == expected
    for value in metrics.values():
        assert torch.isfinite(value).all()
        assert not value.requires_grad


def test_logging_metrics_do_not_require_grad_by_default() -> None:
    z = _embeddings().requires_grad_(True)

    metrics = compute_geometry_metrics(z)

    assert all(not value.requires_grad for value in metrics.values())


def test_geometry_regularizers_return_dict_and_backpropagate() -> None:
    z = _embeddings().requires_grad_(True)

    out = geometry_regularizers(
        z,
        alignment_weight=0.1,
        uniformity_weight=0.1,
        covariance_weight=0.1,
        variance_weight=0.1,
    )

    assert set(out) == {
        "loss",
        "alignment_reg",
        "uniformity_reg",
        "covariance_reg",
        "variance_reg",
    }
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_geometry_regularizers_zero_weights_skip_pair_requirement() -> None:
    z = _embeddings(batch_size=3).requires_grad_(True)

    out = geometry_regularizers(z)

    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert z.grad is not None


def test_geometry_metrics_validate_shapes_and_pairs() -> None:
    with pytest.raises(ValueError, match="shape"):
        uniformity(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="even two-view"):
        alignment(torch.randn(3, 4))


def test_geometry_regularizer_config_loads() -> None:
    cfg = OmegaConf.load("configs/loss/geometry_regularizers.yaml")

    assert cfg.enabled is False
    assert cfg.weights.alignment == 0.0
    assert cfg.uniformity.t == 2.0
