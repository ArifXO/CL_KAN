"""Representation geometry metrics and optional regularizers."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


@torch.no_grad()
def alignment(
    embeddings: torch.Tensor, positives: torch.Tensor | None = None, alpha: float = 2.0
) -> torch.Tensor:
    positives = None if positives is None else positives.detach()
    return _alignment_value(embeddings.detach(), positives, alpha)


@torch.no_grad()
def uniformity(embeddings: torch.Tensor, t: float = 2.0, eps: float = 1e-12) -> torch.Tensor:
    return _uniformity_value(embeddings.detach(), t=t, eps=eps)


@torch.no_grad()
def effective_rank(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    z = _validate_embeddings(embeddings.detach())
    singular_values = torch.linalg.svdvals(z - z.mean(dim=0, keepdim=True))
    total = singular_values.sum()
    if total <= eps:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    probs = singular_values / total.clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return torch.exp(entropy)


@torch.no_grad()
def per_dimension_variance(embeddings: torch.Tensor) -> torch.Tensor:
    return _validate_embeddings(embeddings.detach()).var(dim=0, unbiased=False)


@torch.no_grad()
def covariance_off_diagonal_energy(embeddings: torch.Tensor) -> torch.Tensor:
    return _covariance_off_diagonal_value(embeddings.detach())


@torch.no_grad()
def embedding_norm_statistics(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    z = _validate_embeddings(embeddings.detach())
    norms = z.norm(dim=-1)
    return {
        "norm_mean": norms.mean(),
        "norm_std": norms.std(unbiased=False),
        "norm_min": norms.min(),
        "norm_max": norms.max(),
    }


def compute_geometry_metrics(
    embeddings: torch.Tensor,
    positives: torch.Tensor | None = None,
    uniformity_t: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute no-grad geometry metrics for a [B, D] embedding tensor."""

    variances = per_dimension_variance(embeddings)
    metrics = {
        "alignment": alignment(embeddings, positives=positives),
        "uniformity": uniformity(embeddings, t=uniformity_t, eps=eps),
        "effective_rank": effective_rank(embeddings, eps=eps),
        "per_dim_variance": variances,
        "variance_mean": variances.mean(),
        "variance_min": variances.min(),
        "variance_max": variances.max(),
        "covariance_offdiag_energy": covariance_off_diagonal_energy(embeddings),
    }
    metrics.update(embedding_norm_statistics(embeddings))
    return metrics


def geometry_regularizers(
    embeddings: torch.Tensor,
    positives: torch.Tensor | None = None,
    alignment_weight: float = 0.0,
    uniformity_weight: float = 0.0,
    covariance_weight: float = 0.0,
    variance_weight: float = 0.0,
    alignment_alpha: float = 2.0,
    uniformity_t: float = 2.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Return weighted geometry regularizer components with gradients enabled."""

    z = _validate_embeddings(embeddings)
    if target_std <= 0:
        raise ValueError(f"target_std must be > 0, got {target_std}.")
    for name, value in {
        "alignment_weight": alignment_weight,
        "uniformity_weight": uniformity_weight,
        "covariance_weight": covariance_weight,
        "variance_weight": variance_weight,
    }.items():
        _validate_nonnegative(name, value)

    zero = z.sum() * 0.0
    align = _optional_value(alignment_weight, zero, lambda: _alignment_value(z, positives, alignment_alpha))
    uni = _optional_value(uniformity_weight, zero, lambda: _uniformity_value(z, uniformity_t, eps))
    cov = _optional_value(covariance_weight, zero, lambda: _covariance_off_diagonal_value(z))
    var = _optional_value(variance_weight, zero, lambda: _variance_floor_value(z, target_std, eps))
    total = (
        alignment_weight * align
        + uniformity_weight * uni
        + covariance_weight * cov
        + variance_weight * var
    )
    return {
        "loss": total,
        "alignment_reg": align,
        "uniformity_reg": uni,
        "covariance_reg": cov,
        "variance_reg": var,
    }


def _alignment_value(
    embeddings: torch.Tensor,
    positives: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    left, right = _paired_embeddings(embeddings, positives)
    return (left - right).norm(dim=-1).pow(alpha).mean()


def _uniformity_value(embeddings: torch.Tensor, t: float, eps: float) -> torch.Tensor:
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    z = _validate_embeddings(embeddings)
    if z.shape[0] < 2:
        raise ValueError("uniformity requires at least two embeddings.")
    sq_distances = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-t * sq_distances).mean().clamp_min(eps))


def _covariance_off_diagonal_value(embeddings: torch.Tensor) -> torch.Tensor:
    z = _validate_embeddings(embeddings)
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / z.shape[0]
    off_diag = covariance - torch.diag(torch.diagonal(covariance))
    return off_diag.pow(2).sum() / z.shape[1]


def _variance_floor_value(embeddings: torch.Tensor, target_std: float, eps: float) -> torch.Tensor:
    variances = _validate_embeddings(embeddings).var(dim=0, unbiased=False)
    std = torch.sqrt(variances + eps)
    return F.relu(target_std - std).mean()


def _paired_embeddings(
    embeddings: torch.Tensor,
    positives: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = _validate_embeddings(embeddings)
    if positives is not None:
        p = _validate_embeddings(positives, "positives")
        if p.shape != z.shape:
            raise ValueError(f"positives shape {p.shape} must match embeddings {z.shape}.")
        return z, p
    if z.shape[0] < 2 or z.shape[0] % 2 != 0:
        raise ValueError("alignment requires an even two-view batch [2B, D].")
    half = z.shape[0] // 2
    return z[:half], z[half:]


def _validate_embeddings(embeddings: torch.Tensor, name: str = "embeddings") -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D], got {embeddings.shape}.")
    if embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty batch and feature dimensions.")
    return embeddings


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _optional_value(
    weight: float,
    zero: torch.Tensor,
    factory: Callable[[], torch.Tensor],
) -> torch.Tensor:
    return zero if weight == 0 else factory()
