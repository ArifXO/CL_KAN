"""Downstream evaluation and representation geometry metrics."""

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

__all__ = [
    "alignment",
    "compute_geometry_metrics",
    "covariance_off_diagonal_energy",
    "effective_rank",
    "embedding_norm_statistics",
    "geometry_regularizers",
    "per_dimension_variance",
    "uniformity",
]
