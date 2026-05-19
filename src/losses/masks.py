"""Two-view SimCLR contrastive mask construction."""

from __future__ import annotations

import torch


def build_positive_mask(batch_size: int) -> torch.Tensor:
    """[2B, 2B] bool mask — True at (i, i+B) and (i+B, i) (SimCLR positive pairs).

    No diagonal entries are ever True.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    N = 2 * batch_size
    off_diag = torch.eye(batch_size, dtype=torch.bool)
    mask = torch.zeros(N, N, dtype=torch.bool)
    mask[:batch_size, batch_size:] = off_diag
    mask[batch_size:, :batch_size] = off_diag
    return mask


def build_self_mask(n: int) -> torch.Tensor:
    """[N, N] bool mask — True on the diagonal (self-pairs to exclude)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return torch.eye(n, dtype=torch.bool)
