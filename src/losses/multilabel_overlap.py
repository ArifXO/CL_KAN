"""Multi-label overlap utilities for contrastive mask construction.

Provides two overlap measures:
  - label_overlap_matrix: count of shared labels (float-valued)
  - jaccard_overlap_matrix: Jaccard similarity in [0, 1]
"""

from __future__ import annotations

import torch


def label_overlap_matrix(y: torch.Tensor) -> torch.Tensor:
    """[B, B] count of shared labels from binary label matrix y [B, C].

    Entry (i, j) is the number of labels shared between samples i and j.
    """
    if y.ndim != 2:
        raise ValueError(f"Expected 2D label matrix [B, C], got shape {y.shape}")
    y_f = y.float()
    return y_f @ y_f.T  # [B, B]


def jaccard_overlap_matrix(y: torch.Tensor) -> torch.Tensor:
    """[B, B] Jaccard similarity in [0, 1] from binary label matrix y [B, C].

    Jaccard(i, j) = |intersection| / |union|.
    Samples with no active labels receive 0 similarity with all others (not NaN).
    """
    if y.ndim != 2:
        raise ValueError(f"Expected 2D label matrix [B, C], got shape {y.shape}")
    y_f = y.float()
    intersection = y_f @ y_f.T  # [B, B]
    row_sums = y_f.sum(dim=-1)   # [B]
    union = row_sums.unsqueeze(1) + row_sums.unsqueeze(0) - intersection  # [B, B]
    return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))


def build_multilabel_positive_mask(
    y: torch.Tensor,
    threshold: float = 0.0,
    exclude_self: bool = True,
) -> torch.Tensor:
    """[B, B] bool positive mask — True where Jaccard(y[i], y[j]) > threshold.

    threshold=0.0 means any shared label qualifies as positive.
    exclude_self=True ensures the diagonal is always False.
    """
    if not (0.0 <= threshold < 1.0):
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    mask = jaccard_overlap_matrix(y) > threshold
    if exclude_self:
        mask.fill_diagonal_(False)
    return mask


def expand_positive_mask_to_two_view(mask_bb: torch.Tensor) -> torch.Tensor:
    """Expand [B, B] positive mask to [2B, 2B] two-view space.

    Pair (i, j) is positive in all four quadrants if positive in mask_bb.
    Augmentation pairs (i, i+B) are NOT added here — use build_positive_mask().
    """
    if mask_bb.ndim != 2 or mask_bb.shape[0] != mask_bb.shape[1]:
        raise ValueError(
            f"Expected square 2D boolean mask, got shape {mask_bb.shape}"
        )
    B = mask_bb.shape[0]
    N = 2 * B
    out = torch.zeros(N, N, dtype=torch.bool, device=mask_bb.device)
    out[:B, :B] = mask_bb
    out[:B, B:] = mask_bb
    out[B:, :B] = mask_bb
    out[B:, B:] = mask_bb
    return out
