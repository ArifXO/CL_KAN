"""Supervised Contrastive Loss for multi-label two-view data.

For self-supervised (no-label) mode, use InfoNCELoss instead (Rule 9 — no silent
fallback to different behaviour based on None inputs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.masks import build_positive_mask
from src.losses.multilabel_overlap import (
    build_multilabel_positive_mask,
    expand_positive_mask_to_two_view,
    jaccard_overlap_matrix,
)


class SupConMultilabelLoss(nn.Module):
    """Supervised contrastive loss for multi-label two-view batches.

    Positives: augmentation pairs union label-overlap pairs above overlap_threshold.

    Uses the "out" variant (Khosla et al., 2020):
        L_i = log Z_i  -  (1/|P(i)|) · Σ_{p∈P(i)} sim(i,p)/τ
    where Z_i = Σ_{a≠i} exp(sim(i,a)/τ). Dividing outside the log gives better
    gradient properties than the "in" variant.

    Returns dict of named loss components (Rule 7).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        overlap_threshold: float = 0.0,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0.0 <= overlap_threshold < 1.0):
            raise ValueError(
                f"overlap_threshold must be in [0, 1), got {overlap_threshold}"
            )
        self.temperature = temperature
        self.overlap_threshold = overlap_threshold
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute SupCon loss.

        Args:
            z: [2B, D] embeddings — rows 0..B-1 = view 1, rows B..2B-1 = view 2.
            labels: [B, C] binary multi-label tensor.

        Returns:
            loss, pos_term, neg_term, pos_sim_mean, neg_sim_mean,
            mean_pos_overlap, n_no_positive, temperature.
        """
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input [2B, D], got shape {z.shape}")
        N = z.shape[0]
        if N % 2 != 0:
            raise ValueError(f"Batch size must be even (two views), got {N}")
        B = N // 2
        if B < 2:
            raise ValueError(
                f"Requires at least 2 pairs per view (B >= 2), got B={B}"
            )
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D labels [B, C], got shape {labels.shape}")
        if labels.shape[0] != B:
            raise ValueError(
                f"labels.shape[0]={labels.shape[0]} must match B={B}"
            )

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)

        # Cosine similarity [2B, 2B]; sim is τ-scaled for loss
        cos_sim = z @ z.T
        sim = cos_sim / self.temperature

        # Exclude self-pairs from the softmax denominator
        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        log_denom = torch.logsumexp(
            sim.masked_fill(self_mask, float("-inf")), dim=-1
        )  # [2B]

        # Positive mask: augmentation pairs union label-overlap pairs
        aug_mask = build_positive_mask(B).to(z.device)
        label_mask_bb = build_multilabel_positive_mask(
            labels.to(z.device), threshold=self.overlap_threshold
        )
        label_mask_2b = expand_positive_mask_to_two_view(label_mask_bb)
        pos_mask = aug_mask | label_mask_2b  # [2B, 2B]
        neg_mask = ~pos_mask & ~self_mask

        # Per-anchor counts and τ-scaled positive similarity sums
        pos_count = pos_mask.float().sum(dim=-1)  # [2B]
        no_positive = pos_count == 0
        safe_count = pos_count.clamp(min=1.0)
        pos_sim_sum = (sim * pos_mask.float()).sum(dim=-1)  # [2B]

        # Out-variant: L_i = log_denom[i] − (1/|P(i)|) · pos_sim_sum[i]
        loss_per_anchor = log_denom - pos_sim_sum / safe_count
        loss_per_anchor = loss_per_anchor.masked_fill(no_positive, 0.0)

        n_no_positive = no_positive.long().sum()
        valid = ~no_positive
        if valid.any():
            loss = loss_per_anchor[valid].mean()
            pos_term = (-(pos_sim_sum / safe_count)[valid]).mean().detach()
            # neg_term is the full logsumexp denominator (includes positives), not purely negatives
            neg_term = log_denom[valid].mean().detach()
        else:
            # Structurally unreachable in two-view mode (aug pairs always supply one positive).
            # Kept for safety if pos_mask is externally zeroed.
            loss = z.sum() * 0.0  # grad-safe zero
            pos_term = z.new_tensor(0.0)
            neg_term = z.new_tensor(0.0)

        # Monitoring — τ-independent cosine similarities
        pos_sim_mean = (
            cos_sim[pos_mask].mean().detach()
            if pos_mask.any()
            else z.new_tensor(0.0)
        )
        neg_sim_mean = (
            cos_sim[neg_mask].mean().detach()
            if neg_mask.any()
            else z.new_tensor(0.0)
        )

        # Mean Jaccard of label-based positive pairs (overlap statistics)
        jaccard_2b = jaccard_overlap_matrix(labels.float().to(z.device)).repeat(2, 2)
        mean_pos_overlap = (
            jaccard_2b[label_mask_2b].mean().detach()
            if label_mask_2b.any()
            else z.new_tensor(0.0)
        )

        return {
            "loss": loss,
            "pos_term": pos_term,
            "neg_term": neg_term,
            "pos_sim_mean": pos_sim_mean,
            "neg_sim_mean": neg_sim_mean,
            "mean_pos_overlap": mean_pos_overlap,
            "n_no_positive": n_no_positive,
            "temperature": z.new_tensor(self.temperature),
        }
