"""Standard InfoNCE (NT-Xent) loss for two-view SimCLR batches.

Input: z [2B, D] — rows 0..B-1 are view-1, rows B..2B-1 are view-2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.masks import build_positive_mask


class InfoNCELoss(nn.Module):
    """Two-view SimCLR InfoNCE loss.

    Returns dict (Rule 7):
        loss          — scalar to call .backward() on
        pos_sim_mean  — mean inner product of positive pairs before τ scaling (= cosine sim iff embeddings are normalized; τ-independent)
        neg_sim_mean  — mean inner product of negative pairs before τ scaling (τ-independent)
        temperature   — τ used (for logging/sweep tracking)
    """

    def __init__(self, temperature: float = 0.07, normalize_embeddings: bool = True) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input [2B, D], got shape {z.shape}")
        N = z.shape[0]
        if N % 2 != 0:
            raise ValueError(f"Batch dimension must be even (two views), got {N}")
        B = N // 2
        if B < 2:
            raise ValueError(
                f"InfoNCELoss requires at least 2 pairs per view (B >= 2), got B={B}. "
                "With B=1 there are no negative pairs and neg_sim_mean is undefined."
            )

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)

        # Raw cosine similarity [2B, 2B] — kept before τ scaling for monitoring
        cos_sim = z @ z.T

        # τ-scaled similarity; diagonal → -inf to exclude self-pairs from denominator
        sim = cos_sim / self.temperature
        diag_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Positive pair labels: row i → column (i+B) % N
        labels = (torch.arange(N, device=z.device) + B) % N

        # Numerically stable InfoNCE via explicit logsumexp (not F.cross_entropy)
        loss = (
            -sim[torch.arange(N, device=z.device), labels]
            + torch.logsumexp(sim, dim=-1)
        ).mean()

        # Monitoring — raw cosine sims are τ-independent, so comparable across sweeps
        pos_mask = build_positive_mask(B).to(z.device)
        neg_mask = ~pos_mask & ~diag_mask

        return {
            "loss": loss,
            "pos_sim_mean": cos_sim[pos_mask].mean().detach(),
            "neg_sim_mean": cos_sim[neg_mask].mean().detach(),
            "temperature": torch.tensor(self.temperature, device=z.device),
        }
