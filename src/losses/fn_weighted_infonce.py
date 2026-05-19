"""False-negative-weighted InfoNCE loss.

Formula (per anchor i, positive p, negatives k):
    L_i = −sim(i,p)/τ + log[ exp(sim(i,p)/τ) + Σ_{k∈neg(i)} (1−p_fn(i,k)) exp(sim(i,k)/τ) ]

Setting p_fn = 0 everywhere recovers standard InfoNCE exactly (Rule 2 baseline parity).

For self-supervised (no FN correction) mode, use InfoNCELoss instead (Rule 9).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.masks import build_positive_mask


class FNWeightedInfoNCELoss(nn.Module):
    """False-negative-weighted InfoNCE for two-view batches.

    Positive pairs: augmentation pairs only (rows 0..B-1 paired with B..2B-1).
    Negative pairs: all remaining off-diagonal pairs, downweighted by (1 − p_fn).
    p_fn = 0 → standard InfoNCE; p_fn = 1 → negative fully removed from denominator.

    Returns dict (Rule 7):
        loss                  — scalar for .backward()
        pos_sim_mean          — mean cosine sim of positive pairs (τ-independent)
        neg_sim_mean          — mean cosine sim of negative pairs (τ-independent)
        p_fn_mean             — mean p_fn over negative pairs only
        p_fn_max              — max p_fn over negative pairs only
        downweighted_fraction — fraction of negative pairs where p_fn > 0.5
        temperature           — τ used

    Args:
        temperature: τ > 0.
        normalize_embeddings: L2-normalize z before similarity computation.
        max_fn_weight: cap applied to valid p_fn values before use; must be in (0, 1].
            1.0 = negatives may be fully removed (p_fn=1 → weight=0).
            0.5 = negatives are at most half-downweighted regardless of scorer output.
            This is a semantic cap on correction strength, not input validation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize_embeddings: bool = True,
        max_fn_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0.0 < max_fn_weight <= 1.0):
            raise ValueError(
                f"max_fn_weight must be in (0, 1], got {max_fn_weight}"
            )
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.max_fn_weight = max_fn_weight

    def forward(
        self, z: torch.Tensor, p_fn: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute FN-weighted InfoNCE.

        Args:
            z: [2B, D] embeddings — rows 0..B-1 = view 1, rows B..2B-1 = view 2.
            p_fn: [B, B] false-negative probabilities in [0, 1]. Must not contain NaN.
                  p_fn[i, k] is the FN probability for the (i, k) image pair.
                  The diagonal (self-pairs) is ignored by the loss but must be in [0, 1].
                  Typically: MLPPairScorer(z[:B]).

        Returns:
            dict with keys: loss, pos_sim_mean, neg_sim_mean, p_fn_mean,
                            p_fn_max, downweighted_fraction, temperature.
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
        if p_fn.ndim != 2:
            raise ValueError(f"Expected 2D p_fn [B, B], got shape {p_fn.shape}")
        if p_fn.shape != (B, B):
            raise ValueError(
                f"p_fn shape {tuple(p_fn.shape)} must be ({B}, {B}) to match B={B}"
            )
        if torch.isnan(p_fn).any():
            raise ValueError("p_fn contains NaN values")
        if (p_fn < 0.0).any() or (p_fn > 1.0).any():
            raise ValueError(
                f"p_fn values must be in [0, 1], got range "
                f"[{p_fn.min().item():.4f}, {p_fn.max().item():.4f}]"
            )

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)

        cos_sim = z @ z.T  # [2B, 2B] — τ-independent, for monitoring
        sim = cos_sim / self.temperature  # [2B, 2B]

        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        pos_mask = build_positive_mask(B).to(z.device)  # [2B, 2B]
        neg_mask = ~pos_mask & ~self_mask               # [2B, 2B]

        # Expand p_fn to [2B, 2B]: both views of image-pair (i, k) share the same FN prob
        p_fn_2b = p_fn.to(z.device).repeat(2, 2).clamp(max=self.max_fn_weight)

        # Weight matrix: 0 → excluded, 1 → fully included, (1-p_fn) → down-weighted
        weight = 1.0 - p_fn_2b          # all entries start as 1 − p_fn
        weight = weight.masked_fill(self_mask, 0.0)   # self → 0
        weight = weight.masked_fill(pos_mask, 1.0)    # positives → 1

        # Log-weight: −∞ for w ≤ 0 (their exp contribution is 0).
        # clamp(min=1e-10) guards against log(0) NaN in autograd: torch.where
        # evaluates both branches before selecting, so without the clamp the
        # backward through the TRUE branch would produce NaN gradients even
        # though the forward value is correct. The FALSE branch still returns −∞.
        log_weight = torch.where(
            weight > 0,
            torch.log(weight.clamp(min=1e-10)),
            torch.full_like(weight, float("-inf")),
        )

        # Stable log denominator: log(Σ_k w[i,k] exp(sim[i,k]/τ))
        log_denom = torch.logsumexp(sim + log_weight, dim=-1)  # [2B]

        # τ-scaled positive similarity for each anchor (exactly one per anchor)
        pos_sim_scaled = (sim * pos_mask.float()).sum(dim=-1)  # [2B]

        loss = (log_denom - pos_sim_scaled).mean()

        # Monitoring: τ-independent cosine similarities
        pos_sim_mean = cos_sim[pos_mask].mean().detach()
        neg_sim_mean = (
            cos_sim[neg_mask].mean().detach()
            if neg_mask.any()
            else z.new_tensor(0.0)
        )

        # FN diagnostics (negative pairs only — self and positives excluded)
        p_fn_neg = p_fn_2b[neg_mask]
        if p_fn_neg.numel() > 0:
            p_fn_mean = p_fn_neg.mean().detach()
            p_fn_max = p_fn_neg.max().detach()
            downweighted_fraction = (p_fn_neg > 0.5).float().mean().detach()
        else:
            p_fn_mean = z.new_tensor(0.0)
            p_fn_max = z.new_tensor(0.0)
            downweighted_fraction = z.new_tensor(0.0)

        return {
            "loss": loss,
            "pos_sim_mean": pos_sim_mean,
            "neg_sim_mean": neg_sim_mean,
            "p_fn_mean": p_fn_mean,
            "p_fn_max": p_fn_max,
            "downweighted_fraction": downweighted_fraction,
            "temperature": z.new_tensor(self.temperature),
        }
