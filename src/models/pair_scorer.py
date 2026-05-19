"""MLP and KAN pair scorers: estimate false-negative probability for embedding pairs."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.kan.fastkan import FastKANLayer


class MLPPairScorer(nn.Module):
    """Scores all B^2 embedding pairs and returns [B, B] FN probabilities.

    Takes [B, D] view-1 embeddings, constructs all pairs by concatenating
    z_i and z_j ([z_i; z_j] ∈ R^{2D}), and maps each pair through an MLP
    followed by sigmoid.

    The diagonal is NOT zeroed here — the FN-weighted loss excludes self-pairs
    via its own self_mask. Zeroing here would obscure that p_fn(i,i) is ignored.

    Output is intentionally asymmetric: p_fn[i,k] ≠ p_fn[k,i] in general because
    the MLP sees [z_i; z_k] vs [z_k; z_i] as different inputs. This is valid since
    the contrastive loss evaluates each anchor independently (p_fn[i,k] weights
    anchor-i's view of the pair; p_fn[k,i] weights anchor-k's view). If symmetric
    FN probabilities are needed, symmetrize the output externally:
        p_fn_sym = 0.5 * (p_fn + p_fn.T)

    Args:
        input_dim: embedding dimension D; pair input has size 2D.
        hidden_dim: hidden layer width.
        num_layers: total depth (>= 1); depth=1 means linear-then-sigmoid.
        dropout: dropout probability in [0, 1).
        clip_val: output is clamped to [0, clip_val]; 1.0 means no cap beyond [0,1].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not 0.0 < clip_val <= 1.0:
            raise ValueError(f"clip_val must be in (0, 1], got {clip_val}")

        self.input_dim = input_dim
        self.clip_val = clip_val

        pair_dim = 2 * input_dim
        dims = [pair_dim] + [hidden_dim] * (num_layers - 1) + [1]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            is_last = idx == len(dims) - 2
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if not is_last:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Score all B^2 pairs.

        Args:
            z: [B, D] embeddings (typically view-1 only: z_full[:B]).

        Returns:
            p_fn: [B, B] false-negative probabilities in [0, clip_val].
        """
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input [B, D], got shape {z.shape}")
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {z.shape[1]}"
            )
        B = z.shape[0]
        zi = z.unsqueeze(1).expand(B, B, -1)   # [B, B, D]
        zj = z.unsqueeze(0).expand(B, B, -1)   # [B, B, D]
        pairs = torch.cat([zi, zj], dim=-1)     # [B, B, 2D]
        logits = self.net(pairs.reshape(B * B, -1)).reshape(B, B)  # [B, B]
        p_fn = torch.sigmoid(logits)
        if self.clip_val < 1.0:
            p_fn = p_fn.clamp(max=self.clip_val)
        return p_fn


class KANPairScorer(nn.Module):
    """Scores all B^2 embedding pairs using FastKAN (Gaussian RBF) layers.

    Drop-in replacement for MLPPairScorer: same forward signature, same [B,B]
    output semantics, same clip_val convention. Uses FastKANLayer instead of
    Linear+ReLU. No dropout argument — FastKANLayer has no built-in dropout slot.

    Output is asymmetric by the same reasoning as MLPPairScorer: KAN sees
    [zi;zj] vs [zj;zi] as distinct inputs. Symmetrize externally if needed:
        p_fn_sym = 0.5 * (p_fn + p_fn.T)

    Parameter comparison at D=16, L=2 (Rule 1 reference):
        MLPPairScorer(H=32)          ≈ 1 089 params
        KANPairScorer(H=4, K=8)      ≈ 1 193 params  (~10 % more)

    Args:
        input_dim: embedding dimension D; pair input has size 2D.
        hidden_dim: hidden layer width.
        num_layers: total depth (>= 1); depth=1 = single KAN layer + sigmoid.
        num_centers: RBF grid points per feature.
        grid_min: lower bound of the RBF grid.
        grid_max: upper bound of the RBF grid.
        use_base_linear: include residual linear branch in each KAN layer.
        base_activation: activation for the residual branch ("silu","relu","identity").
        clip_val: output clamped to [0, clip_val]; 1.0 means no cap beyond [0,1].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0.0 < clip_val <= 1.0:
            raise ValueError(f"clip_val must be in (0, 1], got {clip_val}")

        self.input_dim = input_dim
        self.clip_val = clip_val

        pair_dim = 2 * input_dim
        dims = [pair_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList([
            FastKANLayer(
                dims[i],
                dims[i + 1],
                num_centers=num_centers,
                grid_min=grid_min,
                grid_max=grid_max,
                use_base_linear=use_base_linear,
                base_activation=base_activation,
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Score all B^2 pairs.

        Args:
            z: [B, D] embeddings (view-1 only: z_full[:B]).

        Returns:
            p_fn: [B, B] false-negative probabilities in [0, clip_val].
        """
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input [B, D], got shape {z.shape}")
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {z.shape[1]}"
            )
        B = z.shape[0]
        zi = z.unsqueeze(1).expand(B, B, -1)
        zj = z.unsqueeze(0).expand(B, B, -1)
        pairs = torch.cat([zi, zj], dim=-1)   # [B, B, 2D]
        x = pairs.reshape(B * B, -1)           # [B*B, 2D]
        for layer in self.layers:
            x = layer(x)
        logits = x.reshape(B, B)
        p_fn = torch.sigmoid(logits)
        if self.clip_val < 1.0:
            p_fn = p_fn.clamp(max=self.clip_val)
        return p_fn
