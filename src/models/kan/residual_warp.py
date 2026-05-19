"""Residual FastKAN warp: z_tilde = normalize(z + alpha * g_KAN(z))."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kan.fastkan import FastKANProjector


class ResidualFastKANWarp(nn.Module):
    """Residual warp applied to projected embeddings.

    z_tilde = normalize(z + alpha * g_KAN(z))

    When alpha=0, reduces to normalize(z) (identity on unit-norm inputs).
    g_KAN is a FastKANProjector with input_dim == output_dim and no output normalization.
    The warp is applied *after* a projector; it does not change the embedding dimension.
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
        rbf_init_scale: float = 0.1,
        alpha_init: float = 0.0,
        learnable_alpha: bool = True,
        clamp_alpha: bool = True,
        clamp_max: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if alpha_init < 0.0:
            raise ValueError(f"alpha_init must be >= 0, got {alpha_init}.")
        if clamp_max <= 0.0:
            raise ValueError(f"clamp_max must be > 0, got {clamp_max}.")
        if clamp_alpha and alpha_init > clamp_max:
            raise ValueError(
                f"alpha_init={alpha_init} exceeds clamp_max={clamp_max} with "
                "clamp_alpha=True. Set clamp_alpha=False or reduce alpha_init."
            )

        self.input_dim = input_dim
        self.clamp_alpha = clamp_alpha
        self.clamp_max = clamp_max

        # g_KAN: same-dimension perturbation network; output not normalized
        self.kan = FastKANProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            num_centers=num_centers,
            grid_min=grid_min,
            grid_max=grid_max,
            use_base_linear=use_base_linear,
            base_activation=base_activation,
            normalize=False,
            rbf_init_scale=rbf_init_scale,
        )

        alpha_tensor = torch.tensor(float(alpha_init))
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha", alpha_tensor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected 2D input [B, D], got shape {z.shape}.")
        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {z.shape[1]}."
            )
        alpha = self.alpha
        if self.clamp_alpha:
            alpha = alpha.clamp(0.0, self.clamp_max)
        return F.normalize(z + alpha * self.kan(z), dim=-1)
