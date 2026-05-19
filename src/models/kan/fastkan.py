"""FastKAN projection layers using Gaussian RBF basis functions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_BASE_ACTIVATIONS = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class FastKANLayer(nn.Module):
    """Linear KAN layer with per-feature Gaussian RBF expansions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        bias: bool = True,
        rbf_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_dims(input_dim, output_dim, num_centers, grid_min, grid_max)
        if rbf_init_scale <= 0:
            raise ValueError(f"rbf_init_scale must be > 0, got {rbf_init_scale}.")
        if base_activation not in _BASE_ACTIVATIONS:
            raise ValueError(
                f"Unsupported base_activation '{base_activation}'. "
                f"Expected one of {sorted(_BASE_ACTIVATIONS)}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.use_base_linear = use_base_linear

        centers = torch.linspace(grid_min, grid_max, num_centers)
        bandwidth = (grid_max - grid_min) / (num_centers - 1)
        self.register_buffer("centers", centers)
        self.register_buffer("bandwidth", torch.tensor(float(bandwidth)))

        self.rbf_weight = nn.Parameter(torch.empty(output_dim, input_dim, num_centers))
        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None
        self.base_activation = _BASE_ACTIVATIONS[base_activation]()
        self.base_linear = (
            nn.Linear(input_dim, output_dim, bias=False) if use_base_linear else None
        )
        self.reset_parameters(rbf_init_scale)

    def reset_parameters(self, rbf_init_scale: float = 0.1) -> None:
        std = rbf_init_scale / math.sqrt(self.input_dim * self.num_centers)
        nn.init.normal_(self.rbf_weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.base_linear is not None:
            nn.init.xavier_uniform_(self.base_linear.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"Expected input tensor [B, D], got {inputs.shape}.")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs.shape[1]}."
            )

        basis = self._rbf_basis(inputs)
        outputs = torch.einsum("bic,oic->bo", basis, self.rbf_weight)
        if self.base_linear is not None:
            outputs = outputs + self.base_linear(self.base_activation(inputs))
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs

    def _rbf_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = (inputs.unsqueeze(-1) - self.centers) / self.bandwidth.clamp_min(1e-6)
        return torch.exp(-(scaled**2))


class FastKANProjector(nn.Module):
    """Stacked FastKAN projection head with optional L2 normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_centers: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        use_base_linear: bool = True,
        base_activation: str = "silu",
        normalize: bool = True,
        rbf_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_projector(input_dim, hidden_dim, output_dim, num_layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            FastKANLayer(
                dims[idx],
                dims[idx + 1],
                num_centers=num_centers,
                grid_min=grid_min,
                grid_max=grid_max,
                use_base_linear=use_base_linear,
                base_activation=base_activation,
                rbf_init_scale=rbf_init_scale,
            )
            for idx in range(len(dims) - 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected feature tensor [B, D], got {features.shape}.")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {features.shape[1]}."
            )
        outputs = features
        for layer in self.layers:
            outputs = layer(outputs)
        if self.normalize:
            outputs = F.normalize(outputs, dim=-1)
        return outputs


def _validate_dims(
    input_dim: int,
    output_dim: int,
    num_centers: int,
    grid_min: float,
    grid_max: float,
) -> None:
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
    if output_dim < 1:
        raise ValueError(f"output_dim must be >= 1, got {output_dim}.")
    if num_centers < 2:
        raise ValueError(f"num_centers must be >= 2, got {num_centers}.")
    if grid_min >= grid_max:
        raise ValueError(f"grid_min must be < grid_max, got {grid_min} >= {grid_max}.")


def _validate_projector(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> None:
    _validate_dims(input_dim, output_dim, 2, -1.0, 1.0)
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
