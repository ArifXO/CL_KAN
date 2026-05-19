"""Projection heads for contrastive representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLPProjector(nn.Module):
    """Configurable MLP projection head with optional L2-normalized output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self._validate_config(input_dim, hidden_dim, output_dim, num_layers, dropout)
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Expected one of {sorted(_ACTIVATIONS)}."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        act_cls = _ACTIVATIONS[activation]

        for idx in range(len(dims) - 1):
            is_last = idx == len(dims) - 2
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if not is_last:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[idx + 1]))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _validate_config(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("output_dim", output_dim),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected feature tensor [B, D], got {features.shape}.")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {features.shape[1]}."
            )
        embeddings = self.net(features)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class ContrastiveModel(nn.Module):
    """Backbone plus projection head used by contrastive baselines."""

    def __init__(self, backbone: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        if not hasattr(projector, "input_dim"):
            raise ValueError("Projector must expose an input_dim attribute.")
        if hasattr(backbone, "output_dim") and backbone.output_dim != projector.input_dim:
            raise ValueError(
                "Backbone output_dim must match projector input_dim: "
                f"{backbone.output_dim} != {projector.input_dim}."
            )
        self.backbone = backbone
        self.projector = projector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return self.projector(features)
