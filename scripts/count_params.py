"""Compare MLP and FastKAN parameter counts for Stage 4 baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.projectors import MLPProjector  # noqa: E402


DEFAULT_MLP_CONFIG = REPO_ROOT / "configs" / "model" / "resnet18_mlp.yaml"
DEFAULT_FASTKAN_CONFIG = REPO_ROOT / "configs" / "model" / "resnet18_fastkan.yaml"


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of parameters in a module."""

    parameters = module.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)


def projector_parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    """Count projector parameters on a projector or ContrastiveModel-like module."""

    projector = getattr(model, "projector", model)
    if not isinstance(projector, nn.Module):
        raise ValueError("model.projector must be a torch.nn.Module.")
    return count_parameters(projector, trainable_only=trainable_only)


def compare_parameter_counts(
    mlp_model: nn.Module,
    fastkan_model: nn.Module,
    trainable_only: bool = False,
) -> dict[str, Any]:
    """Compare total and projector-level parameter counts."""

    mlp_total = count_parameters(mlp_model, trainable_only)
    kan_total = count_parameters(fastkan_model, trainable_only)
    mlp_projector = projector_parameter_count(mlp_model, trainable_only)
    kan_projector = projector_parameter_count(fastkan_model, trainable_only)
    return {
        "trainable_only": trainable_only,
        "mlp": {"total": mlp_total, "projector": mlp_projector},
        "fastkan": {"total": kan_total, "projector": kan_projector},
        "delta": {
            "total": kan_total - mlp_total,
            "projector": kan_projector - mlp_projector,
            "projector_relative": _relative_delta(mlp_projector, kan_projector),
        },
    }


def find_parameter_matched_mlp(
    input_dim: int,
    output_dim: int,
    target_params: int,
    num_layers: int = 2,
    use_batch_norm: bool = True,
    max_hidden_dim: int = 4096,
) -> dict[str, int | float]:
    """Brute-force an MLP hidden width closest to a target projector count."""

    if target_params < 1:
        raise ValueError(f"target_params must be >= 1, got {target_params}.")
    best_hidden = 1
    best_count = _mlp_count(input_dim, best_hidden, output_dim, num_layers, use_batch_norm)
    for hidden_dim in range(2, max_hidden_dim + 1):
        count = _mlp_count(input_dim, hidden_dim, output_dim, num_layers, use_batch_norm)
        if abs(count - target_params) < abs(best_count - target_params):
            best_hidden = hidden_dim
            best_count = count
    return {
        "hidden_dim": best_hidden,
        "parameter_count": best_count,
        "target_params": target_params,
        "relative_delta": _relative_delta(target_params, best_count),
    }


def compare_default_configs(trainable_only: bool = False) -> dict[str, Any]:
    """Load default Stage 2/4 configs and compare MLP vs FastKAN counts."""

    mlp_model = _instantiate_model(DEFAULT_MLP_CONFIG)
    fastkan_model = _instantiate_model(DEFAULT_FASTKAN_CONFIG)
    report = compare_parameter_counts(mlp_model, fastkan_model, trainable_only)
    report["matched_mlp_helper"] = find_parameter_matched_mlp(
        input_dim=fastkan_model.projector.input_dim,
        output_dim=fastkan_model.projector.output_dim,
        target_params=report["fastkan"]["projector"],
    )
    return report


def _instantiate_model(config_path: Path) -> nn.Module:
    cfg = OmegaConf.load(config_path)
    model = instantiate(cfg)
    if not isinstance(model, nn.Module):
        raise ValueError(f"Config {config_path} did not instantiate an nn.Module.")
    return model


def _mlp_count(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    use_batch_norm: bool,
) -> int:
    model = MLPProjector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        use_batch_norm=use_batch_norm,
    )
    return count_parameters(model)


def _relative_delta(reference: int, candidate: int) -> float:
    return abs(candidate - reference) / max(reference, candidate)


def main() -> None:
    print(json.dumps(compare_default_configs(trainable_only=False), indent=2))


if __name__ == "__main__":
    main()
