"""Tests for MLP/FastKAN parameter count reporting."""

from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.count_params import (
    compare_default_configs,
    compare_parameter_counts,
    count_parameters,
    find_parameter_matched_mlp,
)
from src.models.kan.fastkan import FastKANProjector
from src.models.projectors import MLPProjector


def test_count_parameters_matches_manual_numel() -> None:
    projector = MLPProjector(input_dim=8, hidden_dim=6, output_dim=4)

    assert count_parameters(projector) == sum(
        param.numel() for param in projector.parameters()
    )


def test_compare_parameter_counts_reports_projector_delta() -> None:
    mlp = MLPProjector(input_dim=8, hidden_dim=6, output_dim=4)
    fastkan = FastKANProjector(
        input_dim=8,
        hidden_dim=3,
        output_dim=4,
        num_centers=4,
    )

    report = compare_parameter_counts(mlp, fastkan)

    assert "projector_relative" in report["delta"]
    assert report["mlp"]["projector"] == count_parameters(mlp)
    assert report["fastkan"]["projector"] == count_parameters(fastkan)


def test_parameter_matched_mlp_helper_finds_close_count() -> None:
    target = count_parameters(
        FastKANProjector(input_dim=16, hidden_dim=5, output_dim=8, num_centers=4)
    )

    match = find_parameter_matched_mlp(
        input_dim=16,
        output_dim=8,
        target_params=target,
        max_hidden_dim=64,
    )

    assert match["hidden_dim"] >= 1
    assert match["relative_delta"] <= 0.10


def test_default_resnet18_fastkan_config_instantiates() -> None:
    cfg = OmegaConf.load("configs/model/resnet18_fastkan.yaml")
    model = instantiate(cfg)

    assert isinstance(model.projector, FastKANProjector)
    assert model.projector.output_dim == 128


def test_default_configs_have_projector_param_parity() -> None:
    report = compare_default_configs()

    assert report["mlp"]["projector"] > 0
    assert report["fastkan"]["projector"] > 0
    assert report["delta"]["projector_relative"] <= 0.05
