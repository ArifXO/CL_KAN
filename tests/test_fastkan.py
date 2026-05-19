"""Tests for Stage 4 FastKAN/RBF-KAN projection heads."""

import pytest
import torch

from src.models.kan.fastkan import FastKANLayer, FastKANProjector


def test_fastkan_layer_forward_shape() -> None:
    layer = FastKANLayer(input_dim=4, output_dim=3, num_centers=5)

    output = layer(torch.randn(6, 4))

    assert output.shape == (6, 3)


def test_fastkan_layer_centers_are_non_trainable_grid() -> None:
    layer = FastKANLayer(
        input_dim=4,
        output_dim=3,
        num_centers=5,
        grid_min=-1.0,
        grid_max=1.0,
    )

    assert layer.centers.shape == (5,)
    assert not layer.centers.requires_grad
    assert torch.allclose(layer.centers, torch.linspace(-1.0, 1.0, 5))


def test_fastkan_layer_without_base_linear_branch() -> None:
    layer = FastKANLayer(
        input_dim=4,
        output_dim=3,
        num_centers=4,
        use_base_linear=False,
    )

    assert layer.base_linear is None
    assert layer(torch.randn(2, 4)).shape == (2, 3)


def test_fastkan_projector_returns_normalized_embeddings() -> None:
    projector = FastKANProjector(
        input_dim=8,
        hidden_dim=6,
        output_dim=4,
        num_layers=2,
        num_centers=5,
    )

    embeddings = projector(torch.randn(3, 8))

    assert embeddings.shape == (3, 4)
    assert torch.allclose(
        embeddings.norm(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )


def test_fastkan_projector_gradient_flows() -> None:
    projector = FastKANProjector(
        input_dim=8,
        hidden_dim=6,
        output_dim=4,
        num_layers=2,
        num_centers=5,
    )
    features = torch.randn(3, 8, requires_grad=True)

    projector(features)[:, 0].sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert projector.layers[0].rbf_weight.grad is not None
    assert torch.isfinite(projector.layers[0].rbf_weight.grad).all()


def test_fastkan_projector_validates_input_dim() -> None:
    projector = FastKANProjector(input_dim=8, hidden_dim=6, output_dim=4)

    with pytest.raises(ValueError, match="Expected input_dim=8"):
        projector(torch.randn(3, 7))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_dim": 0, "output_dim": 3}, "input_dim"),
        ({"input_dim": 4, "output_dim": 0}, "output_dim"),
        ({"input_dim": 4, "output_dim": 3, "num_centers": 1}, "num_centers"),
        ({"input_dim": 4, "output_dim": 3, "grid_min": 1.0}, "grid_min"),
        ({"input_dim": 4, "output_dim": 3, "base_activation": "gelu"}, "base_activation"),
    ],
)
def test_fastkan_layer_invalid_config_raises(
    kwargs: dict[str, int | float | str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FastKANLayer(grid_max=1.0, **kwargs)
