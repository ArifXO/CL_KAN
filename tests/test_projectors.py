"""Tests for Stage 2 backbone and MLP projector components."""

import pytest
import torch

from src.models.backbones import ResNet18Backbone, TorchvisionBackbone
from src.models.projectors import ContrastiveModel, MLPProjector


def test_mlp_projector_returns_normalized_embeddings() -> None:
    projector = MLPProjector(input_dim=16, hidden_dim=32, output_dim=8)
    features = torch.randn(4, 16)

    embeddings = projector(features)

    assert embeddings.shape == (4, 8)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_mlp_projector_validates_input_dim() -> None:
    projector = MLPProjector(input_dim=16, hidden_dim=32, output_dim=8)

    with pytest.raises(ValueError, match="Expected input_dim=16"):
        projector(torch.randn(4, 15))


def test_mlp_projector_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        MLPProjector(input_dim=16, hidden_dim=32, output_dim=8, activation="tanh")


def test_torchvision_resnet18_backbone_accepts_single_channel_images() -> None:
    backbone = TorchvisionBackbone(name="resnet18", in_channels=1, pretrained=False)
    backbone.eval()

    with torch.no_grad():
        features = backbone(torch.randn(2, 1, 32, 32))

    assert features.shape == (2, 512)
    assert backbone.output_dim == 512


def test_resnet18_backbone_alias_uses_expected_output_dim() -> None:
    backbone = ResNet18Backbone(in_channels=1, pretrained=False)

    assert backbone.name == "resnet18"
    assert backbone.output_dim == 512


def test_contrastive_model_runs_backbone_and_projector() -> None:
    backbone = TorchvisionBackbone(name="resnet18", in_channels=1, pretrained=False)
    projector = MLPProjector(input_dim=512, hidden_dim=64, output_dim=16)
    model = ContrastiveModel(backbone=backbone, projector=projector)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 1, 32, 32))

    assert embeddings.shape == (2, 16)
    assert torch.allclose(
        embeddings.norm(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_contrastive_model_rejects_backbone_projector_dim_mismatch() -> None:
    backbone = TorchvisionBackbone(name="resnet18", in_channels=1, pretrained=False)
    projector = MLPProjector(input_dim=128, hidden_dim=64, output_dim=16)

    with pytest.raises(ValueError, match="Backbone output_dim must match"):
        ContrastiveModel(backbone=backbone, projector=projector)
