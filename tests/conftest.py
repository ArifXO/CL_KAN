"""Shared pytest fixtures for the CXR-KAN-Contrastive test suite."""
import pytest
import torch


@pytest.fixture
def batch_embeddings() -> torch.Tensor:
    """Normalized embeddings for 16 samples, dim=128."""
    torch.manual_seed(42)
    return torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)


@pytest.fixture
def multi_label_targets() -> torch.Tensor:
    """Binary multi-label targets: 16 samples × 14 labels (CheXpert-style)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (16, 14)).float()
