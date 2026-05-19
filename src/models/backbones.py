"""Torchvision image backbones used by contrastive pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class TorchvisionBackbone(nn.Module):
    """Feature extractor wrapper for supported torchvision backbones."""

    def __init__(
        self,
        name: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if name != "resnet18":
            raise ValueError(f"Unsupported torchvision backbone '{name}'.")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.output_dim = model.fc.in_features
        self.in_channels = in_channels
        self.name = name

        if in_channels != model.conv1.in_channels:
            model.conv1 = self._adapt_input_conv(model.conv1, in_channels, pretrained)
        model.fc = nn.Identity()
        self.model = model

    @staticmethod
    def _adapt_input_conv(
        conv: nn.Conv2d, in_channels: int, pretrained: bool
    ) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
                elif in_channels < conv.in_channels:
                    new_conv.weight.copy_(conv.weight[:, :in_channels])
                else:
                    new_conv.weight[:, : conv.in_channels].copy_(conv.weight)
                    mean_weight = conv.weight.mean(dim=1, keepdim=True)
                    repeats = in_channels - conv.in_channels
                    new_conv.weight[:, conv.in_channels :].copy_(
                        mean_weight.repeat(1, repeats, 1, 1)
                    )
                if conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(conv.bias)
        return new_conv

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected image tensor [B, C, H, W], got {images.shape}.")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {images.shape[1]}."
            )
        return self.model(images)


class ResNet18Backbone(TorchvisionBackbone):
    """Convenience alias for the Stage 2 baseline backbone."""

    def __init__(self, in_channels: int = 1, pretrained: bool = False) -> None:
        super().__init__(
            name="resnet18",
            in_channels=in_channels,
            pretrained=pretrained,
        )
