"""KAN projection head variants."""

from src.models.kan.fastkan import FastKANLayer, FastKANProjector
from src.models.kan.residual_warp import ResidualFastKANWarp

__all__ = ["FastKANLayer", "FastKANProjector", "ResidualFastKANWarp"]
