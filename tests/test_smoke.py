"""Stage 0 smoke test: verify the package installs and all sub-packages are importable."""
import importlib
import pytest


SUBPACKAGES = [
    "src",
    "src.data",
    "src.models",
    "src.models.kan",
    "src.losses",
    "src.metrics",
    "src.utils",
]


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_package_importable(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"
