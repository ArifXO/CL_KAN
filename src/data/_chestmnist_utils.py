"""Private utilities for the ChestMNIST Stage 1 loader."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CHESTMNIST_LABEL_NAMES: tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural",
    "hernia",
)

VALID_SPLITS = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 10_000, "test": 20_000}


def _validate_split(split: str) -> str:
    if split not in VALID_SPLITS:
        valid = ", ".join(VALID_SPLITS)
        raise ValueError(f"Invalid ChestMNIST split '{split}'. Expected one of: {valid}.")
    return split


def _validate_positive_int(value: int, name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return int(value)


def _to_image_tensor(image: Any, image_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(image))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported ChestMNIST image shape: {tuple(tensor.shape)}.")

    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    if tensor.shape[1:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def _to_label_tensor(label: Any, expected_labels: int) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(label), dtype=torch.float32).flatten()
    if tensor.numel() != expected_labels:
        raise ValueError(
            f"Expected {expected_labels} ChestMNIST labels, got {tensor.numel()}."
        )
    return tensor


def _extract_patient_ids(dataset: Any, patient_id_col: str) -> list[str] | None:
    for attr_name in (patient_id_col, "patient_ids"):
        if hasattr(dataset, attr_name):
            values = getattr(dataset, attr_name)
            if values is not None:
                return [str(value) for value in values]
    return None


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "items"):
        return {str(key): value for key, value in cfg.items()}
    raise ValueError("ChestMNIST config must be dict-like.")
