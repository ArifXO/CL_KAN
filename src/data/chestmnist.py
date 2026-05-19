"""ChestMNIST dataset wrapper for Stage 1 smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.data._chestmnist_utils import (
    CHESTMNIST_LABEL_NAMES,
    SPLIT_SEED_OFFSETS,
    _as_plain_dict,
    _extract_patient_ids,
    _to_image_tensor,
    _to_label_tensor,
    _validate_positive_int,
    _validate_split,
)


class ChestMNISTDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """Return ChestMNIST samples as image tensor, label tensor, and metadata."""

    label_names = CHESTMNIST_LABEL_NAMES

    def __init__(
        self,
        split: str,
        image_size: int,
        root: str | Path = "data/medmnist",
        download: bool = False,
        fake_data: bool = False,
        fake_size: int = 16,
        seed: int = 42,
        patient_id_col: str = "patient_id",
    ) -> None:
        self.split = _validate_split(split)
        self.image_size = _validate_positive_int(image_size, "image_size")
        self.root = Path(root)
        self.download = download
        self.fake_data = fake_data
        self.fake_size = _validate_positive_int(fake_size, "fake_size")
        self.seed = seed
        self.patient_id_col = patient_id_col

        self._dataset: Any | None = None
        self._fake_images: torch.Tensor | None = None
        self._fake_labels: torch.Tensor | None = None
        self.patient_ids: list[str] | None = None

        if self.fake_data:
            self._init_fake_data()
        else:
            self._init_medmnist()

    def __len__(self) -> int:
        if self.fake_data:
            return self.fake_size
        if self._dataset is None:
            raise ValueError("ChestMNISTDataset was not initialized correctly.")
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for split '{self.split}'.")

        if self.fake_data:
            if self._fake_images is None or self._fake_labels is None:
                raise ValueError("Fake ChestMNIST tensors were not initialized.")
            image = self._fake_images[index]
            label = self._fake_labels[index]
        else:
            if self._dataset is None:
                raise ValueError("MedMNIST dataset was not initialized.")
            raw_image, raw_label = self._dataset[index]
            image = _to_image_tensor(raw_image, self.image_size)
            label = _to_label_tensor(raw_label, len(self.label_names))

        metadata = {
            "index": index,
            "split": self.split,
            "source": "fake" if self.fake_data else "medmnist",
            "patient_id": self.patient_ids[index] if self.patient_ids else None,
        }
        return image, label, metadata

    def _init_fake_data(self) -> None:
        generator = torch.Generator().manual_seed(
            self.seed + SPLIT_SEED_OFFSETS[self.split]
        )
        image_shape = (self.fake_size, 1, self.image_size, self.image_size)
        self._fake_images = torch.rand(image_shape, generator=generator)
        self._fake_labels = torch.randint(
            0,
            2,
            (self.fake_size, len(self.label_names)),
            generator=generator,
            dtype=torch.int64,
        ).float()
        self.patient_ids = [
            f"fake-{self.split}-{sample_idx:06d}" for sample_idx in range(self.fake_size)
        ]

    def _init_medmnist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            from medmnist import ChestMNIST
        except ImportError as exc:
            raise ImportError(
                "medmnist is required for ChestMNISTDataset(fake_data=False). "
                "Install dependencies or set fake_data=True for offline tests."
            ) from exc

        self._dataset = ChestMNIST(
            split=self.split,
            root=str(self.root),
            download=self.download,
            as_rgb=False,
        )
        self.patient_ids = _extract_patient_ids(self._dataset, self.patient_id_col)


class ChestMNISTDataModule:
    """Small DataLoader factory used by Hydra configs and smoke scripts."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        batch_size: int = 32,
        root: str | Path = "data/medmnist",
        download: bool = False,
        fake_data: bool = False,
        fake_size: int = 16,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        patient_id_col: str = "patient_id",
    ) -> None:
        self.split = _validate_split(split)
        self.image_size = _validate_positive_int(image_size, "image_size")
        self.batch_size = _validate_positive_int(batch_size, "batch_size")
        self.root = root
        self.download = download
        self.fake_data = fake_data
        self.fake_size = _validate_positive_int(fake_size, "fake_size")
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patient_id_col = patient_id_col
        self.label_names = CHESTMNIST_LABEL_NAMES

    def dataset(self, split: str | None = None) -> ChestMNISTDataset:
        selected_split = self.split if split is None else split
        return ChestMNISTDataset(
            split=selected_split,
            image_size=self.image_size,
            root=self.root,
            download=self.download,
            fake_data=self.fake_data,
            fake_size=self.fake_size,
            seed=self.seed,
            patient_id_col=self.patient_id_col,
        )

    def dataloader(
        self, split: str | None = None, shuffle: bool | None = None
    ) -> DataLoader:
        selected_split = self.split if split is None else _validate_split(split)
        should_shuffle = selected_split == "train" if shuffle is None else shuffle
        return DataLoader(
            self.dataset(selected_split),
            batch_size=self.batch_size,
            shuffle=should_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def get_dataloader(cfg: Any, split: str | None = None) -> DataLoader:
    """Build a ChestMNIST DataLoader from a dict-like config."""

    cfg_dict = _as_plain_dict(cfg)
    cfg_dict.pop("_target_", None)
    data_module = ChestMNISTDataModule(**cfg_dict)
    return data_module.dataloader(split=split)
