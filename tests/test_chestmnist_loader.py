"""Stage 1 ChestMNIST loader tests using fake data only."""

import pytest
import torch

from src.data.chestmnist import (
    CHESTMNIST_LABEL_NAMES,
    ChestMNISTDataModule,
    ChestMNISTDataset,
    get_dataloader,
)


def test_fake_chestmnist_item_shapes_and_metadata() -> None:
    dataset = ChestMNISTDataset(
        split="train",
        image_size=32,
        fake_data=True,
        fake_size=3,
        seed=7,
    )

    image, label, metadata = dataset[0]

    assert image.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert label.shape == (len(CHESTMNIST_LABEL_NAMES),)
    assert label.dtype == torch.float32
    assert metadata["split"] == "train"
    assert metadata["source"] == "fake"
    assert metadata["patient_id"] == "fake-train-000000"
    assert "atelectasis" in dataset.label_names


def test_fake_chestmnist_splits_have_disjoint_patient_ids() -> None:
    train = ChestMNISTDataset("train", image_size=16, fake_data=True, fake_size=2)
    val = ChestMNISTDataset("val", image_size=16, fake_data=True, fake_size=2)
    test = ChestMNISTDataset("test", image_size=16, fake_data=True, fake_size=2)

    train_ids = set(train.patient_ids or [])
    val_ids = set(val.patient_ids or [])
    test_ids = set(test.patient_ids or [])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_datamodule_collates_fake_batch() -> None:
    data_module = ChestMNISTDataModule(
        split="val",
        image_size=24,
        batch_size=4,
        fake_data=True,
        fake_size=5,
    )

    images, labels, metadata = next(iter(data_module.dataloader(shuffle=False)))

    assert images.shape == (4, 1, 24, 24)
    assert labels.shape == (4, len(data_module.label_names))
    assert metadata["split"] == ["val"] * 4


def test_get_dataloader_accepts_config_dict() -> None:
    loader = get_dataloader(
        {
            "_target_": "src.data.chestmnist.ChestMNISTDataModule",
            "split": "test",
            "image_size": 20,
            "batch_size": 2,
            "fake_data": True,
            "fake_size": 2,
        }
    )

    images, labels, metadata = next(iter(loader))

    assert images.shape == (2, 1, 20, 20)
    assert labels.shape == (2, len(CHESTMNIST_LABEL_NAMES))
    assert metadata["split"] == ["test", "test"]


def test_invalid_split_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid ChestMNIST split"):
        ChestMNISTDataset("dev", image_size=16, fake_data=True)
