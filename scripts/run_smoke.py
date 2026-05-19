"""Run the Stage 1 ChestMNIST data smoke pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import hydra
    from hydra.utils import instantiate
    from omegaconf import DictConfig
except ImportError as exc:
    raise ImportError(
        "scripts/run_smoke.py requires hydra-core and omegaconf. "
        "Install project dependencies with: pip install -e \".[dev]\""
    ) from exc


@hydra.main(version_base=None, config_path="../configs/data", config_name="chestmnist")
def main(cfg: DictConfig) -> None:
    data_module = instantiate(cfg)
    loader = data_module.dataloader()
    images, labels, metadata = next(iter(loader))

    print(f"split={data_module.split}")
    print(f"images={tuple(images.shape)} dtype={images.dtype}")
    print(f"labels={tuple(labels.shape)} dtype={labels.dtype}")
    print(f"label_names={list(data_module.label_names)}")
    print(f"metadata_keys={sorted(metadata.keys())}")


if __name__ == "__main__":
    main()
