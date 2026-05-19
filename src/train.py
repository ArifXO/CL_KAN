"""Minimal Hydra pretraining loop for the Stage 2 MLP + InfoNCE baseline."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("trainer.device='cuda' was requested, but CUDA is unavailable.")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported trainer.device '{device_name}'.")
    return torch.device(device_name)


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    name = str(cfg.name).lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer '{cfg.name}'.")


def _build_train_loader(data_module: Any, cfg: DictConfig) -> DataLoader:
    split = str(cfg.data.split)
    dataset = data_module.dataset(split)
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=split == "train",
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        drop_last=bool(cfg.trainer.drop_last),
    )


def _augment(images: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    view = images.clone()
    noise_std = float(cfg.noise_std)
    if noise_std > 0:
        view = torch.clamp(view + torch.randn_like(view) * noise_std, 0.0, 1.0)

    flip_prob = float(cfg.horizontal_flip_prob)
    if flip_prob > 0:
        flip_mask = torch.rand(view.shape[0], device=view.device) < flip_prob
        view[flip_mask] = torch.flip(view[flip_mask], dims=[-1])
    return view


def _make_two_views(images: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    return torch.cat([_augment(images, cfg), _augment(images, cfg)], dim=0)


def _count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def _git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not read git hash: {result.stderr.strip()}")
    return result.stdout.strip()


def _save_initial_artifacts(cfg: DictConfig, model: torch.nn.Module, output_dir: Path) -> None:
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(config=resolved, f=output_dir / "config.yaml")
    (output_dir / "git_hash.txt").write_text(_git_hash() + "\n", encoding="utf-8")
    counts = _count_parameters(model)
    (output_dir / "param_count.txt").write_text(
        f"total={counts['total']}\ntrainable={counts['trainable']}\n",
        encoding="utf-8",
    )


def _tensor_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in loss_dict.items():
        if value.ndim != 0:
            raise ValueError(f"Loss metric '{key}' must be scalar, got {value.shape}.")
        metrics[key] = float(value.detach().cpu())
    return metrics


def _write_metrics(output_dir: Path, rows: list[dict[str, float | int]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Training produced no metric rows.")
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "steps": len(rows),
        "last": rows[-1],
        "mean_loss": sum(float(row["loss"]) for row in rows) / len(rows),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _train(cfg: DictConfig) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(cfg.trainer.seed))
    device = _resolve_device(str(cfg.trainer.device))
    data_module = instantiate(cfg.data)
    loader = _build_train_loader(data_module, cfg)
    model = instantiate(cfg.model).to(device)
    loss_fn = instantiate(cfg.loss).to(device)
    optimizer = _build_optimizer(model, cfg.optimizer)
    _save_initial_artifacts(cfg, model, output_dir)

    model.train()
    rows: list[dict[str, float | int]] = []
    global_step = 0
    max_steps = cfg.trainer.get("max_steps")

    for epoch in range(int(cfg.trainer.max_epochs)):
        for images, _labels, _metadata in loader:
            if max_steps is not None and global_step >= int(max_steps):
                break
            if images.shape[0] < 2:
                raise ValueError("InfoNCE training requires batch_size >= 2.")

            views = _make_two_views(images.to(device), cfg.augmentation)
            loss_dict = loss_fn(model(views))
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            scalars = _tensor_metrics(loss_dict)
            rows.append({"epoch": epoch, "step": global_step, **scalars})
            global_step += 1
        if max_steps is not None and global_step >= int(max_steps):
            break

    summary = _write_metrics(output_dir, rows)
    runtime = time.perf_counter() - start
    (output_dir / "runtime_sec.txt").write_text(f"{runtime:.6f}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/smoke_chestmnist_mlp_infonce")
def main(cfg: DictConfig) -> None:
    _train(cfg)


if __name__ == "__main__":
    main()
